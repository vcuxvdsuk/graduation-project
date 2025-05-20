import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import pandas as pd
import numpy as np
import logging
import wandb
from tqdm import tqdm
from model_funcs import *
from utils import *
from clustering import *
from adaptation import *
from evaluation import *
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from scipy.stats import mode
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

class ClassificationDatasetFromList(Dataset):
    def __init__(self, paths_labels):
        self.paths_labels = paths_labels

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        return self.paths_labels[idx]

def extract_and_save_family_embeddings(speaker_brain, audio_dir, family_df, emb_dir, family_id):
    os.makedirs(emb_dir, exist_ok=True)
    emb_paths = []
    valid_rows = []
    for idx, row in family_df.iterrows():
        audio_path = os.path.join(audio_dir, row['path'])
        emb_path = os.path.join(emb_dir, f"{family_id}_{os.path.splitext(os.path.basename(row['path']))[0]}.npy")
        emb = extract_single_embedding(speaker_brain, audio_path)
        if emb is not None:
            np.save(emb_path, emb.cpu().numpy())
            emb_paths.append(emb_path)
            valid_rows.append(row)
        else:
            logging.warning(f"Failed to extract embedding for {audio_path}. Skipping.")
    return emb_paths, pd.DataFrame(valid_rows)

def load_embeddings(emb_paths):
    embeddings = []
    for emb_path in emb_paths:
        emb = np.load(emb_path)
        emb = emb.flatten()
        embeddings.append(emb)
    return np.stack(embeddings)

def ensemble_clustering(embeddings, n_clusters, n_runs=5):
    all_labels = []
    for seed in range(n_runs):
        km = KMeans(n_clusters=n_clusters, random_state=seed)
        all_labels.append(km.fit_predict(embeddings))
    for seed in range(n_runs):
        sc = SpectralClustering(n_clusters=n_clusters, random_state=seed, affinity='nearest_neighbors', n_neighbors=min(n_clusters, len(embeddings)-1, 4))
        all_labels.append(sc.fit_predict(embeddings))

    agg = AgglomerativeClustering(n_clusters=n_clusters)
    all_labels.append(agg.fit_predict(embeddings))
    all_labels = np.stack(all_labels, axis=1)
    consensus_labels, _ = mode(all_labels, axis=1)
    consensus_labels = consensus_labels.flatten()
    return consensus_labels



def train_epoch(speaker_brain, train_loader, config, epoch):
    speaker_brain.on_stage_start(sb.Stage.TRAIN)
    classifier = speaker_brain.hparams.classifier
    ce_loss = speaker_brain.hparams.compute_cost

    optimizer = speaker_brain.hparams.opt_class(
        list(speaker_brain.modules.embedding_model.parameters()) + list(classifier.parameters())
    )

    total_loss = 0.0
    for paths, numeric_labels in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        embeddings = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths)
        if embeddings.size(0) == 0:
            continue

        if isinstance(numeric_labels, torch.Tensor):
            labels_tensor = numeric_labels
        else:
            labels_tensor = torch.tensor(numeric_labels)
        logits = classifier(embeddings,labels_tensor)
        loss = ce_loss(logits, labels_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    logging.info(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")
    wandb.log({"train_loss": avg_loss})

    
def log_embedding_plot(embeddings, labels, epoch, split="train"):
    """
    Plots embeddings using PCA and logs to wandb.
    Args:
        embeddings (np.ndarray): shape (N, D)
        labels (list or np.ndarray): shape (N,)
        epoch (int): current epoch
        split (str): "train" or "val"
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
    plt.title(f"PCA Embedding Plot - {split} - Epoch {epoch}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Label")
    plt.tight_layout()
    img_path = f"plots/embedding_plot_{split}_epoch_{epoch}.png"
    plt.savefig(img_path)
    plt.close()
    wandb.log({f"{split}_embedding_plot": wandb.Image(img_path)})

def main(run=None, config_file='/app/config.yaml'):
    config = load_config(config_file)
    speaker_model = load_model_ecapa_from_speechbrain(config)
    embedding_model = speaker_model.mods.embedding_model
    for p in embedding_model.parameters():
        p.requires_grad = True

    df = pd.read_csv(config['train_audio_list_file'], delimiter="\t")
    families = df['family_id'].unique()
    emb_dir = config.get('embedding_output_file', '/app/embeddings_per_family')

    label_set = sorted(set(df['client_id']))
    label_map = {label: idx for idx, label in enumerate(label_set)}
    num_classes = len(label_map)

    classifier_head = AMSoftmaxHead(192, num_classes)
    optimizer = AdamW(
        list(embedding_model.parameters()) + list(classifier_head.parameters()), lr=1e-5, weight_decay=1e-1
    )

    speaker_brain = modelTune(
        modules=speaker_model.mods,
        hparams={"compute_cost": nn.CrossEntropyLoss(), 
                 "opt_class": lambda params: optimizer, 
                 "classifier": classifier_head}
    )

    for epoch in range(config['adaptation']['num_epochs']):
        logging.info(f"Epoch {epoch}: Per-family unsupervised adaptation")
        all_train_data = []
        for family_id in families:
            family_df = df[df['family_id'] == family_id]
            if len(family_df) < 2:
                logging.info(f"Skipping family {family_id} (not enough samples)")
                continue

            # Step 1: Extract and save embeddings for this family
            emb_paths, valid_df = extract_and_save_family_embeddings(speaker_brain, config['audio_dir'], family_df, emb_dir, family_id)
            if len(emb_paths) < 2:
                logging.info(f"Skipping family {family_id} (not enough valid embeddings)")
                continue

            # Step 2: Cluster using ensemble clustering
            embeddings_np = load_embeddings(emb_paths)
            n_clusters = valid_df['client_id'].nunique()  # or use a heuristic
            cluster_labels = ensemble_clustering(embeddings_np, n_clusters=n_clusters, n_runs=3)
            label_set = sorted(set(cluster_labels))
            label_map_family = {str(label): idx for idx, label in enumerate(label_set)}
            num_classes_family = len(label_map_family)

            # Step 3: Prepare data for this family
            valid_df = valid_df.reset_index(drop=True)
            valid_df['cluster_label'] = [str(l) for l in cluster_labels]
            train_df, val_df = train_test_split(valid_df, test_size=0.2)
            all_train_data.extend([(p, label_map_family[l]) for p, l in zip(train_df['path'], train_df['cluster_label'])])
        
        # Create a single DataLoader for all families
        train_loader = DataLoader(ClassificationDatasetFromList(all_train_data), batch_size=16, shuffle=True)
        train_epoch(speaker_brain, train_loader, config, epoch)

        # --- After all families, evaluate on the entire dataset (train and val splits) ---
        logging.info(f"Epoch {epoch}: Evaluating on all data (train/val splits)")
        all_emb_paths = []
        for idx, row in df.iterrows():
            emb_path = os.path.join(emb_dir, f"{row['family_id']}_{os.path.splitext(os.path.basename(row['path']))[0]}.npy")
            if os.path.exists(emb_path):
                all_emb_paths.append(emb_path)
        if len(all_emb_paths) > 1:
            all_embeddings = load_embeddings(all_emb_paths)
            # Cluster all embeddings
            n_clusters = df['client_id'].nunique()
            all_cluster_labels = ensemble_clustering(all_embeddings, n_clusters=n_clusters, n_runs=3)
            label_set = sorted(set(all_cluster_labels))
            label_map_all = {str(label): idx for idx, label in enumerate(label_set)}
            all_predicted_labels = [label_map_all[str(l)] for l in all_cluster_labels]

            # Split into train/val using the same split as before
            df = df.reset_index(drop=True)
            train_idx, val_idx = train_test_split(df.index, test_size=0.2, random_state=42)
            train_embeddings = all_embeddings[train_idx]
            val_embeddings = all_embeddings[val_idx]
            train_predicted_labels = [all_predicted_labels[i] for i in train_idx]
            val_predicted_labels = [all_predicted_labels[i] for i in val_idx]
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            # Evaluate and log for train split
            evaluate_and_log(config, train_df, None, train_embeddings, train_predicted_labels, train=True)
            log_embedding_plot(train_embeddings, train_predicted_labels, epoch, split="train_all")

            # Evaluate and log for val split
            evaluate_and_log(config, val_df, None, val_embeddings, val_predicted_labels, train=False)
            log_embedding_plot(val_embeddings, val_predicted_labels, epoch, split="val_all")

    Save_Model_Localy(speaker_brain.mods.embedding_model, config, name="ecapa_epoch_finetuned_per_family.pth")
    if run:
        run.finish()

if __name__ == '__main__':
    main()