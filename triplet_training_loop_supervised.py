from model_funcs import *
from meta_data_preproccesing import *
from utils import *
from SV import *
from adaptation import *
from evaluation import *
from clustering import *
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from scipy.stats import mode
from torch.optim import AdamW
import wandb
import os
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import gc

logging.basicConfig(level=logging.INFO)

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, list_file, audio_dir):
        self.df = pd.read_csv(list_file, delimiter="\t")
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.audio_dir, row['path'])
        label = row['client_id']
        return path, label

class ClassificationDatasetFromList(Dataset):
    def __init__(self, paths_labels):
        self.paths_labels = paths_labels

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        return self.paths_labels[idx]

    
def ensemble_clustering(embeddings, n_clusters, n_runs=5):
    """
    Perform ensemble clustering using multiple algorithms and seeds.
    Returns consensus labels (majority vote).
    """
    all_labels = []

    # Run KMeans with different seeds
    for seed in range(n_runs):
        km = KMeans(n_clusters=n_clusters, random_state=seed)
        all_labels.append(km.fit_predict(embeddings))

    # Run Spectral Clustering with different seeds
    for seed in range(n_runs):
        sc = SpectralClustering(n_clusters=n_clusters, random_state=seed, affinity='nearest_neighbors', n_neighbors=min(n_clusters, len(embeddings)-1, 4))
        all_labels.append(sc.fit_predict(embeddings))

    # Run Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    all_labels.append(agg.fit_predict(embeddings))

    # Stack and compute consensus (mode across runs)
    all_labels = np.stack(all_labels, axis=1)  # shape: (N, n_ensemble)
    consensus_labels, _ = mode(all_labels, axis=1)
    consensus_labels = consensus_labels.flatten()
    return consensus_labels

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

def train_epoch(speaker_brain, train_loader, config, epoch):
    speaker_brain.on_stage_start(sb.Stage.TRAIN)
    device = speaker_brain.device
    classifier = speaker_brain.hparams.classifier
    ce_loss_fn = speaker_brain.hparams.compute_cost
    cosine_loss_fn  = speaker_brain.hparams.cosine_loss_fn
    triplet_loss_fn = speaker_brain.hparams.triplet_loss
    miner           = miners.TripletMarginMiner(margin=0.8, type_of_triplets="semihard")
    alpha_cos       = speaker_brain.hparams.cosine_weight
    alpha_trip      = speaker_brain.hparams.triplet_weight
    optimizer = speaker_brain.hparams.opt_class(
        list(speaker_brain.modules.embedding_model.parameters()) + list(classifier.parameters())
    )

    total_loss = 0.0
    for paths, numeric_labels in train_loader:
        emb = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths)
        if emb.size(0) == 0:
            continue

        if isinstance(numeric_labels, torch.Tensor):
            labels_tensor = numeric_labels.detach().clone().to(device)
        else:
            labels_tensor = torch.tensor(numeric_labels, device=device)

        # Classification head + CE loss
        logits  = classifier(emb, labels_tensor)
        ce_loss = ce_loss_fn(logits, labels_tensor)

        # CosineEmbeddingLoss against class weights
        class_wt = classifier.weight[labels_tensor]
        cos_tgts = torch.ones(emb.size(0), device=device)
        cos_loss = cosine_loss_fn(emb, class_wt, cos_tgts)

        # Triplet loss
        a_idx, p_idx, n_idx = miner(emb, labels_tensor)
        if a_idx.numel() > 0:
            anc = emb[a_idx]
            pos = emb[p_idx]
            neg = emb[n_idx]
            triplet_loss = triplet_loss_fn(anc, pos, neg)
        else:
            triplet_loss = torch.tensor(0.0, device=device)

        # Combine
        loss = ce_loss + alpha_cos * cos_loss + alpha_trip * triplet_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    wandb.log({"train_loss": avg_loss})

def main(run=None, config_file='/app/config.yaml'):
    config = load_config(config_file)
    speaker_model = load_model_ecapa_from_speechbrain(config)
    embedding_model = speaker_model.mods.embedding_model
    for p in embedding_model.parameters():
        p.requires_grad = True

    df = pd.read_csv(config['train_audio_list_file'], delimiter="\t")
    families = df['family_id'].unique()
    emb_dir = config.get('embedding_output_file', '/app/embeddings_per_family')

    # Use a large enough number for classifier head (max possible clusters)
    num_classes = df['client_id'].nunique()
    classifier_head = AMSoftmaxHead(192, num_classes)
    optimizer = AdamW(
        list(embedding_model.parameters()) + list(classifier_head.parameters()), lr=1e-4, weight_decay=1e-2
    )

    speaker_brain = modelTune(
        modules=speaker_model.mods,
        hparams={
            "compute_cost": nn.CrossEntropyLoss(),
            "opt_class": lambda params: optimizer,
            "classifier": classifier_head,
            "cosine_loss_fn": nn.CosineEmbeddingLoss(margin=0.0, reduction='mean'),
            "triplet_loss": nn.TripletMarginLoss(margin=0.8, p=2, swap=True, reduction="mean"),
            "cosine_weight": 1,
            "triplet_weight": 1,
        },
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

            # Step 2: Cluster using ensemble_clustering from clustering.py
            embeddings_np = load_embeddings(emb_paths)
            n_clusters = valid_df['client_id'].nunique()  # or use a heuristic
            cluster_labels = ensemble_clustering(embeddings_np, n_clusters=n_clusters, n_runs=3)
            label_set = sorted(set(cluster_labels))
            label_map_family = {str(label): idx for idx, label in enumerate(label_set)}

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
            # Optionally, add embedding plot here

            # Evaluate and log for val split
            evaluate_and_log(config, val_df, None, val_embeddings, val_predicted_labels, train=False)
            # Optionally, add embedding plot here

            # Clean up large objects to free memory
            del all_embeddings, all_emb_paths, train_loader
            gc.collect()

    Save_Model_Localy(speaker_brain.mods.embedding_model, config, name="ecapa_epoch_finetuned_per_family.pth")
    if run:
        run.finish()

if __name__ == '__main__':
    main()
