import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import pandas as pd
import numpy as np
import logging
import wandb
import shutil
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier
from model_funcs import *
from meta_data_preproccesing import *
from utils import *
from SV import *
from clustering import *
from adaptation import *
from evaluation import *
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from scipy.stats import mode

logging.basicConfig(level=logging.INFO)


def main(run=None, config_file='/app/config.yaml'):
    config = load_config(config_file)

    speaker_model = load_model_ecapa_from_speechbrain(config)
    embedding_model = speaker_model.mods.embedding_model
    for p in embedding_model.parameters():
        p.requires_grad = True

    df = pd.read_csv(config['train_audio_list_file'], delimiter="\t")
    num_speakers = count_unique_speakers(config['train_audio_list_file'])

    classifier_head = AMSoftmaxHead(192, num_speakers)
    optimizer = AdamW(
        list(embedding_model.parameters()) + list(classifier_head.parameters()), lr=1e-4, weight_decay=1e-2
    )

    speaker_brain = modelTune(
        modules=speaker_model.mods,
        hparams={"compute_cost": nn.CrossEntropyLoss(), 
                 "opt_class": lambda params: optimizer, 
                 "classifier": classifier_head}
    )

    emb_dir = config.get('embedding_output_file', '/app/embeddings')

    # ------------------- WARM-UP PHASE -------------------
    num_warmup_epochs = config['adaptation'].get('num_warmup_epochs', 3)
    freeze_module(embedding_model, freeze=True)
    freeze_module(classifier_head, freeze=False)
    logging.info("Warm-up: training classifier only")
    for epoch in range(num_warmup_epochs):
        # Step 1: Extract and save all embeddings to disk using the current model
        valid_files_path = os.path.join(emb_dir, "valid_files.tsv")
        if epoch == 0 or not os.path.exists(valid_files_path):
            emb_paths, valid_df = extract_and_save_embeddings(speaker_brain, config['audio_dir'], df, emb_dir)
            valid_df.to_csv(valid_files_path, sep="\t", index=False)
        else:
            valid_df = pd.read_csv(valid_files_path, sep="\t")
            emb_paths, valid_df = extract_and_save_embeddings(speaker_brain, config['audio_dir'], valid_df, emb_dir)

        # Step 2: Cluster using disk-based embeddings
        clustered_df = apply_spectral_clustering_to_disk_embeddings(valid_df, emb_paths, config)
        label_set = sorted(set(clustered_df['cluster_label']))
        label_map = {label: idx for idx, label in enumerate(label_set)}
        num_classes = len(label_map)

        merged_df = pd.merge(clustered_df, df[['path', 'client_id']], on='path', how='left')
        train_df, _ = train_test_split(merged_df, test_size=0.2)
        train_loader = DataLoader(
            ClassificationDatasetFromList([(p, label_map[l]) for p, l in zip(train_df['path'], train_df['cluster_label'])]),
            batch_size=16, shuffle=True
        )
        train_epoch_warmup(speaker_brain, train_loader, config, epoch)
        del train_loader

    # ------------------- MAIN TRAINING PHASE -------------------
    freeze_module(embedding_model, freeze=False)
    freeze_module(classifier_head, freeze=False)

    for epoch in range(config['adaptation']['num_epochs']):
        valid_files_path = os.path.join(emb_dir, "valid_files.tsv")
        if epoch == 0 or not os.path.exists(valid_files_path):
            emb_paths, valid_df = extract_and_save_embeddings(speaker_brain, config['audio_dir'], df, emb_dir)
            valid_df.to_csv(valid_files_path, sep="\t", index=False)
        else:
            valid_df = pd.read_csv(valid_files_path, sep="\t")
            emb_paths, valid_df = extract_and_save_embeddings(speaker_brain, config['audio_dir'], valid_df, emb_dir)

        clustered_df = apply_spectral_clustering_to_disk_embeddings(valid_df, emb_paths, config)
        label_set = sorted(set(clustered_df['cluster_label']))
        label_map = {label: idx for idx, label in enumerate(label_set)}
        num_classes = len(label_map)

        merged_df = pd.merge(clustered_df, df[['path', 'client_id']], on='path', how='left')
        train_df, val_df = train_test_split(merged_df, test_size=0.2)

        train_loader = DataLoader(
            ClassificationDatasetFromList([(p, label_map[l]) for p, l in zip(train_df['path'], train_df['cluster_label'])]),
            batch_size=16, shuffle=True
        )
        val_loader = DataLoader(
            ClassificationDatasetFromList([(p, label_map[l]) for p, l in zip(val_df['path'], val_df['cluster_label'])]),
            batch_size=16
        )

        logging.info(f"Epoch {epoch}: Training with {num_classes} pseudo-labels")
        if epoch % 1 == 0:
            train_emb_paths = [os.path.join(emb_dir, f"{os.path.splitext(os.path.basename(p))[0]}.npy") for p in train_df['path']]
            train_embeddings = load_embeddings(train_emb_paths)
            train_predicted_labels = [label_map[l] for l in train_df['cluster_label']]
            evaluate_and_log(config, train_df, None, train_embeddings, train_predicted_labels, train=True)
            log_embedding_plot(train_embeddings, train_predicted_labels, epoch, split="train")

            val_emb_paths = [os.path.join(emb_dir, f"{os.path.splitext(os.path.basename(p))[0]}.npy") for p in val_df['path']]
            val_embeddings = load_embeddings(val_emb_paths)
            val_predicted_labels = [label_map[l] for l in val_df['cluster_label']]
            logging.info(f"Epoch {epoch}: Training with {num_classes} pseudo-labels")
            evaluate_and_log(config, val_df, None, val_embeddings, val_predicted_labels, train=False)
            log_embedding_plot(val_embeddings, val_predicted_labels, epoch, split="val")

        train_epoch(speaker_brain, train_loader, config, epoch)

    Save_Model_Localy(speaker_brain.mods.embedding_model, config, name="ecapa_epoch_finetuned.pth")
    if run:
        run.finish()

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

def freeze_module(module, freeze=True):
    for p in module.parameters():
        p.requires_grad = not freeze

def train_epoch_warmup(speaker_brain, train_loader, config, epoch):
    """Train only the classifier head (embedding frozen) with current loss."""
    speaker_brain.on_stage_start(sb.Stage.TRAIN)
    classifier = speaker_brain.hparams.classifier
    ce_loss = speaker_brain.hparams.compute_cost
    optimizer = speaker_brain.hparams.opt_class(list(classifier.parameters()))
    total_loss = 0.0
    for paths, numeric_labels in train_loader:
        embeddings = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths)
        if embeddings.size(0) == 0:
            continue
        if isinstance(numeric_labels, torch.Tensor):
            labels_tensor = numeric_labels
        else:
            labels_tensor = torch.tensor(numeric_labels)
        logits = classifier(embeddings, labels_tensor)
        loss = ce_loss(logits, labels_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    logging.info(f"Warmup Training Loss: {avg_loss:.4f}")
    wandb.log({"warmup_train_loss": avg_loss})

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


class ClassificationDatasetFromList(Dataset):
    def __init__(self, paths_labels):
        self.paths_labels = paths_labels

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        return self.paths_labels[idx]

def extract_and_save_embeddings(speaker_brain, audio_dir, df, emb_dir):
    os.makedirs(emb_dir, exist_ok=True)
    emb_paths = []
    valid_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        audio_path = os.path.join(audio_dir, row['path'])
        emb_path = os.path.join(emb_dir, f"{os.path.splitext(os.path.basename(row['path']))[0]}.npy")
        emb = extract_single_embedding(speaker_brain, audio_path)
        if emb is not None:
            np.save(emb_path, emb.cpu().numpy())
            emb_paths.append(emb_path)
            valid_rows.append(row)
        else:
            logging.warning(f"Failed to extract embedding for {audio_path}. Skipping.")
    # Return a DataFrame of only valid rows
    return emb_paths, pd.DataFrame(valid_rows)

def load_embeddings(emb_paths):
    embeddings = []
    for emb_path in emb_paths:
        emb = np.load(emb_path)
        # Flatten in case emb is not 1D
        emb = emb.flatten()
        embeddings.append(emb)
    return np.stack(embeddings)

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

def apply_spectral_clustering_to_disk_embeddings(df, emb_paths, config):
    embeddings_np = load_embeddings(emb_paths)
    n_clusters = count_unique_speakers(config['train_audio_list_file'])
    # Use ensemble clustering instead of a single clustering method
    cluster_labels = ensemble_clustering(embeddings_np, n_clusters=n_clusters, n_runs=3)
    clustered_data = []
    for i, path in enumerate(df['path']):
        clustered_data.append({"path": path, "cluster_label": f"{cluster_labels[i]}"})
    return pd.DataFrame(clustered_data)


if __name__ == '__main__':
    main()