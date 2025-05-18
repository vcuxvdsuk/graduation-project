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
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier
from model_funcs import *
from meta_data_preproccesing import *
from utils import *
from SV import *
from clustering import *
from adaptation import *
from evaluation import *

logging.basicConfig(level=logging.INFO)

class ClassificationDatasetFromList(Dataset):
    def __init__(self, paths_labels):
        self.paths_labels = paths_labels

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        return self.paths_labels[idx]

def extract_embedding_from_path(model, full_path):
    try:
        signal = model.load_audio(full_path)
        emb = model.encode_batch(signal.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu()
        return emb
    except Exception as e:
        logging.warning(f"Failed on {full_path}: {e}")
        return None


def apply_spectral_clustering_to_df(df, model, config):
    clustered_data = []
    families = df['family_id'].unique()

    for family_id in families:
        family_df = df[df['family_id'] == family_id]
        paths, embeddings = [], []

        for path in family_df['path']:
            emb = extract_embedding_from_path(model, os.path.join(config['audio_dir'], path))
            if emb is not None:
                paths.append(path)
                embeddings.append(emb)

        if len(embeddings) < 2:
            continue

        embeddings_np = torch.stack(embeddings).cpu().numpy()
        n_clusters = get_unique_speakers_in_family(config['train_audio_list_file'], family_id)
        cluster_labels, _ = spectral_clustering(embeddings_np, n_clusters=n_clusters, family_id=family_id)

        for i, path in enumerate(paths):
            clustered_data.append({"path": path, "cluster_label": f"{family_id}_{cluster_labels[i]}"})

    return pd.DataFrame(clustered_data)

def train_epoch(speaker_brain, train_loader, config, epoch):
    speaker_brain.on_stage_start("train")
    device = speaker_brain.device
    classifier = speaker_brain.hparams.classifier
    ce_loss = speaker_brain.hparams.compute_cost

    optimizer = speaker_brain.hparams.opt_class(
        list(speaker_brain.modules.embedding_model.parameters()) + list(classifier.parameters())
    )

    total_loss = 0.0
    for paths, numeric_labels in train_loader:
        embeddings = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths)
        if embeddings.size(0) == 0:
            continue

        labels_tensor = torch.tensor(numeric_labels, device=device)
        logits = classifier(embeddings.to(device))
        loss = ce_loss(logits, labels_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    logging.info(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")
    wandb.log({"train_loss": avg_loss})

def evaluate_on_true_labels(speaker_brain, df, label_map, config, epoch, train=True):
    """
    Evaluate the fine-tuned embedding model on the true speaker labels (not clusters).

    Args:
        speaker_brain: your SpeechBrain Brain object.
        df (pd.DataFrame): must contain columns ['path', 'client_id'].
        label_map (dict): mapping from client_id string → numeric label.
        config: configuration dict with 'audio_dir' etc.
        epoch (int)
        train (bool): whether this is train or val eval (for logging).
    """
    y_true_all, y_score_all = [], []
    tsne_embeddings, tsne_labels = [], []
    device = speaker_brain.device
    classifier = speaker_brain.hparams.classifier

    # Drop any missing client_id rows
    print(df)
    df = df.dropna(subset=['client_id'])
    # Build (path, numeric_label) list
    data = [(row['path'], label_map[row['client_id']]) for _, row in df.iterrows()]
    loader = DataLoader(ClassificationDatasetFromList(data), batch_size=16)

    with torch.no_grad():
        for paths, numeric_labels in loader:
            # 1) Extract embeddings in a memory‑efficient batch
            embeddings = extract_batch_embeddings_train(
                speaker_brain, config['audio_dir'], paths
            )
            if embeddings.size(0) == 0:
                continue

            # 2) True labels → tensor on device
            labels_tensor = torch.tensor(numeric_labels, device=device)

            # 3) Build verification pairs (subsample up to 1000 per batch)
            y_true, y_score = Evaluations.build_verification_pairs(
                embeddings.cpu().numpy(),
                labels_tensor.cpu().numpy(),
                max_pairs=1000
            )
            y_true_all.extend(y_true)
            y_score_all.extend(y_score)

            # 4) Collect a small sample for t-SNE
            if len(tsne_embeddings) < 2000:
                tsne_embeddings.append(embeddings.cpu())
                tsne_labels.extend(labels_tensor.cpu().numpy())

    # 5) Compute metrics
    eer = Evaluations.calculate_eer(y_true_all, y_score_all)
    ieer = Evaluations.calculate_ieer(y_true_all, y_score_all)
    tag = "train" if train else "val"
    logging.info(f"[Epoch {epoch} {tag}] True‑label EER: {eer:.4f}, IEER: {ieer:.4f}")
    wandb.log({f"{tag}_EER_true": eer, f"{tag}_IEER_true": ieer})

    # 6) t-SNE visualization (if we gathered any)
    if tsne_embeddings:
        emb_mat = torch.cat(tsne_embeddings, dim=0).numpy()
        reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(emb_mat)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=tsne_labels, cmap='tab10', s=10)
        plt.legend(*scatter.legend_elements(), title="Speakers",
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"t-SNE on True Labels – Epoch {epoch}")
        out_path = f"/app/plots/true_tsne_epoch_{epoch}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        wandb.log({f"{tag}_tsne_true": wandb.Image(out_path)})


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
        hparams={"compute_cost": nn.CrossEntropyLoss(), "opt_class": lambda params: optimizer, "classifier": classifier_head}
    )

    for epoch in range(config['adaptation']['num_epochs']):
        clustered_df = apply_spectral_clustering_to_df(df, speaker_model, config)
        label_set = sorted(set(clustered_df['cluster_label']))
        label_map = {label: idx for idx, label in enumerate(label_set)}
        num_classes = len(label_map)

        # After clustered_df is created
        clustered_df = apply_spectral_clustering_to_df(df, speaker_model, config)
        # Merge cluster labels back to original df to retain client_id
        merged_df = pd.merge(clustered_df, df[['path', 'client_id']], on='path', how='left')

        # Now split merged_df
        train_df, val_df = train_test_split(merged_df, test_size=0.2)

        train_loader = DataLoader(
            ClassificationDatasetFromList([(p, label_map[l]) for p, l in zip(train_df['path'], train_df['cluster_label'])]),
            batch_size=16, shuffle=True
        )
        val_loader = DataLoader(
            ClassificationDatasetFromList([(p, label_map[l]) for p, l in zip(val_df['path'], val_df['cluster_label'])]),
            batch_size=16
        )
        
        label_set = sorted(set(df['client_id']))
        label_map = {label: idx for idx, label in enumerate(label_set)}

        logging.info(f"Epoch {epoch}: Training with {num_classes} pseudo-labels")
        if epoch % 5 == 0 or epoch < 5:
            evaluate_on_true_labels(speaker_brain, train_df, label_map, config, epoch, train=True)
            evaluate_on_true_labels(speaker_brain, val_df, label_map, config, epoch, train=False)

        train_epoch(speaker_brain, train_loader, config, epoch)

    Save_Model_Localy(speaker_brain.mods.embedding_model, config, name="ecapa_epoch_finetuned.pth")
    if run:
        run.finish()

if __name__ == '__main__':
    main()
