from model_funcs import *
from meta_data_preproccesing import *
from utils import *
from SV import *
from adaptation import *
from evaluation import *
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
import wandb
import os
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import logging
from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F
from pytorch_metric_learning import miners

def main(run=None, config_file='/app/config.yaml'):
    config = load_config(config_file)

    # Load ECAPA model
    speaker_model = load_model_ecapa_from_speechbrain(config)
    embedding_model = speaker_model.mods.embedding_model
    for param in embedding_model.parameters():
        param.requires_grad = True

    df = pd.read_csv(config['all_samples'], delimiter="\t")
    label_set = sorted(set(df['client_id']))
    label_map = {label: idx for idx, label in enumerate(label_set)}
    num_classes = len(label_map)

    classifier_head = AMSoftmaxHead(192, num_classes)
    optimizer = AdamW(list(embedding_model.parameters()) + list(classifier_head.parameters()), lr=1e-3, weight_decay=1e-2)

    # Losses
    ce_loss_fn = nn.CrossEntropyLoss()
    speaker_brain = modelTune(
        modules=speaker_model.mods,
        hparams={
            "compute_cost": ce_loss_fn,
            "classifier": classifier_head,
            "optimizer": optimizer,
        },
    )

    train_loader, val_loader = load_and_split_dataset(df, label_map, config, test_size=0.2, random_state=42)

    # ------------------- WARM-UP PHASE -------------------
    num_warmup_epochs = config['adaptation'].get('num_warmup_epochs', 3)
    for param in embedding_model.parameters():
        param.requires_grad = False
    for param in classifier_head.parameters():
        param.requires_grad = True

    logging.info("Warm-up: training classifier only")
    for epoch in range(num_warmup_epochs):
        train_epoch_warmup(speaker_brain, train_loader, config, epoch)
        
    # ------------------- MAIN TRAINING PHASE -------------------
    for param in embedding_model.parameters():
        param.requires_grad = True

    for epoch in range(config['adaptation']['num_epochs']):
        logging.info(f"epoch: {epoch}")
        if epoch % 1 == 0 or epoch < 10:
            logging.info(f"eval")
            evaluate_model(speaker_brain, train_loader, label_map, config, epoch, train=True)
            evaluate_model(speaker_brain, val_loader, label_map, config, epoch, train=False)

        logging.info(f"train")
        train_epoch_with_losses(speaker_brain, train_loader, label_map, config, epoch)

    Save_Model_Localy(speaker_brain.mods.embedding_model, config, name="ecapa_epoch_finetuned.pth")
    if run:
        run.finish()

def load_and_split_dataset(df, label_map, config, test_size=0.2, random_state=42):
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

    def to_indexed_list(df):
        return [(path, label_map[label]) for path, label in zip(df['path'], df['client_id'])]

    train_list = to_indexed_list(train_df)
    val_list = to_indexed_list(val_df)

    train_dataset = ClassificationDatasetFromList(train_list)
    val_dataset = ClassificationDatasetFromList(val_list)

    return DataLoader(train_dataset, batch_size=16, shuffle=True), DataLoader(val_dataset, batch_size=16)

def train_epoch_warmup(speaker_brain, train_loader, config, epoch):
    """Train only the classifier head (embedding frozen) with cross-entropy loss."""
    speaker_brain.on_stage_start(sb.Stage.TRAIN)
    device = speaker_brain.device
    classifier = speaker_brain.hparams.classifier
    ce_loss_fn = speaker_brain.hparams.compute_cost
    optimizer = speaker_brain.hparams.optimizer

    total_loss = 0.0
    for paths, numeric_labels in train_loader:
        embeddings = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths)
        if embeddings.size(0) == 0:
            continue

        if not isinstance(numeric_labels, torch.Tensor):
            labels_tensor = torch.tensor(numeric_labels, device=device)
        else:
            labels_tensor = numeric_labels.to(device)

        logits = classifier(embeddings, labels_tensor)
        loss = ce_loss_fn(logits, labels_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Warmup Training Loss: {avg_loss:.4f}")
    wandb.log({"warmup_train_loss": avg_loss})

def train_epoch_with_losses(speaker_brain, train_loader, label_map, config, epoch):
    """Train with triplet, cross-entropy, and self-supervised regression losses."""
    speaker_brain.on_stage_start(sb.Stage.TRAIN)
    device =            speaker_brain.device
    classifier =        speaker_brain.hparams.classifier
    ce_loss_fn =        speaker_brain.hparams.compute_cost
    optimizer =         speaker_brain.hparams.optimizer
    
    total_loss = 0.0
    for paths, numeric_labels in train_loader:
        embeddings = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths)
        if embeddings.size(0) == 0:
            continue

        if not isinstance(numeric_labels, torch.Tensor):
            labels_tensor = torch.tensor(numeric_labels, device=device)
        else:
            labels_tensor = numeric_labels.to(device)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        logits = classifier(embeddings, labels_tensor)
        ce_loss = ce_loss_fn(logits, labels_tensor)

        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()
        total_loss += ce_loss.item()

        wandb.log({
            "ce_loss": ce_loss.item()
        })

    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    wandb.log({"train_loss": avg_loss})

def evaluate_model(speaker_brain, dataloader, label_map, config, epoch, train=False):
    y_true_all, y_score_all = [], []
    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for paths, numeric_labels in dataloader:
            if isinstance(numeric_labels, torch.Tensor):
                labels_tensor = numeric_labels.detach().clone()
            else:
                labels_tensor = torch.tensor(numeric_labels)

            embeddings = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths)
            if embeddings.size(0) == 0:
                continue

            labels_tensor = labels_tensor[:len(embeddings)]

            logits = speaker_brain.hparams.classifier(embeddings, labels_tensor)
            preds = torch.argmax(logits, dim=1)

            y_true, y_score = Evaluations.build_verification_pairs(
                embeddings.cpu().numpy(),
                labels_tensor.cpu().numpy(),
            )
            # Use append instead of extend
            y_true_all.append(np.asarray(y_true))
            y_score_all.append(np.asarray(y_score))

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels_tensor.cpu().numpy())

    # Concatenate after the loop
    if y_true_all:
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_score_all = np.concatenate(y_score_all, axis=0)
    else:
        y_true_all = np.array([])
        y_score_all = np.array([])

    if all_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_embeddings = np.array([])
        all_labels = np.array([])

    eer = Evaluations.calculate_eer(y_true_all, y_score_all) if y_true_all.size > 0 else float('nan')
    ieer = Evaluations.calculate_ieer(y_true_all, y_score_all) if y_true_all.size > 0 else float('nan')

    tag = "train" if train else "val"
    logging.info(f"[Epoch {epoch} {tag}] EER: {eer:.4f}, IEER: {ieer:.4f}")
    wandb.log({f"{tag} EER": eer, f"{tag} IEER": ieer})

    if all_embeddings.size > 0:
        plot_embeddings(all_embeddings, all_labels, epoch, train=train)

    del embeddings, all_embeddings, all_labels, labels_tensor, logits

def plot_embeddings(embeddings, labels, epoch, train=False):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    labels = np.array(labels)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='tab10', s=10)
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"t-SNE Embedding Visualization - Epoch {epoch}")
    plt.tight_layout()

    path = f"/app/plots/{'train_' if train else ''}tsne_epoch_{epoch}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    wandb.log({f"{'clustering_train_plot' if train else 'clustering_plot'}": wandb.Image(path)})

class ClassificationDatasetFromList(torch.utils.data.Dataset):
    def __init__(self, paths_labels):
        self.paths_labels = paths_labels

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        path, label = self.paths_labels[idx]
        return path, label

if __name__ == '__main__':
    main()
