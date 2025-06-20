from model_funcs import *
from meta_data_preproccesing import *
from utils import *
from SV import *
from adaptation import *
from evaluation import *
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import wandb
import os
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import gc

logging.basicConfig(level=logging.INFO)

def main(run=None, config_file='/app/config.yaml'):
    config = load_config(config_file)
    speaker_model = load_model_ecapa_from_speechbrain(config)
    embedding_model = speaker_model.mods.embedding_model
    for p in embedding_model.parameters():
        p.requires_grad = True

    df = pd.read_csv(config['train_audio_list_file'], delimiter="\t")
    families = df['family_id'].unique()
    emb_dir = config.get('embedding_output_file', '/app/embeddings_per_family')

    # Map client_id to integer indices
    label_set = sorted(df['client_id'].unique())
    label_map = {label: idx for idx, label in enumerate(label_set)}
    df['client_id_idx'] = df['client_id'].map(label_map)

    num_classes = df['client_id'].nunique()
    classifier_head = AMSoftmaxHead(192, num_classes)
    optimizer = AdamW(
        list(embedding_model.parameters()) + list(classifier_head.parameters()), lr=1e-3, weight_decay=1e-2
    )

    speaker_brain = modelTune(
        modules=speaker_model.mods,
        hparams={
            "compute_cost": nn.CrossEntropyLoss(),
            "opt_class": lambda params: optimizer,
            "classifier": classifier_head,
            "cosine_loss_fn": nn.CosineEmbeddingLoss(margin=config['adaptation']['margin'], reduction='mean'),
            "triplet_loss": nn.TripletMarginLoss(margin=config['adaptation']['margin'], p=2, swap=True, reduction="mean"),
            "cosine_weight": 1,
            "triplet_weight": 1,
        },
    )

    # ------------------- WARM-UP PHASE -------------------
    num_warmup_epochs = config['adaptation'].get('num_warmup_epochs', 3)
    for p in embedding_model.parameters():
        p.requires_grad = False
    for p in classifier_head.parameters():
        p.requires_grad = True

    logging.info("Warm-up: training classifier only")
    for epoch in range(num_warmup_epochs):
        all_train_data = prepare_train_data(df, families, speaker_brain, config, emb_dir)
        train_loader = DataLoader(ClassificationDatasetFromList(all_train_data), batch_size=16, shuffle=True)
        train_epoch_warmup(speaker_brain, train_loader, config, epoch)
        del train_loader
        gc.collect()

    # ------------------- MAIN TRAINING PHASE -------------------
    for p in embedding_model.parameters():
        p.requires_grad = True

    run_training_loop(config, speaker_brain, df, families, emb_dir)

    Save_Model_Localy(speaker_brain.mods.embedding_model, config, name="ecapa_epoch_finetuned_per_family.pth")
    if run:
        run.finish()

def run_training_loop(config, speaker_brain, df, families, emb_dir):
    for epoch in range(config['adaptation']['num_epochs']):
        logging.info(f"Epoch {epoch}: Per-family supervised adaptation")
        all_train_data = prepare_train_data(df, families, speaker_brain, config, emb_dir)
        train_loader = DataLoader(ClassificationDatasetFromList(all_train_data), batch_size=16, shuffle=True)
        train_epoch(speaker_brain, train_loader, config, epoch)

        # --- After all families, evaluate on the entire dataset (train and val splits) ---
        logging.info(f"Epoch {epoch}: Evaluating on all data (train/val splits)")
        all_embeddings = extract_all_embeddings(df, emb_dir)
        if all_embeddings is not None:
            df = df.reset_index(drop=True)
            train_idx, val_idx = train_test_split(df.index, test_size=0.2, random_state=42)
            train_embeddings = all_embeddings[train_idx]
            val_embeddings = all_embeddings[val_idx]
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            evaluate_split(config, train_df, train_embeddings, "train")
            evaluate_split(config, val_df, val_embeddings, "val")
            del all_embeddings, train_loader
            gc.collect()

def train_epoch_warmup(speaker_brain, train_loader, config, epoch):
    """Train only the classifier head (embedding frozen) with cross-entropy loss."""
    speaker_brain.on_stage_start(sb.Stage.TRAIN)
    device = speaker_brain.device
    classifier = speaker_brain.hparams.classifier
    ce_loss_fn = speaker_brain.hparams.compute_cost
    optimizer = speaker_brain.hparams.opt_class(
        list(classifier.parameters())
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

        logits = classifier(emb, labels_tensor)
        ce_loss = ce_loss_fn(logits, labels_tensor)

        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()
        total_loss += ce_loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Warmup Training Loss: {avg_loss:.4f}")
    wandb.log({"warmup_train_loss": avg_loss})

def train_epoch(speaker_brain, train_loader, config, epoch):
    speaker_brain.on_stage_start(sb.Stage.TRAIN)
    device          = speaker_brain.device
    classifier      = speaker_brain.hparams.classifier
    ce_loss_fn      = speaker_brain.hparams.compute_cost
    cosine_loss_fn  = speaker_brain.hparams.cosine_loss_fn
    triplet_loss_fn = speaker_brain.hparams.triplet_loss
    miner           = miners.TripletMarginMiner(margin=config['adaptation']['margin'], type_of_triplets="semihard")
    alpha_cos       = speaker_brain.hparams.cosine_weight
    alpha_trip      = speaker_brain.hparams.triplet_weight
    optimizer       = speaker_brain.hparams.opt_class(
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

def extract_all_embeddings(df, emb_dir):
    all_emb_paths = []
    for idx, row in df.iterrows():
        emb_path = os.path.join(emb_dir, f"{row['family_id']}_{os.path.splitext(os.path.basename(row['path']))[0]}.npy")
        if os.path.exists(emb_path):
            all_emb_paths.append(emb_path)
    if len(all_emb_paths) > 1:
        return load_embeddings(all_emb_paths)
    else:
        return None

def evaluate_split(config, df, embeddings, split_name):
    true_labels = df['client_id_idx'].values
    evaluate_and_log(config, df, None, embeddings, true_labels, train=(split_name == "train"))
    plot_embedding(config, embeddings, true_labels)

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

def prepare_train_data(df, families, speaker_brain, config, emb_dir):
    all_train_data = []
    for family_id in families:
        family_df = df[df['family_id'] == family_id]
        if len(family_df) < 2:
            logging.info(f"Skipping family {family_id} (not enough samples)")
            continue
        emb_paths, valid_df = extract_and_save_family_embeddings(
            speaker_brain, config['audio_dir'], family_df, emb_dir, family_id)
        if len(emb_paths) < 2:
            logging.info(f"Skipping family {family_id} (not enough valid embeddings)")
            continue
        valid_df = valid_df.reset_index(drop=True)
        train_df, _ = train_test_split(valid_df, test_size=0.2, random_state=42)
        all_train_data.extend([(p, l) for p, l in zip(train_df['path'], train_df['client_id_idx'])])
    return all_train_data

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, list_file, audio_dir):
        self.df = pd.read_csv(list_file, delimiter="\t")
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.audio_dir, row['path'])
        label = row['client_id_idx']
        return path, label

class ClassificationDatasetFromList(Dataset):
    def __init__(self, paths_labels):
        self.paths_labels = paths_labels

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        return self.paths_labels[idx]

if __name__ == '__main__':
    main()
