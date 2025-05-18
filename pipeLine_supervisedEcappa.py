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
from speechbrain.pretrained import EncoderClassifier
from pytorch_metric_learning import miners, losses as mm_losses

def main(run=None, config_file='/app/config.yaml'):
    config = load_config(config_file)

    speaker_model = load_model_ecapa_from_speechbrain(config)
    embedding_model = speaker_model.mods.embedding_model

    for param in embedding_model.parameters():
        param.requires_grad = True

    num_classes = len(set(pd.read_csv(config['all_samples'], delimiter="\t")['client_id']))
    classifier_head = AMSoftmaxHead(192, num_classes)

    optimizer = AdamW(list(embedding_model.parameters()) + list(classifier_head.parameters()), lr=1e-4, weight_decay=1e-2)
    

    speaker_brain = modelTune(
        modules=speaker_model.mods,
        hparams={
            "compute_cost": nn.CrossEntropyLoss(),
            "opt_class": lambda params: optimizer,
            "classifier": classifier_head,
            "cosine_loss_fn": nn.CosineEmbeddingLoss(margin=0.0, reduction='mean'),
            "triplet_loss": nn.TripletMarginLoss(margin=0.8, p=2, swap=True, reduction="mean"),
            "cosine_weight": 2,
            "triplet_weight": 2,
        },
    )

    dataset = ClassificationDataset(config['all_samples'], config['audio_dir'])
    all_paths_labels = [(path, label) for path, label in dataset]
    train_paths_labels, val_paths_labels = train_test_split(all_paths_labels, test_size=0.2, random_state=42)

    train_dataset = ClassificationDatasetFromList(train_paths_labels, config['audio_dir'])
    val_dataset = ClassificationDatasetFromList(val_paths_labels, config['audio_dir'])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    for epoch in range(config['adaptation']['num_epochs']):

        logging.info(f"epoch: {epoch}")
        if epoch % 5 == 0:
            evaluate_model(speaker_brain, train_loader, config, epoch, True)
            evaluate_model(speaker_brain, val_loader, config, epoch,False)

        train_epoch(speaker_brain, train_dataset, config, epoch)

    Save_Model_Localy(speaker_brain.mods.embedding_model, config, name=f"ecapa_epoch_finetuned.pth")

    if run:
        run.finish()


def train_epoch(speaker_brain, train_dataset, config, epoch):
    speaker_brain.on_stage_start(sb.Stage.TRAIN, epoch=epoch)
    device = speaker_brain.device

    # Losses & miner
    cosine_loss_fn  = speaker_brain.hparams.cosine_loss_fn       # nn.CosineEmbeddingLoss
    triplet_loss_fn = speaker_brain.hparams.triplet_loss        # mm_losses.TripletMarginLoss(margin=0.8)
    miner           = miners.TripletMarginMiner(margin=0.8,type_of_triplets="semihard")
    alpha_cos       = speaker_brain.hparams.cosine_weight
    alpha_trip      = speaker_brain.hparams.triplet_weight
    classifier      = speaker_brain.hparams.classifier
    ce_loss_fn      = speaker_brain.hparams.compute_cost        # CrossEntropyLoss
    optimizer       = speaker_brain.hparams.opt_class(
                         list(speaker_brain.modules.embedding_model.parameters()) +
                         list(classifier.parameters()))

    train_loader = DataLoader(train_dataset,
                              batch_size=config['adaptation']['batch_size'],
                              shuffle=True,
                              num_workers=config.get('num_workers',4))

    total_loss = 0.0
    label_map, next_label = {}, 0

    for paths, labels in train_loader:
        # 1) Extract embeddings
        emb = extract_batch_embeddings_train(
                  speaker_brain, config['audio_dir'], paths
              )
        if emb.size(0) == 0:
            continue

        # 2) String → int labels
        numeric = []
        for l in labels:
            if l not in label_map:
                label_map[l] = next_label; next_label += 1
            numeric.append(label_map[l])
        labels_tensor = torch.tensor(numeric, device=device)

        # 3) Classification head + CE loss
        logits  = classifier(emb, labels_tensor)
        ce_loss = ce_loss_fn(logits, labels_tensor)

        # 4) CosineEmbeddingLoss against class weights
        class_wt = classifier.weight[labels_tensor]            # [B, D]
        cos_tgts = torch.ones(emb.size(0), device=device)     # positive pairs
        cos_loss = cosine_loss_fn(emb, class_wt, cos_tgts)

        # 5) Mine semi‑hard triplets and compute triplet loss
        a_idx, p_idx, n_idx = miner(emb, labels_tensor)
        if a_idx.numel() > 0:
            anc = emb[a_idx]
            pos = emb[p_idx]
            neg = emb[n_idx]
            triplet_loss = triplet_loss_fn(anc, pos, neg)
        else:
            triplet_loss = torch.tensor(0.0, device=device)

        # 6) Combine
        loss = ce_loss \
             + alpha_cos  * cos_loss \
             + alpha_trip * triplet_loss

        # 7) Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    wandb.log({"train_loss": avg_loss})



def evaluate_model(speaker_brain, dataloader, config, epoch, train = False):
    all_preds = []
    all_labels = []
    all_embeddings = []
    label_map = {}
    label_counter = 0

    with torch.no_grad():
        for paths, labels in dataloader:
            numeric_labels = []
            for label in labels:
                if label not in label_map:
                    label_map[label] = label_counter
                    label_counter += 1
                numeric_labels.append(label_map[label])

            labels_tensor = torch.tensor(numeric_labels)
            emb = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths)
            if not len(emb):
                print("No embeddings in batch, skipping...")
                continue

            logits = speaker_brain.hparams.classifier(emb, labels_tensor[:len(emb)])
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_tensor[:len(preds)].cpu().numpy().tolist())
            all_embeddings.extend(emb.cpu().numpy().tolist())

    assert len(all_preds) == len(all_labels) == len(all_embeddings)

    evaluations = Evaluations()
    y_true, y_scores = evaluations.build_verification_pairs(all_embeddings, all_labels)
    eer = evaluations.calculate_eer(y_true, y_scores)

    ieer = evaluations.calculate_ieer(y_true, y_scores)
    logging.info(f"[Epoch {epoch}] EER: {eer}, IEER: {ieer}")
    if train:
        wandb.log({"train EER": eer,
                   "train IEER": ieer})
    else:
        wandb.log({"val EER": eer,
                   "val IEER": ieer})

    plot_embeddings(np.array(all_embeddings), all_labels, epoch, train)

def plot_embeddings(embeddings, labels, epoch,train = False):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    labels = np.array(labels)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='tab10', s=10)
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"t-SNE Embedding Visualization - Epoch {epoch}")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.tight_layout()

    if train:
        plot_path = f"/app/plots/train_tsne_epoch_{epoch}.png"
    else:
        plot_path = f"/app/plots/tsne_epoch_{epoch}.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    # Log the plot to wandb
    
    if train:
        wandb.log({"clustering_train_plot": wandb.Image(plot_path)})
    else:
        wandb.log({"clustering_plot": wandb.Image(plot_path)})


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

class ClassificationDatasetFromList(torch.utils.data.Dataset):
    def __init__(self, paths_labels, audio_dir):
        self.paths_labels = paths_labels
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        path, label = self.paths_labels[idx]
        return path, label

if __name__ == '__main__':
    main()
