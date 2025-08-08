from model_funcs import *
from clustering import *
from meta_data_preproccesing import *
from SV import *
from ploting_funcs import *
from adaptation import *
from evaluation import *
from utils import *
from scipy.stats import mode

import wandb
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score
from pytorch_metric_learning import miners
import torchaudio.transforms as T
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import logging
from collections import Counter, defaultdict
import numpy as np

# Global counter for clustering methods
CLUSTER_METHOD_COUNTER = Counter()

SILHOUETTE_THRESHOLD = 0.15

def main(run=None, config_file='/app/config.yaml'):
    config = load_config(config_file)
    speaker_model = load_model_ecapa_from_speechbrain(config)
    embedding_model = speaker_model.mods.embedding_model

    # leave embeddor trainable
    for p in embedding_model.parameters():
        p.requires_grad = True


    # Add dropout before classifier if not present in embedding model
    embedding_model = DropoutWrapper(embedding_model, p=0.3)

    # instantiate both losses
    num_speakers = count_unique_speakers(config['train_audio_list_file'])
    classifier_head = AMSoftmaxHead(192, num_speakers, margin=config['adaptation']['margin'], scale=50.0)
    ce_loss_fn = nn.CrossEntropyLoss()
    triplet_loss_fn = nn.TripletMarginLoss(margin=config['adaptation']['margin'], p=2, swap=True, reduction="mean")
    ssreg_loss_fn = nn.MSELoss(reduction="mean")


    # single optimizer over both embedding model + AM‑Softmax weights
    optimizer = torch.optim.AdamW(
        list(embedding_model.parameters()) + list(classifier_head.parameters()), lr=1e-6, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    speaker_brain = modelTune(
        modules=speaker_model.mods,
        hparams={
            "triplet_loss":     triplet_loss_fn,
            "soft_weight":      1,
            "classifier":       classifier_head,
            "soft_loss":        ce_loss_fn,
            "ssreg_weight":     1,
            "ssreg_loss":       ssreg_loss_fn,
            "optimizer":        optimizer,
            "scheduler":        scheduler
        },
        run_opts={"device": speaker_model.device}
    )

    # 1. Freeze embedding model
    for p in embedding_model.parameters():
        p.requires_grad = False

    # 2. Train only the classifier for a few epochs
    num_warmup_epochs = config['adaptation'].get('num_warmup_epochs', 3)  # Default to 3 if not specified
    logging.info(f"warm up")
    extract_all_families_embeddings(speaker_brain,  # fine-tuned model
                                    config['audio_dir'],
                                    config['train_audio_list_file'],
                                    config['embedding_output_file'],
                                    config['familes_emb'])
    for epoch in range(num_warmup_epochs):
        train_per_family(config, speaker_brain, epoch)

    # 3. Unfreeze embedding model for full training
    for p in embedding_model.parameters():
        p.requires_grad = True

    global SILHOUETTE_THRESHOLD
    # Continue with your main training loop as before
    for epoch in range(config['adaptation']['num_epochs']):
        if epoch%1 == 0 :
            SILHOUETTE_THRESHOLD = SILHOUETTE_THRESHOLD - 0.01

        logging.info(f"[Epoch {epoch}] Extract embeddings")
        extract_all_families_embeddings(speaker_brain,  # fine-tuned model
                                        config['audio_dir'],
                                        config['train_audio_list_file'],
                                        config['embedding_output_file'],
                                        config['familes_emb'])

        logging.info(f"[Epoch {epoch}] Test full DB")
        test_entire_database(config, speaker_model, epoch)
        
        logging.info(f"[Epoch {epoch}] Adapt per family")
        train_per_family(config, speaker_brain, epoch)

    Save_Model_Localy(
        speaker_brain.modules.embedding_model,
        config, name="fine_tuned_model.pth"
    )
    if run: run.finish()



def train_per_family(config, speaker_brain, epoch):
    """Curriculum learning: train on families with highest silhouette first."""
    df = pd.read_csv(config['familes_emb'], sep="\t")
    labels_csv = f"{config['familes_labels']}{epoch}.csv"
    clean_csv(labels_csv)

    triplet_loss_fn = speaker_brain.hparams.triplet_loss
    soft_loss_fn    = speaker_brain.hparams.soft_loss
    optimizer       = speaker_brain.hparams.optimizer
    miner           = miners.TripletMarginMiner(margin=config['adaptation']['margin'], type_of_triplets="hard")
    alpha           = speaker_brain.hparams.soft_weight
    beta            = speaker_brain.hparams.ssreg_weight

    # -------- Training loop --------
    train_df = df.head(config['adaptation']['train_nums_of_families'])

    # Compute silhouette for all families and sort
    fam_sil_list = []
    for _, row in train_df.iterrows():
        fam_id = row['family_number']
        fam_emb = np.load(row['embedding_path'])
        n_speakers = get_unique_speakers_in_family(config['train_audio_list_file'], fam_id)
        _, sil_score = prepare_family_labels(config, fam_id, fam_emb, n_speakers, labels_csv)
        fam_sil_list.append((fam_id, sil_score, fam_emb, n_speakers))

    # Sort by silhouette descending (curriculum learning)
    fam_sil_list.sort(key=lambda x: x[1], reverse=True)

    for fam_id, sil_score, fam_emb, n_speakers in fam_sil_list:
        if sil_score < SILHOUETTE_THRESHOLD:
            logging.info(f"Skipping family {fam_id} due to low silhouette score ({sil_score:.3f})")
            continue
        labels, _ = prepare_family_labels(config, fam_id, fam_emb, n_speakers, labels_csv)
        train_single_family(config, speaker_brain, fam_id, labels, optimizer, miner, triplet_loss_fn, soft_loss_fn, alpha, beta)

    # -------- Testing loop --------
    wandb.log({f"epoch": epoch})
    test_df = df.iloc[config['adaptation']['train_nums_of_families']:]
    for _, row in test_df.iterrows():
        test_single_family(config, speaker_brain, row, labels_csv)

    print("Clustering method usage counts:", dict(CLUSTER_METHOD_COUNTER))
    CLUSTER_METHOD_COUNTER.clear()  # Reset counter for next epoch


def train_single_family(config, speaker_brain, fam_id, labels, optimizer, miner, triplet_loss_fn, soft_loss_fn, alpha, beta):
    """Train on a single family with Online Triplet Mining"""
    dataset = OnlineTripletDataset(get_audio_paths_for_family(config['train_audio_list_file'], fam_id), labels)
    if len(dataset) == 0:
        logging.warning(f"Empty dataset for family {fam_id}, skipping")
        return

    loader = DataLoader(dataset, batch_size=config['adaptation']['batch_size'], shuffle=True)
    speaker_brain.on_stage_start(sb.Stage.TRAIN)

    for _ in range(config['adaptation']['epochs_per_family']):
        for paths, lab_t in loader:
            emb = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths)
            if emb.size(0) == 0:
                continue

            lab_t = lab_t[:len(emb)].long().to(emb.device)
            emb = F.normalize(emb, p=2, dim=1)

            a, p, n = miner(emb, lab_t)
            t_loss = triplet_loss_fn(emb[a], emb[p], emb[n]) if a.numel() > 0 else torch.tensor(0.0, device=emb.device)

            aug_emb = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths, 0.05, True)
            aug_emb = F.normalize(aug_emb, p=2, dim=1)
            assert len(emb) == len(aug_emb), "Mismatch in augmented batch size"

            ssreg_loss = torch.nn.functional.mse_loss(emb, aug_emb)
            logits = speaker_brain.hparams.classifier(emb, lab_t)
            s_loss = speaker_brain.hparams.soft_loss(logits, lab_t)

            loss = t_loss + alpha * s_loss + beta * ssreg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            speaker_brain.hparams.scheduler.step()
            torch.nn.utils.clip_grad_norm_(speaker_brain.modules.embedding_model.parameters(), max_norm=1.0)

            wandb.log({"family_loss": loss.item()})

def test_entire_database(config, speaker_model, epoch):
    speaker_model.eval()
    families_df = pd.read_csv(config['familes_emb'], delimiter="\t")
    labels_csv_path = f"{config['familes_labels']}_entire_database_{epoch}.csv"
    clean_csv(labels_csv_path)

    all_embeddings = []
    for _, row in families_df.iterrows():
        emb = np.load(row['embedding_path'])
        all_embeddings.append(emb)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    num_speakers = count_unique_speakers(config['train_audio_list_file'])

    process_entire_dataset(config, labels_csv_path, speaker_model, all_embeddings, num_speakers, to_plot=True)


def process_entire_dataset(config, labels_csv_path, speaker_model, all_embeddings, num_speakers, to_plot):
    # Use prepare_family_labels for the entire dataset (fam_id=None)
    labels, _ = prepare_family_labels(config, None, all_embeddings, num_speakers, labels_csv_path)
    labels = np.asarray(labels).flatten()
    os.makedirs(os.path.dirname(config['familes_labels']), exist_ok=True)
    append_All_to_csv(labels, config['train_audio_list_file'], labels_csv_path)
    
    df = pd.read_csv(config['train_audio_list_file'], delimiter="\t")

    true_ids = df['client_id'].values if 'client_id' in df.columns else None
    if true_ids is not None and len(true_ids) == len(labels):
        evaluate_and_log(config, df, None, all_embeddings, labels)
    else:
        # fallback: call as before, but print a warning
        print("Warning: true_ids and predicted labels length mismatch or missing 'speaker_id' column.")
        evaluate_and_log(config, df, None, all_embeddings, labels)

    if to_plot:
        plot_embedding(config, all_embeddings, labels)


def test_single_family(config, speaker_model, row, labels_csv):
    """Test a single family by clustering embeddings and testing against the cluster labels"""
    family_id = row['family_number']
    family_emb = np.load(row['embedding_path'])

    n_speakers = get_unique_speakers_in_family(config['train_audio_list_file'], family_id)
    test_group = get_audio_paths_for_family(config['train_audio_list_file'], family_id)

    if len(test_group) == 0:
        logging.warning(f"No test samples for family {family_id}, skipping.")
        return

    # Convert test_group list to DataFrame with a 'path' column if it's a list of strings
    if isinstance(test_group, list):
        test_group = pd.DataFrame(test_group, columns=["path"])

    labels, _ = prepare_family_labels(config, family_id, family_emb, n_speakers, labels_csv)
    df = pd.read_csv(config['train_audio_list_file'], delimiter="\t")
    family_df = df[df['family_id'] == family_id]
    
    evaluate_and_log(config, family_df, family_id, family_emb, labels,False)

def prepare_family_labels(config, fam_id, fam_emb, n_speakers, labels_csv, n_runs=20, pca_dim=30):
    """Ensemble clustering with normalization and PCA, select best by silhouette."""
    global CLUSTER_METHOD_COUNTER

    # Normalize embeddings (torch -> numpy)
    emb_norm = F.normalize(torch.tensor(fam_emb), p=2, dim=1).cpu().numpy()
    n_samples, n_features = emb_norm.shape
    n_components = min(pca_dim, n_samples, n_features)

    # Apply PCA if possible
    if n_components >= 1 and n_components < n_features:
        pca = PCA(n_components=n_components)
        emb_proc = pca.fit_transform(emb_norm)
    else:
        emb_proc = emb_norm

    # Normalize after PCA
    X_norm = normalize(emb_proc, norm='l2')

    best_sil = -1
    best_labels = None
    method = "kmeans"  # Default method

    for seed in range(n_runs):
        
        # KMeans clustering
        km = kmeans = KMeans(
            n_clusters=n_speakers,
            init='k-means++',
            n_init=20,
            max_iter=300,
            random_state=seed)
        labels_km = km.fit_predict(X_norm)
        sil_km = silhouette_score(X_norm, labels_km) if len(set(labels_km)) > 1 else -1
        if sil_km > best_sil:
            method = "kmeans"
            best_sil = sil_km
            best_labels = labels_km

        # Spectral clustering (only if enough samples)
        if len(X_norm) > n_speakers:
            try:
                affinity_matrix = (cosine_similarity(X_norm) + 1.0) / 2.0  # Ensure non-negative
                sc = SpectralClustering(
                    n_clusters=n_speakers,
                    affinity='precomputed',
                    assign_labels='kmeans',
                    n_init=10,
                    random_state=seed
                )
                labels_sc = sc.fit_predict(affinity_matrix)
                sil_sc = silhouette_score(X_norm, labels_sc) if len(set(labels_sc)) > 1 else -1
                if sil_sc > best_sil:
                    method = "spectral"
                    best_sil = sil_sc
                    best_labels = labels_sc
            except Exception as e:
                logging.debug(f"Spectral clustering failed: {e}")

    # Agglomerative clustering (no random_state)
    try:
        agg = AgglomerativeClustering(n_clusters=n_speakers, metric='cosine', linkage='average')
        labels_agg = agg.fit_predict(X_norm)
        sil_agg = silhouette_score(X_norm, labels_agg) if len(set(labels_agg)) > 1 else -1
        if sil_agg > best_sil:
            method = "agglomerative"
            best_sil = sil_agg
            best_labels = labels_agg
    except Exception as e:
        logging.debug(f"Agglomerative clustering failed: {e}")

    # Enforce maximum cluster size
    max_size = 6  # your desired max cluster size
    best_labels = enforce_max_cluster_size(best_labels, max_size)

    # Count the chosen method
    CLUSTER_METHOD_COUNTER[method] += 1

    consensus_labels = best_labels
    sil_score = best_sil

    # Only append to CSV if fam_id is not None
    if fam_id is not None:
        os.makedirs(os.path.dirname(labels_csv), exist_ok=True)
        append_to_csv(fam_id, consensus_labels, config['train_audio_list_file'], labels_csv)
        df = pd.read_csv(config['train_audio_list_file'], delimiter="\t")
        family_df = df[df['family_id'] == fam_id]
        #evaluate_and_log(config, family_df, fam_id, fam_emb, consensus_labels)
    return consensus_labels, sil_score

def enforce_max_cluster_size(labels, max_size):

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    new_labels = np.array(labels)
    next_label = max(labels) + 1
    for label, indices in label_to_indices.items():
        if len(indices) > max_size:
            # Split this cluster into chunks of max_size
            chunks = [indices[i:i + max_size] for i in range(0, len(indices), max_size)]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    continue  # Keep the original label for the first chunk
                new_labels[chunk] = next_label
                next_label += 1
    return new_labels

def get_audio_paths_for_family(audio_list_file, family_id):
    df = pd.read_csv(audio_list_file, delimiter="\t")
    return list(df[df["family_id"] == family_id]["path"])


def lr_lambda(current_step: int):
    if current_step > 500: #number of steps until cooldown
        return float(float(500) / current_step)
    else:
        return float(current_step / float(500))


class DropoutWrapper(nn.Module):
    def __init__(self, base_model, p=0.3):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    main()