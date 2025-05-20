import numpy as np
import pandas as pd
import wandb
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

from sklearn.metrics import (
    roc_curve,
    silhouette_score,
    calinski_harabasz_score,
    confusion_matrix,
    balanced_accuracy_score,
)
import matplotlib.pyplot as plt
import torch


class Evaluations:
    """Misc. speaker‐verification and clustering metrics."""

    @staticmethod
    def calculate_eer(y_true, y_scores):    #לעבור על זה שוב
        """Find EER from ROC curve (FPR vs TPR)."""
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        return float((fpr[idx] + fnr[idx]) / 2)

    @staticmethod
    def calculate_ieer(y_true, y_scores, num_thresholds=1000):
        """
        Compute Identification Equal Error Rate (IEER).

        Args:
            y_true (array-like of 0/1): 1 for genuine (enrolled) trials, 0 for impostor (guest) trials.
            y_scores (array-like of float): similarity scores, higher means more likely genuine.
            num_thresholds (int): number of thresholds to sweep between min and max.

        Returns:
            ieer (float): IEER value (where FAR ≈ FNIR).
            threshold (float): threshold at which IEER occurs.
        """
        y_true   = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        # Split out genuine vs. impostor scores
        genuine   = y_scores[y_true == 1]
        impostor  = y_scores[y_true == 0]

        # Build threshold grid
        all_scores = np.concatenate([genuine, impostor])
        thrs = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)

        # Compute False Acceptance Rate (FAR) and False Negative Identification Rate (FNIR)
        fars  = np.array([np.mean(impostor  >= t) for t in thrs])
        fnirs = np.array([np.mean(genuine  <  t) for t in thrs])

        # Find threshold where |FAR - FNIR| is minimized
        diff_idx = np.nanargmin(np.abs(fars - fnirs))
        ieer      = 0.5 * (fars[diff_idx] + fnirs[diff_idx])
        best_thr  = thrs[diff_idx]

        return float(ieer)


    @staticmethod
    def build_verification_pairs(embeddings, labels):
        """
        Efficiently builds verification pairs using cosine similarity matrix.

        Args:
            embeddings (list or np.ndarray): List/array of embeddings.
            labels (list or np.ndarray): Corresponding labels.

        Returns:
            tuple: (y_true, y_scores)
        """
        embeddings = np.array(embeddings)
        labels = np.array(labels)

        sim_matrix = cosine_similarity(embeddings)
        i_upper, j_upper = np.triu_indices(len(labels), k=1)

        y_true = (labels[i_upper] == labels[j_upper]).astype(int)
        y_scores = sim_matrix[i_upper, j_upper]

        return y_true.tolist(), y_scores.tolist()

    @staticmethod
    def cluster_accuracy(true_labels, cluster_labels):
        """
        Compute clustering accuracy via Hungarian matching.
        Both inputs are 1D arrays of length N.
        """
        true_labels = np.asarray(true_labels)
        cluster_labels = np.asarray(cluster_labels)
        # build cost matrix
        D = max(true_labels.max(), cluster_labels.max()) + 1
        cost = np.zeros((D, D), dtype=int)
        for t, c in zip(true_labels, cluster_labels):
            cost[t, c] += 1
        # Hungarian to maximize matching → minimize -cost
        row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
        return cost[row_ind, col_ind].sum() / len(true_labels)


def evaluate_and_log(config, df, family_id, family_emb, predicted_labels, train=False):
    """
    - config['all_samples']: TSV with columns [..., family_id, client_id, path]
    - family_id:        integer or None
    - family_emb:       list/array of shape (N, D)
    - predicted_labels: list/array of cluster IDs length N
    - train:            bool, if True logs with 'train_' prefix
    """
    # 1. Load ground‐truth for this family
    if family_id is not None:
        df = df[df['family_id'] == family_id]
    true_ids = df['client_id'].astype('category').cat.codes.values
    N = len(true_ids)

    if N != len(predicted_labels):
        raise ValueError(f"Length mismatch: {N} true vs {len(predicted_labels)} predicted")

    evals = Evaluations()

    # 2. Clustering accuracy via Hungarian
    acc = evals.cluster_accuracy(true_ids, predicted_labels)

    # 3. Cluster validity metrics
    n_clusters = len(np.unique(predicted_labels))
    sil = silhouette_score(family_emb, predicted_labels) if n_clusters > 1 else -1
    cal = calinski_harabasz_score(family_emb, predicted_labels) if n_clusters > 1 else -1

    # 4. Speaker‐verification EER
    y_true, y_scores = evals.build_verification_pairs(family_emb, true_ids)
    eer = evals.calculate_eer(y_true, y_scores)

    # 5. Speaker identification IEER
    ieer = evals.calculate_ieer(true_ids, predicted_labels)

    # 6. Confusion matrix and balanced accuracy
    cm = confusion_matrix(true_ids, predicted_labels)
    bal_acc = balanced_accuracy_score(true_ids, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_img_path = "confusion_matrix.png"
    plt.savefig(cm_img_path)
    plt.close()

    prefix = "train_" if train else "val_"
    wandb.log({
        f"{prefix}family_{family_id}/cluster_accuracy": acc,
        f"{prefix}family_{family_id}/silhouette": sil,
        f"{prefix}family_{family_id}/calinski_harabasz": cal,
        f"{prefix}family_{family_id}/eer": eer,
        f"{prefix}family_{family_id}/ieer": ieer,
        f"{prefix}family_{family_id}/balanced_accuracy": bal_acc,
        f"{prefix}family_{family_id}/confusion_matrix": wandb.Image(cm_img_path),
    })

    logging.info(
        f"[Family {family_id}] "
        f"{prefix}CluAcc={acc:.4f}, Sil={sil:.4f}, Cal={cal:.4f}, "
        f"EER={eer:.4f}, IEER={ieer:.4f}, BalAcc={bal_acc:.4f}"
    )


def evaluate_model(config, speaker_brain, dataloader, epoch, train=False):
    all_preds = []
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for paths, numeric_labels in dataloader:
            if isinstance(numeric_labels, torch.Tensor):
                labels_tensor = numeric_labels.detach().clone()
            else:
                labels_tensor = torch.tensor(numeric_labels)
            emb = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths)
            if emb.size(0) == 0:
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
        wandb.log({"train EER": eer, "train IEER": ieer})
    else:
        wandb.log({"val EER": eer, "val IEER": ieer})

    plot_embeddings(np.array(all_embeddings), all_labels, epoch, train)