import numpy as np
import pandas as pd
import wandb
import logging

from sklearn.metrics import (
    roc_curve,
    silhouette_score,
    calinski_harabasz_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

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


class Evaluations:
    """Misc. speaker‐verification and clustering metrics."""

    @staticmethod
    def calculate_eer(y_true, y_scores):
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
        Return (y_true, y_scores) for all upper‐triangle pairs.
        embeddings: list/array of shape (N, D)
        labels:       array of shape (N,)
        """
        E = np.vstack(embeddings)
        L = np.asarray(labels)
        sim = cosine_similarity(E)
        i, j = np.triu_indices_from(sim, k=1)
        y_true = (L[i] == L[j]).astype(int)
        y_scores = sim[i, j]
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


def evaluate_and_log(config, family_id, family_emb, predicted_labels):
    """
    - config['all_samples']: TSV with columns [..., family_id, client_id, path]
    - family_id:        integer or None
    - family_emb:       list/array of shape (N, D)
    - predicted_labels: list/array of cluster IDs length N
    """
    # 1. Load ground‐truth for this family
    df = pd.read_csv(config['all_samples'], sep="\t")
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
    thresholds = np.linspace(0.5, 1.0, num=50)
    ieer = evals.calculate_ieer(true_ids, predicted_labels, thresholds)

    # 6. Log everything
    wandb.log({
        f"family_{family_id}/cluster_accuracy": acc,
        f"family_{family_id}/silhouette": sil,
        f"family_{family_id}/calinski_harabasz": cal,
        f"family_{family_id}/eer": eer,
        f"family_{family_id}/ieer": ieer,
    })

    logging.info(
        f"[Family {family_id}] "
        f"CluAcc={acc:.4f}, Sil={sil:.4f}, Cal={cal:.4f}, EER={eer:.4f}, IEER={ieer:.4f}"
    )
