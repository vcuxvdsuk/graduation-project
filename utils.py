import os
import pandas as pd
import shutil
import csv
from collections import defaultdict

def count_unique_speakers(df):
    
    # Extract the number from the 'client_id' field (after the last underscore)
    df['speaker_number'] = df['client_id'].apply(lambda x: int(x.split('_')[-1]))
    
    # Count the number of unique speakers based on the extracted speaker numbers
    unique_speakers = df['speaker_number'].nunique()
    
    return unique_speakers


#gets the file and a family_id and return the num of uniqe client id of given family_id assuming the family_id are sorted
def get_unique_speakers_in_family(file_path, family_id):
    # Read the CSV file
    df = pd.read_csv(file_path, delimiter="\t")
    
    # Filter the DataFrame by the given family_id
    family_df = df[df['family_id'] == family_id]
    
    # Get the unique client_id values
    unique_clients = family_df['client_id'].nunique()
    
    return unique_clients


class Evaluations:
    # Method to calculate Equal Error Rate (EER)
    def calculate_eer(self, false_acceptance_rates, false_rejection_rates):
        eer = 0.0
        for far, frr in zip(false_acceptance_rates, false_rejection_rates):
            if far == frr:
                eer = far
                break
        return eer

    # Method to calculate False Acceptance Rate (FAR)
    def calculate_far(self, false_acceptances, total_impostor_attempts):
        if total_impostor_attempts == 0:
            return 0.0
        return false_acceptances / total_impostor_attempts

    # Method to calculate False Rejection Rate (FRR)
    def calculate_frr(self, false_rejections, total_genuine_attempts):
        if total_genuine_attempts == 0:
            return 0.0
        return false_rejections / total_genuine_attempts


# Function to append clustering results to a CSV file
def append_to_csv(family_id, labels, audio_csv, csv_file='clustering_loss_preparation.csv'):
    # Read the audio CSV file to get the file paths
    audio_df = pd.read_csv(audio_csv, delimiter="\t")
    audio_paths = audio_df[audio_df['family_id'] == family_id]['path'].tolist()

    label_dict = defaultdict(list)
    for i, label in enumerate(labels):
        label_dict[label].append(audio_paths[i])  # appending names from the CSV to label

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for label, audio_files in label_dict.items():
            writer.writerow([label] + audio_files)



import numpy as np
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def domain_shift_factor(baseline_scores, adapted_scores):
    """
    """
    # ����� ������ ����� ������ ������ ������� ����
    baseline_performance = np.mean(baseline_scores)
    adapted_performance = np.mean(adapted_scores)
    
    # ����� ����� ����� ������� - ���� ������
    dsf_improvement = ((adapted_performance - baseline_performance) / baseline_performance) * 100
    
    return dsf_improvement

#maybe?
def multilingual_accuracy(true_labels, predicted_labels, noise_levels, languages):
    """
    """
    # ���� ����
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    
    # ���� ��� ���
    language_accuracy = {}
    unique_languages = set(languages)
    for lang in unique_languages:
        lang_indices = [i for i, l in enumerate(languages) if l == lang]
        lang_accuracy = accuracy_score(
            [true_labels[i] for i in lang_indices],
            [predicted_labels[i] for i in lang_indices]
        )
        language_accuracy[lang] = lang_accuracy
    
    # ���� ��� ��� ���
    noise_accuracy = {}
    unique_noise_levels = set(noise_levels)
    for noise in unique_noise_levels:
        noise_indices = [i for i, n in enumerate(noise_levels) if n == noise]
        noise_acc = accuracy_score(
            [true_labels[i] for i in noise_indices],
            [predicted_labels[i] for i in noise_indices]
        )
        noise_accuracy[noise] = noise_acc
    
    return {
        "overall_accuracy": overall_accuracy,
        "language_accuracy": language_accuracy,
        "noise_accuracy": noise_accuracy
    }

# delete?
def cluster_metrics(embeddings, true_labels=None):
    """
    ���� ���� ������� ���� ������ ������ �� ������
    
    �������:
    embeddings (numpy.ndarray): ������ �� ������ ������ �� ������
    true_labels (numpy.ndarray, optional): ������ �������, �� ������
    
    �����:
    dict: ����� �� ���� Cluster Round Index ����� ������
    """
    from sklearn.cluster import KMeans
    
    # ������ ����� �������� ��� ���� ������� ���������
    if true_labels is not None:
        n_clusters = len(set(true_labels))
    else:
        # �� ��� ������, ������� ����� ��������� ������ ���� ��������
        # (����� ����� ���� ������ ������ ��� silhouette, elbow method ���')
        n_clusters = min(10, len(embeddings) // 5)  # ����� ���� ������
    
    # ����� ����� K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # ����� ���� ������
    silhouette = silhouette_score(embeddings, cluster_labels) if len(set(cluster_labels)) > 1 else 0
    
    # ����� Cluster Round Index (CRI)
    # ��� �� ����� �� ������ ������ ������ ������ �� ��������
    intra_cluster_distances = []
    for i in range(n_clusters):
        cluster_points = embeddings[cluster_labels == i]
        if len(cluster_points) > 1:
            centroid = cluster_points.mean(axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            intra_cluster_distances.append(np.mean(distances))
    
    # �� �� ����� ����� ��� �� ���� ������ ���
    if intra_cluster_distances:
        intra_dist = np.mean(intra_cluster_distances)
        
        # ���� ��� ����� ��������
        centroids = kmeans.cluster_centers_
        inter_dist = 0
        count = 0
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                inter_dist += np.linalg.norm(centroids[i] - centroids[j])
                count += 1
        
        if count > 0:
            inter_dist /= count
            # ����� �-CRI
            cri = (inter_dist - intra_dist) / max(inter_dist, intra_dist)
        else:
            cri = 0
    else:
        cri = 0
    
    return {
        "silhouette_score": silhouette,
        "cluster_round_index": cri
    }

#needs work
def closed_set_metrics(true_speaker_ids, predicted_speaker_ids, similarity_scores):
    """
    ���� ���� ����� ���� ������ ����� �����
    
    �������:
    true_speaker_ids (list): ���� ������� ��������
    predicted_speaker_ids (list): ���� ������� �����
    similarity_scores (numpy.ndarray): ������ ����� ��� �� ������� ��� ������� �������
    
    �����:
    dict: ����� �� ����, ����� ���� ������ ����� (IEER) ������ ������
    """
    # ����� ���� ������
    accuracy = accuracy_score(true_speaker_ids, predicted_speaker_ids)
    
    # ����� ����� ����� ������ ����� (IEER)
    # ������ ������ ���: 1 �� ������ ������ ������, 0 ����
    true_matches = []
    scores = []
    
    for i, true_id in enumerate(true_speaker_ids):
        for j, speaker_id in enumerate(set(true_speaker_ids)):
            true_matches.append(1 if true_id == speaker_id else 0)
            scores.append(similarity_scores[i, j])
    
    # calc error
    fpr, tpr, thresholds = roc_curve(true_matches, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    # ����� ����� ������
    false_acceptance = sum([1 for i, (true, pred) in enumerate(zip(true_speaker_ids, predicted_speaker_ids)) 
                          if true != pred])
    false_rejection = sum([1 for i, (true, pred) in enumerate(zip(true_speaker_ids, predicted_speaker_ids)) 
                          if true == pred and np.max(similarity_scores[i]) < 0.5])  # �� ����� ������: 0.5
    
    total_comparisons = len(true_speaker_ids)
    far = false_acceptance / total_comparisons  # False Acceptance Rate
    frr = false_rejection / total_comparisons   # False Rejection Rate
    
    return {
        "accuracy": accuracy,
        "identification_equal_error_rate": eer,
        "false_acceptance_rate": far,
        "false_rejection_rate": frr
    }

#not valid
def open_set_metrics(enrolled_embeddings, test_embeddings, enrolled_ids, test_ids, threshold=0.5):
    """
    ���� ���� openFEAT ������� ����� �����
    
    �������:
    enrolled_embeddings (numpy.ndarray): ������ ������ �� ������ ������
    test_embeddings (numpy.ndarray): ������ ������ �� ������ ������
    enrolled_ids (list): ���� ������� �������
    test_ids (list): ���� ������� ��� ������
    threshold (float): �� ����� ������ �����
    
    �����:
    dict: ����� �� ���� openFEAT ������ ������ ������ �����
    """
    # ����� ������ ����� ��� ������ ������ ������� �������
    similarity_matrix = np.zeros((len(test_embeddings), len(enrolled_embeddings)))
    for i, test_emb in enumerate(test_embeddings):
        for j, enrolled_emb in enumerate(enrolled_embeddings):
            # ������� ������ �������
            similarity_matrix[i, j] = np.dot(test_emb, enrolled_emb) / (
                np.linalg.norm(test_emb) * np.linalg.norm(enrolled_emb))
    
    # ����� ������: ���� �� ����� ������ �� ����� ����� �� ������ ����� �����
    predicted_ids = []
    for i in range(len(test_embeddings)):
        max_similarity_idx = np.argmax(similarity_matrix[i])
        max_similarity = similarity_matrix[i, max_similarity_idx]
        
        if max_similarity >= threshold:
            predicted_ids.append(enrolled_ids[max_similarity_idx])
        else:
            predicted_ids.append("unknown")  # ���� �� �����
    
    # ����� ����� ������ �������: ������ ��� ������
    known_indices = [i for i, id in enumerate(test_ids) if id in enrolled_ids]
    unknown_indices = [i for i, id in enumerate(test_ids) if id not in enrolled_ids]
    
    # ����� ���� ���� ���� ������ ������
    known_correct = 0
    for i in known_indices:
        if test_ids[i] == predicted_ids[i]:
            known_correct += 1
    
    known_accuracy = known_correct / len(known_indices) if known_indices else 0
    
    # ����� ��� ����� ���� �� ������ �� ������
    unknown_correct = 0
    for i in unknown_indices:
        if predicted_ids[i] == "unknown":
            unknown_correct += 1
    
    unknown_accuracy = unknown_correct / len(unknown_indices) if unknown_indices else 0
    
    # ����� ��� openFEAT
    # openFEAT ���� �� ������ ����� ������ ������ ������ ������ �� ������
    if known_indices and unknown_indices:
        openFEAT = (known_accuracy + unknown_accuracy) / 2
    elif known_indices:
        openFEAT = known_accuracy
    elif unknown_indices:
        openFEAT = unknown_accuracy
    else:
        openFEAT = 0
    
    return {
        "openFEAT": openFEAT,
        "known_speaker_accuracy": known_accuracy,
        "unknown_speaker_accuracy": unknown_accuracy,
        "overall_accuracy": (known_correct + unknown_correct) / len(test_ids) if test_ids else 0
    }