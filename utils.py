import os
import pandas as pd
import shutil
import csv
from collections import defaultdict
import wandb
from model_funcs import *
from clustering import *
from meta_data_preproccesing import *
from SV import *
from ploting_funcs import *
from adaptation import *
from statistics import mode

import numpy as np
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import confusion_matrix

from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import det_curve


def count_unique_speakers(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, delimiter="\t")
    
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
    def calculate_eer(self,y_true, y_scores):
        fpr, fnr, thresholds = det_curve(y_true, y_scores)
        # EER is the point where FPR = FNR
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
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


from sklearn.metrics import calinski_harabasz_score

def calinski_harabasz_index(X, cluster_labels):
    score = calinski_harabasz_score(X, cluster_labels)
    return score


def clean_csv(csv_path):
         # Clean the content of the CSV file
        with open(csv_path, 'w') as file:
            file.truncate()

#main logging function

def evaluate_and_log(config, family_id, family_emb, num_of_speakers, labels, speaker_model):
    evaluations = Evaluations()

    # Read the CSV file containing paths to all families
    df = pd.read_csv(config['train_audio_list_file'], delimiter="\t")
    
    # Filter to only include rows with the relevant family_id
    if family_id is not None:
        family_df = df[df['family_id'] == family_id]
    else:
        family_df = df

    # Extract true labels from the 'client_id' column
    true_labels = family_df['client_id'].values
    
    # Map true labels to integers
    label_mapping = {label: idx for idx, label in enumerate(np.unique(true_labels))}
    true_labels_mapped = np.array([label_mapping[label] for label in true_labels])
    
    # Ensure the lengths of true_labels and labels match
    if len(true_labels_mapped) != len(labels):
        raise ValueError(f"Mismatch in lengths. True Labels: {len(true_labels_mapped)}, Cluster Labels: {len(labels)}")

    # Calculate confusion matrix and accuracy
    cm = confusion_matrix(true_labels_mapped, labels)
    accuracy = accuracy_score(true_labels_mapped, labels)
    
    # Log the results to wandb
    wandb.log({
        f"confusion_matrix_id_{family_id}": wandb.plot.confusion_matrix(probs=None,
                                                        y_true=true_labels_mapped,
                                                        preds=labels,
                                                        class_names=list(label_mapping.keys()))
    })
    
    # Calculate EER
    assert len(family_emb) == len(labels), "Family embeddings and labels must have the same length"
    y_true, y_scores = build_verification_pairs(family_emb, labels)
    assert len(y_true) == len(y_scores), "y_true and y_scores must have the same length"
    eer = evaluations.calculate_eer(y_true, y_scores)

    # Log the averages EER values
    if family_id:
        wandb.log({
            f"family_{family_id}_eer": eer,
            f"family_{family_id}_accuracy": accuracy
        })
    else:
        wandb.log({
            "eer": eer,
            "accuracy": accuracy
        })



#evaluate the SV errors by creating confusion matrix for each label
def evaluate_SV_errors(config, labels, speaker_model, family_id, family_emb, num_of_speakers):
    # Read the CSV file containing paths to all families
    df = pd.read_csv(config['train_audio_list_file'], delimiter="\t")
    
    # Filter to only include rows with the relevant family_id
    if family_id:
        family_df = df[df['family_id'] == family_id]
    else:
        family_df = df

    # Extract file paths
    file_paths = family_df['path'].values
    
    # Initialize confusion matrix values for each true label
    confusion_matrices = {label: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for label in set(labels)}
    
    for true_label, file_path in zip(labels, file_paths):
        if isinstance(file_path, str) and file_path.strip():
            audio_path = os.path.join(config['audio_dir'], file_path)
            test_emb = extract_single_embedding(speaker_model, audio_path)
            if test_emb is not None:
                cenroid_label = identify_cluster(family_emb, labels, test_emb, method="centroid")
                KNN_label = identify_cluster(family_emb, labels, test_emb, method="knn", k=num_of_speakers)
                cosine_label = identify_cluster(family_emb, labels, test_emb, method="cosine")
                
                common_label = mode([cenroid_label, KNN_label, cosine_label])
                
                for label in confusion_matrices:
                    if common_label == true_label:
                        if true_label == label:
                            confusion_matrices[label]['tp'] += 1
                        else:
                            confusion_matrices[label]['tn'] += 1
                    else:
                        if true_label == label:
                            confusion_matrices[label]['fn'] += 1
                        else:
                            confusion_matrices[label]['fp'] += 1
    return confusion_matrices

# in case of label 2
#   true\label1   2   3   4
#    1       tn  fp  tn  tn
#    2       fn  tp  fn  fn
#    3       tn  fp  tn  tn
#    4       tn  fp  tn  tn

def append_to_csv(family_id, labels, audio_csv, csv_file='clustering_loss_preparation.csv'):
    # Read the audio CSV file to get the file paths
    audio_df = pd.read_csv(audio_csv, delimiter="\t")
    audio_paths = audio_df[audio_df['family_id'] == family_id]['path'].tolist()

    # Check if the lengths of labels and audio_paths match
    if len(labels) != len(audio_paths):
        print(f"Error: Mismatch in lengths.(append_to_csv func) Labels: {len(labels)}, Audio Paths: {len(audio_paths)}")
        return

    label_dict = defaultdict(list)
    for i, label in enumerate(labels):
        if i < len(audio_paths):
            label_dict[label].append(audio_paths[i])  # appending names from the CSV to label
        else:
            print(f"Warning: Index {i} out of range for audio_paths")

    # Check if the target CSV file exists, if not, create it and write the header
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        if not file_exists:
            # Write the header row
            writer.writerow(['label'] + [f'file_{i}' for i in range(max(len(files) for files in label_dict.values()))])

        # Write the data rows
        for label, audio_files in label_dict.items():
            writer.writerow([family_id] + [label] + audio_files)


def append_All_to_csv(labels, audio_csv, csv_file='clustering_loss_preparation.csv'):
    # Read the audio CSV file to get the file paths
    audio_df = pd.read_csv(audio_csv, delimiter="\t")
    audio_paths = audio_df['path'].tolist()

    # Check if the lengths of labels and audio_paths match
    if len(labels) != len(audio_paths):
        print(f"Error: Mismatch in lengths.(append_to_csv func) Labels: {len(labels)}, Audio Paths: {len(audio_paths)}")
        return

    label_dict = defaultdict(list)
    for i, label in enumerate(labels):
        if i < len(audio_paths):
            label_dict[label].append(audio_paths[i])  # appending names from the CSV to label
        else:
            print(f"Warning: Index {i} out of range for audio_paths")

    # Check if the target CSV file exists, if not, create it and write the header
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        if not file_exists:
            # Write the header row
            writer.writerow(['label'] + [f'file_{i}' for i in range(max(len(files) for files in label_dict.values()))])

        # Write the data rows
        for label, audio_files in label_dict.items():
            writer.writerow([label] + audio_files)



def build_verification_pairs(embeddings, labels):
    y_true = []
    y_scores = []

    # Compare all combinations of pairs
    for (i, j) in combinations(range(len(embeddings)), 2):
        emb_i, emb_j = embeddings[i], embeddings[j]
        label_i, label_j = labels[i], labels[j]

        # Cosine similarity (can also use Euclidean or other)
        score = cosine_similarity([emb_i], [emb_j])[0][0]

        # 1 = genuine match, 0 = impostor
        is_match = int(label_i == label_j)

        y_true.append(is_match)
        y_scores.append(score)

    return y_true, y_scores
