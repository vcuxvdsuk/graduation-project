import os
import pandas as pd
import csv
from collections import defaultdict
import wandb
import numpy as np
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix, det_curve, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Utility Functions
def count_unique_speakers(file_path, delimiter="\t"):
    """
    Counts the number of unique speakers in a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        delimiter (str): Delimiter used in the CSV file (default: "\t").

    Returns:
        int: Number of unique speakers by counting the unique values in client_id.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the 'client_id' column is missing or improperly formatted.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, delimiter=delimiter)
    if 'client_id' not in df.columns:
        raise ValueError("The 'client_id' column is missing from the CSV file.")

    try:
        df['speaker_number'] = df['client_id'].apply(lambda x: int(x.split('_')[-1]))
    except Exception as e:
        raise ValueError(f"Error processing 'client_id' column: {e}")

    return df['speaker_number'].nunique()


def get_unique_speakers_in_family(file_path, family_id, delimiter="\t"):
    """
    Gets the number of unique client IDs for a given family ID.

    Args:
        file_path (str): Path to the CSV file.
        family_id (int): Family ID to filter by.
        delimiter (str): Delimiter used in the CSV file (default: "\t").

    Returns:
        int: Number of unique client IDs for the given family ID.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, delimiter=delimiter)
    family_df = df[df['family_id'] == family_id]
    return family_df['client_id'].nunique()


def clean_csv(csv_path):
    """
    Cleans the content of a CSV file by truncating it.

    Args:
        csv_path (str): Path to the CSV file.
    """
    try:
        with open(csv_path, 'w') as file:
            file.truncate()
    except Exception as e:
        logging.error(f"Error cleaning CSV file {csv_path}: {e}")



# CSV Utilities
def append_to_csv(family_id, labels, audio_csv, csv_file='clustering_loss_preparation.csv', delimiter="\t"):
    """
    Appends clustering results to a CSV file.

    Args:
        family_id (int): Family ID.
        labels (list): Cluster labels.
        audio_csv (str): Path to the audio CSV file.
        csv_file (str): Path to the output CSV file.
        delimiter (str): Delimiter used in the CSV file (default: "\t").
    """
    try:
        audio_df = pd.read_csv(audio_csv, delimiter=delimiter)
        audio_paths = audio_df[audio_df['family_id'] == family_id]['path'].tolist()

        if len(labels) != len(audio_paths):
            logging.error(f"Mismatch in lengths. Labels: {len(labels)}, Audio Paths: {len(audio_paths)}")
            return

        label_dict = defaultdict(list)
        for i, label in enumerate(labels):
            if i < len(audio_paths):
                label_dict[label].append(audio_paths[i])
            else:
                logging.warning(f"Index {i} out of range for audio_paths")

        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=delimiter)
            if not file_exists:
                writer.writerow(['label'] + [f'file_{i}' for i in range(max(len(files) for files in label_dict.values()))])

            for label, audio_files in label_dict.items():
                writer.writerow([family_id] + [label] + audio_files)

        #logging.info(f"Clustering results appended to {csv_file}")
    except Exception as e:
        logging.error(f"Error appending to CSV file {csv_file}: {e}")


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




    #   1 2   3   4   5
    #1 tp fn fn  fn  fn
    #2 fp 
    #3 fp
    #4 fp
    #5 fp
   