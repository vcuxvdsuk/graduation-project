from model_funcs import *
import numpy as np
import torch
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
import pandas as pd

def create_triplets(family_id, df, target_sr=16000, target_length=2.0):
    triplets = []
    audio_dataset = defaultdict(list)

    # Convert the filtered DataFrame into a dictionary {label: [file_paths]}
    for row_family_id, row in df.iterrows():
        # Filter to only include rows with the relevant family_id
        if row_family_id != family_id:
            continue
        label = str(row.iloc[0])  # Ensure the label is a string
        file_paths = [fp for fp in row.iloc[1:] if isinstance(fp, str) and fp.strip()]
        if file_paths:
            audio_dataset[label].extend(file_paths)

    speakers = list(audio_dataset.keys())
    for speaker in audio_dataset:
        pos_files = audio_dataset[speaker]
        if len(pos_files) < 2:
            continue  # Skip if fewer than 2 samples for a positive pair

        # Create multiple triplets per speaker
        for _ in range(min(len(pos_files) // 2, 5)):  # Limits number of triplets per speaker
            anchor_path, positive_path = random.sample(pos_files, 2)

            # Ensure we pick a different speaker for the negative sample
            negative_speakers = [s for s in speakers if s != speaker]
            if not negative_speakers:
                continue  # Skip if no negative speaker is available

            negative_speaker = random.choice(negative_speakers)
            negative_path = random.choice(audio_dataset[negative_speaker])
            
            triplets.append((anchor_path, positive_path, negative_path))
    return triplets

def create_triplets_all_data(df, target_sr=16000, target_length=2.0):
    triplets = []
    audio_dataset = defaultdict(list)

    # Convert the DataFrame into a dictionary {label: [file_paths]}
    for _, row in df.iterrows():
        label = str(row.iloc[0])  # Ensure the label is a string
        file_paths = [fp for fp in row.iloc[1:] if isinstance(fp, str) and fp.strip()]
        if file_paths:
            audio_dataset[label].extend(file_paths)

    speakers = list(audio_dataset.keys())
    for speaker in audio_dataset:
        pos_files = audio_dataset[speaker]
        if len(pos_files) < 2:
            continue  # Skip if fewer than 2 samples for a positive pair

        # Create multiple triplets per speaker
        for _ in range(min(len(pos_files) // 2, 5)):  # Limits number of triplets per speaker
            anchor_path, positive_path = random.sample(pos_files, 2)

            # Ensure we pick a different speaker for the negative sample
            negative_speakers = [s for s in speakers if s != speaker]
            if not negative_speakers:
                continue  # Skip if no negative speaker is available

            negative_speaker = random.choice(negative_speakers)
            negative_path = random.choice(audio_dataset[negative_speaker])
            
            triplets.append((anchor_path, positive_path, negative_path))
    return triplets

def process_triplet(speaker_model, triplet, config):
    anchor_path, positive_path, negative_path = triplet
    try:
        anchor = extract_single_embedding(speaker_model, f"{config['audio_dir']}/{anchor_path}")
        positive = extract_single_embedding(speaker_model, f"{config['audio_dir']}/{positive_path}")
        negative = extract_single_embedding(speaker_model, f"{config['audio_dir']}/{negative_path}")
        return anchor, positive, negative
    except Exception as e:
        print(f"Error processing triplet(process_triplet) {triplet}: {e}")
        return None


class TripletAudioDataset(Dataset):
    def __init__(self,speaker_model, triplet_list, config):
        self.speaker_model = speaker_model
        self.triplet_list = triplet_list
        self.config = config

    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, idx):
        for _ in range(len(self.triplet_list)):
            triplet = process_triplet(self.speaker_model, self.triplet_list[idx], self.config)
            if triplet is not None:
                return triplet
            idx = (idx + 1) % len(self.triplet_list)
        raise RuntimeError("All triplets are invalid")


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)  # Distance between Anchor & Positive
        neg_dist = F.pairwise_distance(anchor, negative)  # Distance between Anchor & Negative
        loss = torch.mean(F.relu(pos_dist - neg_dist + self.margin))  # Triplet Loss
        return loss
