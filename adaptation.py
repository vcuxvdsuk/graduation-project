import librosa
import numpy as np
import torch
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def extract_mfcc(audio_path, sr=16000, n_mfcc=40):
    try:
        y, _ = librosa.load(audio_path, sr=sr)  # Load audio file
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Extract MFCC features
        return mfcc.T  # Transpose for proper format
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def create_triplets(audio_dataset):
    triplets = []
    speakers = list(audio_dataset.keys())

    for speaker in audio_dataset:
        pos_files = audio_dataset[speaker]
        if len(pos_files) < 2:
            continue  # Need at least two samples for a positive pair

        anchor, positive = random.sample(pos_files, 2)
        negative_speaker = random.choice([s for s in speakers if s != speaker])
        negative = random.choice(audio_dataset[negative_speaker])

        triplets.append((anchor, positive, negative))

    return triplets

def process_triplet(triplet):
    anchor_mfcc = extract_mfcc(triplet[0])
    positive_mfcc = extract_mfcc(triplet[1])
    negative_mfcc = extract_mfcc(triplet[2])

    if anchor_mfcc is None or positive_mfcc is None or negative_mfcc is None:
        return None

    return torch.tensor(anchor_mfcc, dtype=torch.float32), \
           torch.tensor(positive_mfcc, dtype=torch.float32), \
           torch.tensor(negative_mfcc, dtype=torch.float32)

class TripletAudioDataset(Dataset):
    def __init__(self, triplet_list):
        self.triplet_list = triplet_list

    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, idx):
        triplet = process_triplet(self.triplet_list[idx])
        if triplet is None:
            return self.__getitem__((idx + 1) % len(self.triplet_list))  # Retry with the next triplet
        return triplet

    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)  # Distance between Anchor & Positive
        neg_dist = F.pairwise_distance(anchor, negative)  # Distance between Anchor & Negative
        loss = torch.mean(F.relu(pos_dist - neg_dist + self.margin))  # Triplet Loss
        return loss
