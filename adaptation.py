from model_funcs import *
import numpy as np
import torch
import random
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
import pandas as pd
import os
from pytorch_metric_learning import miners
from pytorch_metric_learning.utils.loss_and_miner_utils import get_all_triplets_indices
from torch.utils.data import Dataset, DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_audio_dataset(df, filter_family_id=None):
    """Convert DataFrame to label->file_paths dictionary."""
    audio_dataset = defaultdict(list)
    for row_family_id, row in df.iterrows():
        if filter_family_id is not None and row_family_id != filter_family_id:
            continue
        label = str(row.iloc[0])
        file_paths = [fp for fp in row.iloc[1:] if isinstance(fp, str) and fp.strip()]
        if file_paths:
            audio_dataset[label].extend(file_paths)
    return audio_dataset



def process_triplet(speaker_model, triplet, config):
    anchor_path, positive_path, negative_path = triplet
    try:
        audio_dir = config.get("audio_dir", "")
        anchor = extract_single_embedding(speaker_model, os.path.join(audio_dir, anchor_path))
        positive = extract_single_embedding(speaker_model, os.path.join(audio_dir, positive_path))
        negative = extract_single_embedding(speaker_model, os.path.join(audio_dir, negative_path))
        return anchor, positive, negative
    except Exception as e:
        print(f"[process_triplet ERROR] {triplet} -> {e}")
        return None


class OnlineTripletDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        return self.audio_paths[idx], self.labels[idx]



#################################################################################################################
####################### model
#################################################################################################################

from speechbrain.lobes.features import Fbank
import torch
import speechbrain as sb
import torch


class modelTune(sb.Brain):
    def on_stage_start(self, stage, epoch):
        # Enable grad for fine-tuning during training, and freeze other modules
        if stage == sb.Stage.TRAIN:
            for module in [
                self.modules.embedding_model,
                self.modules.classifier,
            ]:
                for p in module.parameters():
                    p.requires_grad = True

            # Optionally freeze feature extractor and normalization layers
            for module in [
                self.modules.compute_features,
                self.modules.mean_var_norm,
                self.modules.mean_var_norm_emb,
            ]:
                for p in module.parameters():
                    p.requires_grad = False

    def compute_forward(self, batch, stage):
        """Forward computation for speaker embeddings or classification."""
        wavs, wav_lens = batch.signal
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # Feature extraction
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, wav_lens)

        # Embedding extraction
        embeddings = self.modules.embedding_model(feats)

        # Optional normalization
        embeddings = self.modules.mean_var_norm_emb(embeddings, wav_lens)

        # Classification head (add classifier after obtaining embeddings)
        if stage == sb.Stage.TRAIN or stage == sb.Stage.VALID:
            logits = self.modules.classifier(embeddings)
        else:
            logits = None

        return embeddings, wav_lens, logits

    def compute_objectives(self, predictions, batch, stage):
        """Computes classification loss (e.g., cross-entropy)."""
        embeddings, wav_lens, logits = predictions
        _, targets = batch.class_labels
        return self.hparams.compute_cost(logits, targets)


    ##################################3
from torch.nn.functional import normalize
import torch.nn.functional as F

class AMSoftmaxHead(nn.Module):
    def __init__(self, in_features, out_features, margin=0.2, scale=30):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, x, labels):
        x = normalize(x)
        w = normalize(self.weight)
        logits = torch.matmul(x, w.T)
        logits = logits - self.margin * F.one_hot(labels, num_classes=w.shape[0])
        return logits * self.scale
