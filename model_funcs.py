
import os
import yaml
import torch
import random
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import speechbrain as sb
from types import SimpleNamespace
from torch.nn.functional import pad
import torchaudio.transforms as transforms
from speechbrain.inference import Tacotron2, HIFIGAN
from speechbrain.inference import SpeakerRecognition

import logging

MAX_LEN = 16000 * 3  # 3 seconds at 16kHz

# Load configuration from the YAML file
def load_config(config_file='app/config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load the ECAPA-TDNN speaker embedding model
def load_model_ecapa_from_speechbrain(config):
    speaker_model = SpeakerRecognition.from_hparams(source=config['speechbrain_model']['ecapa'],
                                                     savedir=config['speechbrain_model']['savedir'])
    return speaker_model



def Save_Model_Localy(model, config, name="fine_tuned_model.pth"):
    save_path = os.path.join(config['speechbrain_model']['savedir'], name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
    torch.save(model.state_dict(), save_path)


def Load_Model_Localy(model, config, name="fine_tuned_model.pth"):

    model_path = os.path.join(config['speechbrain_model']['savedir'], name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        print(f"Model successfully loaded from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")
    
    return model

def add_noise_to_signal(signal, noise_factor=0.00):
    """Add random noise to the audio signal."""
    noise = torch.randn(signal.size()) * noise_factor
    noisy_signal = signal + noise
    # Ensure the signal stays within the valid range
    noisy_signal = torch.clamp(noisy_signal, -1.0, 1.0)
    return noisy_signal

def extract_single_embedding(speaker_model, audio_path, noise_factor=0.0):
    try:
        speaker_model.modules.embedding_model.eval()
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return None

        signal, fs = torchaudio.load(audio_path)
        
        if signal.dim() > 2:
            signal = signal.squeeze()  

        # Convert stereo to mono (always get [1, time])
        if signal.dim() > 1:
            signal = signal.mean(dim=0, keepdim=True)
        else:
            signal = signal.unsqueeze(0)  # ensure [1, time]

        # Check length after conversion
        if signal.numel() < 16000:
            print(f"Audio too short after mono conversion: {audio_path} ({signal.numel()} samples)")
            return None

        # Add noise and pad
        noisy_signal = add_noise_to_signal(signal, noise_factor)

        if noisy_signal.shape[-1] > MAX_LEN:
            noisy_signal = noisy_signal[..., :MAX_LEN]
        else:
            noisy_signal = pad(noisy_signal, (0, MAX_LEN - noisy_signal.shape[-1]))

        noisy_signal = noisy_signal.to(speaker_model.device)

        with torch.no_grad():
            feats = speaker_model.modules.compute_features(noisy_signal)
            feats = speaker_model.modules.mean_var_norm(feats, torch.tensor([1.0], device=speaker_model.device))

            emb = speaker_model.modules.embedding_model(feats)
            emb = speaker_model.modules.mean_var_norm_emb(emb, torch.tensor([1.0], device=speaker_model.device))
            emb = emb.squeeze(0)

        return emb.detach()

    except Exception as e:
        print(f"Error processing audio file (extract_single_embedding): {e}")
        return None


def extract_all_families_embeddings(speaker_model, audio_dir, file, emb_dir, 
                                    save_csv_file, noise_factor = 0.0, re_do=False):
    family_embeddings = []
    family_ids = []
    embedding_paths = []

    try:
        speaker_model.modules.embedding_model.eval()
        # Read the CSV file containing paths
        df = pd.read_csv(file, delimiter="\t")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Group the rows by family_id
    family_groups = df.groupby("family_id")

    # Loop through each family group
    for family_id, family_df in tqdm(family_groups):
        try:
            family_embeddings_list = []

            # Loop through each row in the family group
            for _, row in family_df.iterrows():
                # Construct the full path for the audio file
                audio_path = os.path.join(audio_dir, row["path"])

                # Check if the file exists
                if not os.path.exists(audio_path):
                    print(f"Audio file not found: {audio_path}")
                    continue

                # Load the audio signal
                signal, fs = torchaudio.load(audio_path)
                
                # Add noise to the audio signal
                noisy_signal = add_noise_to_signal(signal, noise_factor)

                # Ensure the signal is in the expected format (mono)
                if noisy_signal.dim() > 1:
                    noisy_signal = noisy_signal.mean(dim=0, keepdim=True)  # Convert stereo to mono if necessary
                
                # Get the embedding for the signal
                with torch.no_grad():
                    batch = SimpleNamespace(signal=(noisy_signal, torch.tensor([1.0])))
                    embedding, wav_lens, logits = speaker_model.compute_forward(batch, sb.Stage.VALID)

                # Append the embedding to the list for this family
                family_embeddings_list.append(embedding.squeeze().cpu().numpy())

            # Store the embeddings for the entire family (as a list of arrays)
            family_embedding_array = np.array(family_embeddings_list)

            # Save the family embedding array to a file
            family_emb_file = os.path.join(emb_dir, f"family_{family_id}_embedding.npy")

            np.save(family_emb_file, family_embedding_array)

            # Append the family information to the result lists
            family_embeddings.append(family_embedding_array)
            family_ids.append(family_id)
            embedding_paths.append(family_emb_file)

        except Exception as e:
            print(f"Error processing family(extract_family_embeddings) {family_id}: {e}")
            continue

    # Create a DataFrame for family embeddings
    embedding_df = pd.DataFrame({
        'family_number': family_ids,
        'embedding_path': embedding_paths
    })

    try:
        # Save the DataFrame to a CSV file
        embedding_df.to_csv(save_csv_file, sep='\t', index=False)
        print(f"CSV file saved to {save_csv_file}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        
        
def extract_batch_embeddings_train(speaker_brain, audio_dir, batch_paths, noise_factor=0.0, augment=False):
    # Set the model in training mode
    speaker_brain.modules.embedding_model.train()
    embeddings = []

    # Normalize batch_paths input
    if isinstance(batch_paths, tuple):
        batch_paths = list(batch_paths)


    # Construct full paths
    audio_paths = [os.path.join(audio_dir, path) for path in batch_paths]

    signals = []
    for audio_path in audio_paths:
        try:
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                continue

            signal, fs = torchaudio.load(audio_path)

            if signal.dim() > 1:
                signal = signal.mean(dim=0, keepdim=True)

            if signal.numel() < 16000:
                print(f"Audio too short: {audio_path}")
                continue

            # Add noise
            if augment:
                signal = augment_waveform(signal)
            noisy_signal = add_noise_to_signal(signal, noise_factor)
            
            # Ensure the signal is in the expected format (mono)
            if noisy_signal.dim() > 1:
                noisy_signal = noisy_signal.mean(dim=0, keepdim=True)  # Convert stereo to mono if necessary

            # Truncate or pad to MAX_LEN
            if noisy_signal.shape[-1] > MAX_LEN:
                noisy_signal = noisy_signal[..., :MAX_LEN]
            else:
                pad_size = MAX_LEN - noisy_signal.shape[-1]
                noisy_signal = pad(noisy_signal, (0, pad_size))

            signals.append(noisy_signal.unsqueeze(0))  # Add batch dim

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    if len(signals) < 2:
        print(f"Too few valid signals in batch (found {len(signals)}), skipping this batch.")
        return torch.empty((0, 192), device=speaker_brain.device)

    # Prepare batch
    batch_signals = torch.cat(signals, dim=0).to(speaker_brain.device)  # Use speaker_brain.device
    wav_lens = torch.ones(batch_signals.shape[0], device=speaker_brain.device)
    
    # Ensure parameters are trainable
    for param in speaker_brain.modules.embedding_model.parameters():
        param.requires_grad = True

    # Forward pass through model
    with torch.set_grad_enabled(True):  # Enable gradients
        batch_signals = batch_signals.squeeze(1)

        feats = speaker_brain.modules.compute_features(batch_signals)

        emb = speaker_brain.modules.embedding_model(feats)  #shape [batch,fatures,?]

        emb = speaker_brain.modules.mean_var_norm_emb(emb, wav_lens)
        emb = emb.squeeze(1)

    return emb



def augment_waveform(wav, sample_rate = 16000):
    # 1. Additive noise
    #logging.info(f"1")
    noise = torch.randn_like(wav) * 0.01
    wav = wav + noise

    # 2. Reverberation (RIR conv)
    # assume `rir` is a preloaded tensor [1, L]
    # wav = torch.nn.functional.conv1d(wav.unsqueeze(0), rir.unsqueeze(0)).squeeze(0)

    # 3. Speed perturbation
    #logging.info(f"3")
    speed = random.choice([0.9, 1.0, 1.1])
    wav = torchaudio.functional.resample(wav, sample_rate, int(sample_rate*speed))

    # 5. Volume scaling
    #logging.info(f"5")
    gain = random.uniform(0.8, 1.2)
    wav = wav * gain

    return wav