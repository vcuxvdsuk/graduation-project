
from speechbrain.pretrained import Tacotron2, HIFIGAN
from speechbrain.pretrained import SpeakerRecognition
import torch
import yaml
import torchaudio
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

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

# Load the ECAPA-TDNN speaker embedding model
def load_model_ASR(config):
    asr = SpeakerRecognition.from_hparams(source=config['speechbrain_model']['ASR'],
                                        savedir=config['speechbrain_model']['savedir'])
    return asr

def Save_Model_Localy(model,config,name = "fine_tuned_model.pth"):
    # Save model locally
    torch.save(model.state_dict(), f"{config['speechbrain_model']['savedir']}/{name}")

def add_noise_to_signal(signal, noise_factor=0.01):
    """Add Gaussian noise to the audio signal."""
    noise = torch.randn(signal.size()) * noise_factor
    noisy_signal = signal + noise
    # Ensure the signal stays within the valid range
    noisy_signal = torch.clamp(noisy_signal, -1.0, 1.0)
    return noisy_signal

def extract_single_embedding(speaker_model, audio_path, noise_factor= 0.01):
    try:
        # Check if the file exists
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return None

        # Load the audio signal
        signal, fs = torchaudio.load(audio_path)

        # Ensure the signal is in the expected format (mono)
        if signal.dim() > 1:
            signal = signal.mean(dim=0, keepdim=True)  # Convert stereo to mono if necessary

        # Add noise to the audio signal
        noisy_signal = add_noise_to_signal(signal, noise_factor)

        # Get the embedding for the noisy signal
        embedding = speaker_model.encode_batch(noisy_signal)

        # Return the embedding as a numpy array
        return embedding.squeeze().cpu().numpy()

    except Exception as e:
        print(f"Error processing audio file (extract_single_embedding) {audio_path}: {e}")
        return None


def extract_family_embeddings(speaker_model, audio_dir, file, emb_dir, save_csv_file,noise_factor = 0.01, re_do=False):
    family_embeddings = []
    family_ids = []
    embedding_paths = []

    try:
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
                embedding = speaker_model.encode_batch(noisy_signal)

                # Append the embedding to the list for this family
                family_embeddings_list.append(embedding.squeeze().cpu().numpy())

            # Store the embeddings for the entire family (as a list of arrays)
            family_embedding_array = np.array(family_embeddings_list)

            # Save the family embedding array to a file
            if re_do:
                family_emb_file = os.path.join(emb_dir, f"family_{family_id}_RE_embedding.npy")
            else:
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
