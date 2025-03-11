
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
def load_model_ecapa(config):
    speaker_model = SpeakerRecognition.from_hparams(source=config['speechbrain_model']['ecapa'],
                                                     savedir=config['speechbrain_model']['savedir'])
    return speaker_model

# Load the ECAPA-TDNN speaker embedding model
def load_model_ASR(config):
    asr = SpeakerRecognition.from_hparams(source=config['speechbrain_model']['ASR'],
                                        savedir=config['speechbrain_model']['savedir'])
    return asr

def extract_single_embedding(speaker_model, audio_path):
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

        # Get the embedding for the signal
        embedding = speaker_model.encode_batch(signal)

        # Return the embedding as a numpy array
        return embedding.squeeze().cpu().numpy()

    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None


def extract_family_embeddings(speaker_model, audio_dir, file, emb_dir, csv_file):
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

                # Ensure the signal is in the expected format (mono)
                if signal.dim() > 1:
                    signal = signal.mean(dim=0, keepdim=True)  # Convert stereo to mono if necessary

                # Get the embedding for the signal
                embedding = speaker_model.encode_batch(signal)

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
            print(f"Error processing family {family_id}: {e}")
            continue

    # Create a DataFrame for family embeddings
    embedding_df = pd.DataFrame({
        'family_number': family_ids,
        'embedding_path': embedding_paths
    })

    try:
        # Save the DataFrame to a CSV file
        embedding_df.to_csv(csv_file, index=False)
        print(f"CSV file saved to {csv_file}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
