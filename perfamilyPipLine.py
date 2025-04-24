import os
import pandas as pd
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from statistics import mode
from SV import *
from adaptation import *
from model_funcs import *
from clustering import *
from utils import *
from sklearn.metrics import silhouette_score


def main(run, config_file='/app/config.yaml'):
    """
    Main function to process families and train the model.

    Args:
        run: wandb run instance.
        config_file (str): Path to the configuration file.
    """
    # Load configuration settings
    config = load_config(config_file)

    # Load the speaker model
    speaker_model = load_model_ecapa_from_speechbrain(config)

    # Iterate through epochs
    for epoch in range(config['adaptation']['num_epochs']):
        process_families(run, config, speaker_model, epoch)

        # Extract embeddings from audio files after processing all families
        extract_family_embeddings(
            speaker_model,
            config['audio_dir'],
            config['train_audio_list_file'],
            config['embedding_output_file'],
            config['familes_emb']
        )

    # Save the fine-tuned model locally
    Save_Model_Localy(speaker_model, config)

    # Finish the wandb run
    run.finish()


def process_families(run, config, speaker_model, epoch):
    """
    Processes all families for a given epoch.

    Args:
        run: wandb run instance.
        config (dict): Configuration dictionary.
        speaker_model: Speaker model instance.
        epoch (int): Current epoch number.
    """
    # Read the CSV file containing paths to all families
    families_df = pd.read_csv(config['familes_emb'], delimiter="\t")
    labels_csv_file_path = f"{config['familes_labels']}_epoch_{epoch}.csv"

    # Clean the content of the CSV file
    clean_csv(labels_csv_file_path)

    # Process each family
    for _, row in families_df.iterrows():
        family_id = row['family_number']
        family_path = row['embedding_path']

        # Load family embeddings
        print(f"Processing family {family_id}")
        family_emb = np.load(family_path)

        # Get the number of unique speakers in the family
        num_of_speakers = get_unique_speakers_in_family(config['train_audio_list_file'], family_id)

        # Process the family pipeline
        process_single_family(config, labels_csv_file_path, speaker_model, family_id, family_emb, num_of_speakers)


def process_single_family(config, labels_csv_file_path, speaker_model, family_id, family_emb, num_of_speakers):
    """
    Processes a single family.

    Args:
        config (dict): Configuration dictionary.
        labels_csv_file_path (str): Path to the labels CSV file.
        speaker_model: Speaker model instance.
        family_id (int): Family ID.
        family_emb (np.ndarray): Family embeddings.
        num_of_speakers (int): Number of unique speakers in the family.
    """
    assert family_id >= 0, "Family ID must be a positive integer."
    assert num_of_speakers > 0, "Number of speakers must be greater than 0."

    # Cluster the embeddings
    labels, centroids = spectral_clustering(family_emb, num_of_speakers, family_id)
    assert len(family_emb) == len(labels), "Family embeddings and labels must have the same length."
    assert len(centroids) == num_of_speakers, "Number of centroids must match the number of speakers."

    # Ensure the output directory exists
    output_dir = os.path.dirname(config['familes_labels'])
    os.makedirs(output_dir, exist_ok=True)

    # Append clustering results to CSV
    append_to_csv(family_id, labels, config['train_audio_list_file'], labels_csv_file_path)

    # Evaluate clustering and log metrics
    evaluate_and_log(config, family_id, family_emb, num_of_speakers, labels, speaker_model)
    wandb.log({"family_id": family_id, "silhouette_score": silhouette_score(family_emb, labels)})

    # Process test samples for the family
    process_test_samples(config, speaker_model, family_id, family_emb, labels, num_of_speakers)

    # Train using triplet loss
    train_with_triplet_loss(config, speaker_model, family_id, labels_csv_file_path)


def process_test_samples(config, speaker_model, family_id, family_emb, labels, num_of_speakers):
    """
    Processes test samples for a given family.

    Args:
        config (dict): Configuration dictionary.
        speaker_model: Speaker model instance.
        family_id (int): Family ID.
        family_emb (np.ndarray): Family embeddings.
        labels (list): Cluster labels.
        num_of_speakers (int): Number of unique speakers in the family.
    """
    test_families_df = pd.read_csv(config['test_audio_list_file'], delimiter="\t")
    family_test_group = test_families_df[test_families_df["family_id"] == family_id]
    if family_test_group.empty:
        print(f"No test samples found for family_id {family_id}. Skipping...")
        return

    for _, row in family_test_group.iterrows():
        audio_path = os.path.join(config['audio_dir'], row['path'])
        test_emb = extract_single_embedding(speaker_model, audio_path)
        if test_emb is None:
            continue

        cenroid_label = identify_cluster(family_emb, labels, test_emb, method="centroid")
        KNN_label = identify_cluster(family_emb, labels, test_emb, method="knn", k=num_of_speakers)
        cosine_label = identify_cluster(family_emb, labels, test_emb, method="cosine")

        common_label = mode([cenroid_label, KNN_label, cosine_label])
        wandb.log({
            "family_id": family_id,
            "common_label": common_label,
            "cenroid_label": cenroid_label,
            "KNN_label": KNN_label,
            "cosine_label": cosine_label
        })


def train_with_triplet_loss(config, speaker_model, family_id, labels_csv_file_path):
    """
    Trains the model using triplet loss for a given family.

    Args:
        config (dict): Configuration dictionary.
        speaker_model: Speaker model instance.
        family_id (int): Family ID.
        labels_csv_file_path (str): Path to the labels CSV file.
    """
    try:
        triplet_data = create_triplets(family_id, pd.read_csv(labels_csv_file_path, delimiter="\t", on_bad_lines='skip'))
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file {labels_csv_file_path}: {e}")
        return

    if not triplet_data:
        print(f"No valid triplets found for family_id {family_id}. Skipping...")
        return

    dataset = TripletAudioDataset(speaker_model, triplet_data, config)
    if len(dataset) == 0:
        print(f"Dataset is empty for family_id {family_id}. Skipping...")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    triplet_loss = nn.TripletMarginLoss(margin=0.1)
    optimizer = optim.AdamW(speaker_model.parameters(), lr=1e-4, weight_decay=1e-2)

    for batch in dataloader:
        anchor_embeddings, positive_embeddings, negative_embeddings = batch
        anchor_embeddings.requires_grad_(True)
        positive_embeddings.requires_grad_(True)
        negative_embeddings.requires_grad_(True)

        # Compute the triplet loss
        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"family_id": family_id, "triplet_loss": loss.item()})


# Run the pipeline
if __name__ == '__main__':
    main()
