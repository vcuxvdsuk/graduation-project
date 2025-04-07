from model_funcs import *
from clustering import *
from meta_data_preproccesing import *
from SV import *
from ploting_funcs import *
from adaptation import *
from utils import *
from statistics import mode
import wandb
import os
import torch.nn.functional as F
import csv
import torch
import torch.nn as nn
import torch.optim as optim


def main(config_file='/app/config.yaml'):
    # Load configuration settings
    config = load_config(config_file)
    
    # Initialize wandb
    run = wandb.init(                    
                    entity="oribaruch-engineering",     # the wandb entity where the project is logged
                    project="graduation_project",   # the wandb project where this run is logged.
                    )
    
    # Load model
    speaker_model = load_model_ecapa_from_speechbrain(config)

    for epoch in range(config['adaptation']['num_epochs']):
        """
        # Extract embeddings from the audio files
        extract_family_embeddings(speaker_model,
                                  config['audio_dir'],
                                  config['train_audio_list_file'],
                                  config['embedding_output_file'],
                                  config['familes_emb'])
        """
        # Read the CSV file containing paths to all families
        families_df = pd.read_csv(config['familes_emb'], delimiter="\t")
        labels_csv_file_path = f"{config['familes_labels']}_epoch_{epoch}.csv"

        # Clean the content of the CSV file
        clean_csv(labels_csv_file_path)

        # Load all family embeddings into a single array
        all_embeddings = []
        for index, row in families_df.iterrows():
            family_id = row['family_number']  
            family_path = row['embedding_path']

            # Load a single family embeddings as numpy array
            print(f"Processing family {family_id}")
            family_emb = np.load(family_path)
            for emb in family_emb:
                all_embeddings.append(emb)

        all_embeddings = np.array(all_embeddings)
        # Get the total number of speakers in the dataset
        num_of_speakers = count_unique_speakers(config['train_audio_list_file'])

        # Process the entire dataset
        process_entire_dataset(config, labels_csv_file_path, speaker_model,
                              all_embeddings, num_of_speakers, True)

    Save_Model_Localy(speaker_model, config)

    # Finish the run and upload any remaining data.
    run.finish()


def process_entire_dataset(config, labels_csv_file_path, speaker_model, 
                           all_embeddings, num_of_speakers, to_plot ):

    # Cluster the embeddings
    labels, centroids = spectral_clustering(all_embeddings, num_of_speakers, "entire_dataset")

    # Ensure the directory exists
    output_dir = os.path.dirname(config['familes_labels'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Append clustering results to CSV
    append_All_to_csv(labels, config['train_audio_list_file'], labels_csv_file_path)

    # Evaluate clustering
    evaluate_and_log(config, None, all_embeddings, num_of_speakers, labels, speaker_model)
    wandb.log({"dataset": "entire_dataset", "silhouette_score": silhouette_score(all_embeddings, labels)})

    # Read the CSV file containing paths to all test audio files
    test_families_df = pd.read_csv(config['test_audio_list_file'], delimiter="\t")
    test_sample_num = 0

    #if to_plot:
    plot_embedding(config, all_embeddings, labels)

    for _, row in test_families_df.iterrows():
        audio_path = os.path.join(config['audio_dir'], row['path'])
        test_emb = extract_single_embedding(speaker_model, audio_path)
        if test_emb is not None:
            test_sample_num += 1

            cenroid_label = identify_cluster(all_embeddings, labels, test_emb, method="centroid")
            KNN_label = identify_cluster(all_embeddings, labels, test_emb, method="knn", k=num_of_speakers)
            cosine_label = identify_cluster(all_embeddings, labels, test_emb, method="cosine")

            common_label = mode([cenroid_label, KNN_label, cosine_label])
            wandb.log({"common_label": common_label, "cenroid_label": cenroid_label,
                        "KNN_label": KNN_label, "cosine_label": cosine_label})

    # Create triplets
    try:
        triplet_data = create_triplets_all_data(pd.read_csv(labels_csv_file_path, delimiter="\t", on_bad_lines='skip'))
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file {labels_csv_file_path}: {e}")
        return
    if not triplet_data:
        print(f"No valid triplets found for the entire dataset. Skipping...\n ")
        return

    dataset = TripletAudioDataset(speaker_model, triplet_data, config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    triplet_loss = nn.TripletMarginLoss(margin=0.1)
    optimizer = optim.AdamW(speaker_model.parameters(), lr=1e-4, weight_decay=1e-2)

    for batch in dataloader:
        anchor_embeddings, positive_embeddings, negative_embeddings = batch
        # Ensure the tensors require gradients
        anchor_embeddings = anchor_embeddings.clone().detach().requires_grad_(True)
        positive_embeddings = positive_embeddings.clone().detach().requires_grad_(True)
        negative_embeddings = negative_embeddings.clone().detach().requires_grad_(True)

        # Compute the triplet loss
        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"triplet_loss": loss.item()})

# Run the pipeline
if __name__ == '__main__':
    main()

