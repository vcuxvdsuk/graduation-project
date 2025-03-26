from model_funcs import *
from clustering import *
from meta_data_preproccesing import *
from SV import *
from ploting_funcs import *
from adaptation import *
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
                    # Set the wandb entity where your project will be logged (generally your team name).
                    entity="oribaruch-engineering",
                    # Set the wandb project where this run will be logged.
                    project="graduation_project",
                    # Track hyperparameters and run metadata.
                    #   config=config
                    )
    
    #preproccess_pipline(config)

    # Load model
    speaker_model = load_model_ecapa(config)

    # Read the CSV file containing paths to all families
    families_df = pd.read_csv(config['familes_emb'], delimiter="\t")

#######################################################################################################
    iter_num = 0
    for epoch in range(config['num_epochs']):
        # Loop through each family
        for index, row in families_df.head(10).iterrows():
            family_id = row['family_number']  
            family_path = row['embedding_path']

            #   1) load a single family embeddings as numpy array
            print(f"Processing family {family_id}")
            family_emb = np.load(family_path)
            num_of_speakers = get_unique_speakers_in_family(config['train_audio_list_file'], family_id)

    ######################################################################################################
            #steps 2-7
            single_family_pipeline(config,iter_num,speaker_model, family_id, 
                                   family_emb, num_of_speakers,False)
    #####################################################################################################
        #   8) repeat the process for all the families
        iter_num += 1


#####################################################################################################    
        # embedding from the audio file - dosnt need to run every time
    #extract_family_embeddings(speaker_model,
     #                       config['audio_dir'],
      #                      config['train_audio_list_file'],
       #                     config['embedding_output_file'],
        #                    config['familes_emb'])
    # Finish the run and upload any remaining data.
    run.finish()

# Run the pipeline
if __name__ == '__main__':
    main()






def single_family_pipeline(config, iter_num, speaker_model, family_id, 
                           family_emb, num_of_speakers, to_plot):

######################################################################################################
    #   2) cluster the embeddings
    labels, centroids = spectral_clustering(family_emb,num_of_speakers,family_id)
    # Log clustering results to wandb
    wandb.log({"family_id": family_id, "silhouette_score": silhouette_score(family_emb, labels)})

    # Append clustering results to CSV

    append_to_csv(family_id, labels, config['test_audio_list_file'], config["familes_labels"]+str(iter_num))

#####################################################################################################
    #   3) evaluate the clustering
#####################################################################################################
    #   4) extract embeddings from the all audio test files of corresponding family
    # Read the CSV file containing paths to all families test audio files
    test_families_df = pd.read_csv(config['test_audio_list_file'], delimiter="\t")
    family_test_group = test_families_df[test_families_df["family_id"] == family_id]
    test_sample_num = 0
    for _, row in family_test_group.iterrows():
        audio_path = os.path.join(config['audio_dir'], row['path'])
        test_emb = extract_single_embedding(speaker_model, audio_path)
        if test_emb is not None:
            test_sample_num += 1
    #####################################################################################################
    #   5) check extracted embeddings simmelirity to the members of the family embeddings
            cenroid_label = identify_cluster(family_emb, labels, test_emb, method="centroid")
            KNN_label = identify_cluster(family_emb, labels, test_emb, method="knn", k=num_of_speakers)
            cosine_label = identify_cluster(family_emb, labels, test_emb, method="cosine")
                
            # Choose the most common label
            common_label = mode([cenroid_label, KNN_label, cosine_label])
             # Log the common label to wandb
            wandb.log({"common_label": common_label, "cenroid_label": cenroid_label,
                        "KNN_label": KNN_label, "cosine_label": cosine_label})

#####################################################################################################
    #   6) eveluation and fine tune
    
    # Create triplets
    triplet_data = create_triplets(pd.read_csv(config["familes_labels"]+str(iter_num), delimiter="\t"))
    
    # Create dataset and dataloader
    dataset = TripletAudioDataset(triplet_data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    ##############################
    criterion = nn.TripletMarginLoss(margin=0.3)

    # Define Optimizers
    ##############################
    # Initialize the TripletLoss
    triplet_loss = TripletLoss(margin=1.0)

    # Define the optimizer
    optimizer = optim.AdamW(speaker_model.parameters(), lr=1e-4, weight_decay=1e-2)
    # Training
    for batch in dataloader:
        anchor_batch, positive_batch, negative_batch = batch

        # Compute embeddings
        anchor_embeddings = speaker_model(anchor_batch)
        positive_embeddings = speaker_model(positive_batch)
        negative_embeddings = speaker_model(negative_batch)

        # Compute the triplet loss
        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the loss to wandb
        wandb.log({"triplet_loss": loss.item()})


#####################################################################################################
    #   7) plot the clustering results
    if to_plot:
        plot_embeddings_with_new_sample(cenroid_label, KNN_label, cosine_label, family_emb, family_id, test_emb, test_sample_num, labels)


