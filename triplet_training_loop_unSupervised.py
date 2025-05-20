
from model_funcs import *
from clustering import *
from meta_data_preproccesing import *
from SV import *
from ploting_funcs import *
from adaptation import *
from evaluation import *
from utils import *

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import speechbrain as sb


def main(run=None, config_file='/app/config.yaml'):
    # Load configuration
    config = load_config(config_file)

    # Initialize model and embedding sub-module
    speaker_model = load_model_ecapa_from_speechbrain(config)
    embedding_model = speaker_model.mods.embedding_model
    for p in embedding_model.parameters():
        p.requires_grad = True

    # Loss functions
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2, swap=True, reduction="mean")
    ssreg_loss_fn = nn.MSELoss(reduction="mean")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        embedding_model.parameters(),
        lr=1e-3, weight_decay=1e-1
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / 1000, 1.0))

    # Wrap in SpeechBrain brain for training loops
    speaker_brain = modelTune(
        modules=speaker_model.mods,
        opt_class=lambda args: optimizer,
        hparams={
            "triplet_loss": triplet_loss_fn,
            "ssreg_loss":    ssreg_loss_fn,
            "ssreg_weight":  2,
            "optimizer":    optimizer,
            "scheduler":    scheduler
        },
        run_opts={"device": speaker_model.device}
    )

    # Prepare full-dataset paths and pseudo-labels
    df = pd.read_csv(config['train_audio_list_file'], sep="\t")
    audio_paths = df['path'].tolist()

    num_clusters = count_unique_speakers(config['train_audio_list_file'])


    # Initialize Weights & Biases run
    if run:
        wandb.init(config=config)
    

    miner           = miners.TripletMarginMiner(margin=1.0, type_of_triplets="semihard")
    triplet_loss_fn = speaker_brain.hparams.triplet_loss
    ssreg_loss_fn   = speaker_brain.hparams.ssreg_loss
    ssreg_weight    = speaker_brain.hparams.ssreg_weight
    optimizer       = speaker_brain.hparams.optimizer
    scheduler       = speaker_brain.hparams.scheduler
    
    # Training loop over entire dataset with pseudo labels
    for epoch in range(config['adaptation']['num_epochs']):
        logging.info(f"[Epoch {epoch}] full dataset")

        with torch.no_grad():
            #run test every 5 epochs
            if epoch%5==0:
                logging.info(f"[Epoch {epoch}] Test full DB")
                test_entire_database(config, speaker_model, epoch)
            

        # collect embeddings for clustering
        speaker_model.eval()
        families_df = pd.read_csv(config['familes_emb'], delimiter="\t")

        all_embeddings = [
            np.load(row['embedding_path']) for _, row in families_df.iterrows()
        ]
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        # Spectral clustering to obtain pseudo labels
        logging.info("Performing spectral clustering...")
        labels, _ = spectral_clustering(all_embeddings, num_clusters, "all")

        # Create Dataset & DataLoader
        full_dataset = OnlineTripletDataset(audio_paths, labels)
        loader = DataLoader(
            full_dataset,
            batch_size=16,
            shuffle=True
        )

        logging.info(f"[Epoch {epoch}] Training on entire dataset with pseudo labels")
        speaker_brain.on_stage_start(sb.Stage.TRAIN)
        
        for paths, labels in loader:
            embeddings = extract_batch_embeddings_train(
                speaker_brain,
                config['audio_dir'],
                paths
            )
            if embeddings.size(0) == 0:
                continue

            labels = labels.long()
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Triplet mining and loss
            a, p, n = miner(embeddings, labels)
            if a.numel() > 0:
                t_loss = triplet_loss_fn(embeddings[a], embeddings[p], embeddings[n])
            else:
                t_loss = torch.tensor(0.0)

            # Self-supervised loss (SSReg)
            aug_embeddings = extract_batch_embeddings_train(speaker_brain,
                                                            config['audio_dir'],
                                                            paths,0.05,True)
            aug_embeddings = F.normalize(aug_embeddings, p=2, dim=1)
            ssreg_loss = ssreg_loss_fn(embeddings, aug_embeddings)

            # Combined loss
            loss = 10* t_loss + ssreg_weight * ssreg_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Log metrics
            if run:
                wandb.log({
                    'triplet_loss': t_loss.item(),
                    'ssreg_loss': ssreg_loss.item(),
                    'combined_loss': loss.item()
                })

        with torch.no_grad():

            logging.info(f"[Epoch {epoch}] Extract embeddings")
            extract_all_families_embeddings(speaker_brain,  # fine-tuned model
                                            config['audio_dir'],
                                            config['train_audio_list_file'],
                                            config['embedding_output_file'],
                                            config['familes_emb'])
        

    # Save the fine-tuned embedding model
    Save_Model_Localy(
        embedding_model,
        config,
        name="fine_tuned_full_dataset_unsupervised.pth")

    if run:
        wandb.finish()


def test_entire_database(config, speaker_model, epoch):
    speaker_model.eval()
    families_df = pd.read_csv(config['familes_emb'], delimiter="\t")
    labels_csv_path = f"{config['familes_labels']}_entire_database_{epoch}.csv"
    clean_csv(labels_csv_path)

    all_embeddings = []
    for _, row in families_df.iterrows():
        all_embeddings.extend(np.load(row['embedding_path']))

    all_embeddings = np.array(all_embeddings)
    num_speakers = count_unique_speakers(config['train_audio_list_file'])

    process_entire_dataset(config, labels_csv_path, speaker_model, all_embeddings, num_speakers, to_plot=True)


def process_entire_dataset(config, labels_csv_path, speaker_model, all_embeddings, num_speakers, to_plot):

    labels, _ = spectral_clustering(all_embeddings, num_speakers, "entire_dataset")
    os.makedirs(os.path.dirname(config['familes_labels']), exist_ok=True)
    append_All_to_csv(labels, config['train_audio_list_file'], labels_csv_path)

    evaluate_and_log(config, None, all_embeddings, labels)

    if to_plot:
        plot_embedding(config, all_embeddings, labels)


if __name__ == '__main__':
    main()
