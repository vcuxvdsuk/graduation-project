from model_funcs import *
from clustering import *
from meta_data_preproccesing import *
from SV import *
from ploting_funcs import *
from adaptation import *
from evaluation import *
from utils import *
from statistics import mode

import wandb
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score

def main(run=None, config_file='/app/config.yaml'):
    config = load_config(config_file)
    speaker_model = load_model_ecapa_from_speechbrain(config)

    # Adaptation using triplet mining
    # This is the training class
    speaker_brain = modelTune(
        modules=speaker_model.mods,
        hparams={
            "compute_cost": nn.TripletMarginLoss(margin=0.8, p=2, swap=True, reduction="mean"),
            "opt_class": lambda x: torch.optim.AdamW(x, lr=1e-4, weight_decay=1e-2),
        },
        run_opts={"device": speaker_model.device},
    )

    for epoch in range(config['adaptation']['num_epochs']):
        logging.info(f"epoch: {epoch}")
        logging.info("embedding")
        extract_all_families_embeddings(
            speaker_brain,  # fine-tuned model
            config['audio_dir'],
            config['train_audio_list_file'],
            config['embedding_output_file'],
            config['familes_emb'])
        
        logging.info("testing")
        test_entire_database(config, speaker_model, epoch)  # Enter the pure model
        
        logging.info("training")
        train_per_family(config, speaker_brain, epoch)  # Enter the training class


    Save_Model_Localy(speaker_brain.modules.embedding_model, config, name="fine_tuned_model.pth")

    if run:
        run.finish()

def train_per_family(config, speaker_brain, epoch):
    families_df = pd.read_csv(config['familes_emb'], delimiter="\t")
    labels_csv_path = f"{config['familes_labels']}{epoch}.csv"
    clean_csv(labels_csv_path)

    for _, row in families_df.head(config['adaptation']['train_nums_of_families']).iterrows():
        family_id = row['family_number']
        family_emb = np.load(row['embedding_path'])
        num_speakers = get_unique_speakers_in_family(config['train_audio_list_file'], family_id)

        single_family_pipeline(
            config, labels_csv_path, speaker_brain, family_id, family_emb, num_speakers, to_plot=True
        )

def single_family_pipeline(config, labels_csv_path, speaker_brain, family_id, family_emb, num_speakers, to_plot):
    labels, _ = spectral_clustering(family_emb, num_speakers, family_id)
    os.makedirs(os.path.dirname(config['familes_labels']), exist_ok=True)

    append_to_csv(family_id, labels, config['train_audio_list_file'], labels_csv_path)
    evaluate_and_log(config, family_id, family_emb, num_speakers, labels)

    speaker_brain.on_stage_start(sb.Stage.TRAIN, epoch=None)

    miner = miners.TripletMarginMiner(margin=0.8, type_of_triplets="semi_hard")
    dataset = OnlineTripletDataset(get_audio_paths_for_family(config['train_audio_list_file'], family_id), labels)
    if len(dataset) == 0:
        print(f"Empty dataset for family {family_id}, skipping...")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for _ in range(config['adaptation']['epochs_per_family']):
        for paths, labels_tensor in dataloader:
            emb = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths)  # Use fine-tuned model here
            if not len(emb):
                print("No embeddings in batch, skipping...")
                continue

            labels_tensor = labels_tensor[:len(emb)].long()
            emb = F.normalize(emb, p=2, dim=1)

            triplet_idxs = miner(emb, labels_tensor)
            if triplet_idxs[0].numel() == 0:
                print("No valid triplets, skipping... len: ",len(triplet_idxs))
                continue

            anchor, positive, negative = emb[triplet_idxs[0]], emb[triplet_idxs[1]], emb[triplet_idxs[2]]
            loss = speaker_brain.hparams.compute_cost(anchor, positive, negative)
            
            speaker_brain.optimizer = speaker_brain.hparams.opt_class(speaker_brain.modules.parameters())  # Calling optimizer from factory
            speaker_brain.optimizer.zero_grad()
            loss.backward()
            speaker_brain.optimizer.step()

            wandb.log({"triplet_loss_online": loss.item()})


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

    evaluate_and_log(config, None, all_embeddings, num_speakers, labels)

    if to_plot:
        plot_embedding(config, all_embeddings, labels)


def family_test(speaker_model, config, family_id, test_group, family_emb, labels, num_speakers, to_plot):
    speaker_model.eval()

    for i, row in test_group.iterrows():
        test_emb = extract_single_embedding(speaker_model, os.path.join(config['audio_dir'], row['path']))
        if test_emb is not None:
            votes = [
                identify_cluster(family_emb, labels, test_emb.reshape(1, -1).detach().cpu().numpy(), method="centroid"),
                identify_cluster(family_emb, labels, test_emb.reshape(1, -1).detach().cpu().numpy(), method="knn", k=num_speakers),
                identify_cluster(family_emb, labels, test_emb.reshape(1, -1).detach().cpu().numpy(), method="cosine")]

            if to_plot:
                plot_embeddings_with_new_sample(config, family_emb, family_id, test_emb, i, labels)


def get_audio_paths_for_family(audio_list_file, family_id):
    df = pd.read_csv(audio_list_file, delimiter="\t")
    return list(df[df["family_id"] == family_id]["path"])


if __name__ == '__main__':
    main()
