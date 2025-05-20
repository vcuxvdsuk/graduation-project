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
from pytorch_metric_learning import miners
import torchaudio.transforms as T


def main(run=None, config_file='/app/config.yaml'):
    config = load_config(config_file)
    speaker_model = load_model_ecapa_from_speechbrain(config)
    embedding_model = speaker_model.mods.embedding_model

    # leave embeddor trainable
    for p in embedding_model.parameters():
        p.requires_grad = True

    # instantiate both losses
    num_speakers = count_unique_speakers(config['train_audio_list_file'])
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.3, p=2, swap=True, reduction="mean")
    soft_loss_fn   = AMSoftmaxLoss(192, num_speakers, s=30.0, m=1.0) #angular margin loss
    ssreg_loss_fn   = nn.MSELoss(reduction="mean")

    # single optimizer over both embedding model + AM‑Softmax weights
    optimizer = torch.optim.AdamW(
        list(embedding_model.parameters()) + [soft_loss_fn.weight], lr=1e-3, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    speaker_brain = modelTune(
        modules=speaker_model.mods,
        hparams={
            "triplet_loss":     triplet_loss_fn,
            "soft_weight":      0.5,
            "soft_loss":        soft_loss_fn,
            "ssreg_weight":     0.5,
            "ssreg_loss":       ssreg_loss_fn,
            "optimizer":        optimizer,
            "scheduler":        scheduler
        },
        run_opts={"device": speaker_model.device}
    )

    for epoch in range(config['adaptation']['num_epochs']):
        
        logging.info(f"[Epoch {epoch}] Adapt per family")
        train_per_family(config, speaker_brain, epoch)

        logging.info(f"[Epoch {epoch}] Extract embeddings")
        extract_all_families_embeddings(speaker_brain,  # fine-tuned model
                                        config['audio_dir'],
                                        config['train_audio_list_file'],
                                        config['embedding_output_file'],
                                        config['familes_emb'])
        
        logging.info(f"[Epoch {epoch}] Test full DB")
        test_entire_database(config, speaker_model, epoch)
        
        

    Save_Model_Localy(
        speaker_brain.modules.embedding_model,
        config, name="fine_tuned_model.pth"
    )
    if run: run.finish()


def prepare_family_labels(config, fam_id, fam_emb, n_speakers, labels_csv):
    """Cluster and save labels for a family"""
    labels, _ = spectral_clustering(fam_emb, n_speakers, family_id=fam_id)
    os.makedirs(os.path.dirname(labels_csv), exist_ok=True)
    append_to_csv(fam_id, labels, config['train_audio_list_file'], labels_csv)
    evaluate_and_log(config, fam_id, fam_emb, labels)
    return labels


def train_single_family(config, speaker_brain, fam_id, labels, optimizer, miner, triplet_loss_fn, soft_loss_fn, alpha, beta):
    """Train on a single family with Online Triplet Mining"""
    dataset = OnlineTripletDataset(get_audio_paths_for_family(config['train_audio_list_file'], fam_id), labels)
    if len(dataset) == 0:
        logging.warning(f"Empty dataset for family {fam_id}, skipping")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    speaker_brain.on_stage_start(sb.Stage.TRAIN)

    for _ in range(config['adaptation']['epochs_per_family']):
        for paths, lab_t in loader:
            emb = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths)
            if emb.size(0) == 0:
                continue

            lab_t = lab_t[:len(emb)].long().to(emb.device)
            emb = F.normalize(emb, p=2, dim=1)

            a, p, n = miner(emb, lab_t)
            t_loss = triplet_loss_fn(emb[a], emb[p], emb[n]) if a.numel() > 0 else torch.tensor(0.0, device=emb.device)

            aug_emb = extract_batch_embeddings_train(speaker_brain, config['audio_dir'], paths, 0.05, True)
            aug_emb = F.normalize(aug_emb, p=2, dim=1)
            assert len(emb) == len(aug_emb), "Mismatch in augmented batch size"

            ssreg_loss = torch.nn.functional.mse_loss(emb, aug_emb)
            s_loss = soft_loss_fn(emb, lab_t)

            loss = t_loss + alpha * s_loss + beta * ssreg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            speaker_brain.hparams.scheduler.step()
            torch.nn.utils.clip_grad_norm_(speaker_brain.modules.embedding_model.parameters(), max_norm=1.0)

            wandb.log({"family_loss": loss.item()})


def test_single_family(config, speaker_model, row, labels_df=None):
    """Test a single family by clustering embeddings and testing against the cluster labels"""
    family_id = row['family_number']
    family_emb = np.load(row['embedding_path'])

    n_speakers = get_unique_speakers_in_family(config['train_audio_list_file'], family_id)
    test_group = get_audio_paths_for_family(config['train_audio_list_file'], family_id)

    if len(test_group) == 0:
        logging.warning(f"No test samples for family {family_id}, skipping.")
        return

    # Convert test_group list to DataFrame with a 'path' column if it's a list of strings
    if isinstance(test_group, list):
        test_group = pd.DataFrame(test_group, columns=["path"])

    labels, _ = spectral_clustering(family_emb, n_speakers, family_id=family_id)

    family_test(
        speaker_model=speaker_model,
        config=config,
        family_id=family_id,
        test_group=test_group,
        family_emb=family_emb,
        labels=labels,
        num_speakers=n_speakers,
        to_plot=False
    )


def train_per_family(config, speaker_brain, epoch):
    """Orchestrates training and testing for each family"""
    df = pd.read_csv(config['familes_emb'], sep="\t")
    labels_csv = f"{config['familes_labels']}{epoch}.csv"
    clean_csv(labels_csv)

    # Pull hparams
    triplet_loss_fn = speaker_brain.hparams.triplet_loss
    soft_loss_fn    = speaker_brain.hparams.soft_loss
    optimizer       = speaker_brain.hparams.optimizer
    miner           = miners.TripletMarginMiner(margin=0.3, type_of_triplets="semihard")
    alpha           = speaker_brain.hparams.soft_weight
    beta            = speaker_brain.hparams.ssreg_weight

    
    # -------- Training loop --------
    train_df = df.head(config['adaptation']['train_nums_of_families'])
    for _, row in train_df.iterrows():
        fam_id = row['family_number']
        fam_emb = np.load(row['embedding_path'])
        n_speakers = get_unique_speakers_in_family(config['train_audio_list_file'], fam_id)

        labels = prepare_family_labels(config, fam_id, fam_emb, n_speakers, labels_csv)
        train_single_family(config, speaker_brain, fam_id, labels, optimizer, miner, triplet_loss_fn, soft_loss_fn, alpha, beta)
    
    # -------- Testing loop --------
    test_df = df.iloc[config['adaptation']['train_nums_of_families']:]

    for _, row in test_df.iterrows():
        test_single_family(config, speaker_brain, row)
        

def test_entire_database(config, speaker_model, epoch):
    speaker_model.eval()
    families_df = pd.read_csv(config['familes_emb'], delimiter="\t")
    labels_csv_path = f"{config['familes_labels']}_entire_database_{epoch}.csv"
    clean_csv(labels_csv_path)

    all_embeddings = []
    for _, row in families_df.iterrows():
        emb = np.load(row['embedding_path'])
        if np.isnan(emb).any():
            logging.warning(f"NaNs found in embeddings from {row['embedding_path']}")
            continue
        all_embeddings.append(emb)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    num_speakers = count_unique_speakers(config['train_audio_list_file'])

    process_entire_dataset(config, labels_csv_path, speaker_model, all_embeddings, num_speakers, to_plot=True)


def process_entire_dataset(config, labels_csv_path, speaker_model, all_embeddings, num_speakers, to_plot):

    labels, _ = spectral_clustering(all_embeddings, num_speakers, "entire_dataset")
    os.makedirs(os.path.dirname(config['familes_labels']), exist_ok=True)
    append_All_to_csv(labels, config['train_audio_list_file'], labels_csv_path)

    evaluate_and_log(config, None, all_embeddings, labels)

    if to_plot:
        plot_embedding(config, all_embeddings, labels)


def family_test(speaker_model, config, family_id, test_group, family_emb, labels, num_speakers, to_plot):
    speaker_model.set_eval_mode()

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


def lr_lambda(current_step: int):
    if current_step > 1000: #number of steps until cooldown
        return float(float(1000) / current_step)
    return 1.0


if __name__ == '__main__':
    main()