import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import torch
import pandas as pd
from model_funcs import *
from evaluation import *
from utils import *
from adaptation import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

def get_config():
    # Adjust path if needed
    return load_config("config.yaml")

def get_sample_df(config, n=20):
    # Use a small sample for quick tests
    if "all_samples" in config:
        df = pd.read_csv(config["all_samples"], delimiter="\t")
    else:
        df = pd.read_csv(config["train_audio_list_file"], delimiter="\t")
    return df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)

def test_voice_processing():
    config = get_config()
    df = get_sample_df(config, n=100)
    paths = df["path"].tolist()
    model = load_model_ecapa_from_speechbrain(config)

    speaker_brain = modelTune(
        modules=model.mods,
        hparams={
        },
        run_opts={"device": model.device}
    )

    start = time.time()
    emb = extract_batch_embeddings_train(speaker_brain, config["audio_dir"], paths)
    elapsed = time.time() - start
    emb = emb.cpu().numpy() if hasattr(emb, "cpu") else emb
    assert emb is not None and np.isfinite(emb).all() and emb.shape[0] == len(paths), "Embedding extraction failed or noisy"
    avg_time = elapsed / len(paths)
    print(f"test_voice_processing: PASS (Processed {len(paths)} files, avg {avg_time:.3f}s per file)")

def test_clustering():
    config = get_config()
    df = get_sample_df(config, n=20)
    paths = df["path"].tolist()
    labels = df["client_id"].astype("category").cat.codes.values
    model = load_model_ecapa_from_speechbrain(config)
    embeddings = extract_batch_embeddings_train(model, config["audio_dir"], paths).cpu().numpy()
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pred_labels = kmeans.fit_predict(embeddings)
    sil = silhouette_score(embeddings, pred_labels)
    acc = Evaluations.cluster_accuracy(labels, pred_labels)
    assert sil > 0.01, f"Silhouette too low: {sil}"
    print(f"test_clustering: PASS (Silhouette={sil:.4f}, CluAcc={acc:.4f})")

def test_processing_time():
    config = get_config()
    df = get_sample_df(config, n=1)
    audio_path = df.iloc[0]["path"]
    audio_full_path = f"{config['audio_dir']}/{audio_path}"
    model = load_model_ecapa_from_speechbrain(config)
    start = time.time()
    emb = extract_single_embedding(model, audio_full_path)
    elapsed = time.time() - start
    assert elapsed < 30, f"Processing took too long: {elapsed}s"
    print(f"test_processing_time: PASS (Elapsed={elapsed:.2f}s)")

def test_eer():
    config = get_config()
    df = get_sample_df(config, n=30)
    paths = df["path"].tolist()
    labels = df["client_id"].astype("category").cat.codes.values
    model = load_model_ecapa_from_speechbrain(config)
    embeddings = extract_batch_embeddings_train(model, config["audio_dir"], paths).cpu().numpy()
    y_true, y_scores = Evaluations.build_verification_pairs(embeddings, labels)
    eer = Evaluations.calculate_eer(y_true, y_scores)
    assert eer <= 0.2, f"EER too high: {eer}"
    print(f"test_eer: PASS (EER={eer:.4f})")

def test_identification_accuracy():
    config = get_config()
    df = get_sample_df(config, n=30)
    paths = df["path"].tolist()
    labels = df["client_id"].astype("category").cat.codes.values
    model = load_model_ecapa_from_speechbrain(config)
    embeddings = extract_batch_embeddings_train(model, config["audio_dir"], paths).cpu().numpy()
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, -np.inf)
    preds = np.argmax(sim, axis=1)
    pred_labels = labels[preds]
    correct = np.sum(pred_labels == labels)
    assert correct >= int(0.8 * len(labels)), f"Only {correct}/{len(labels)} correct identifications"
    print(f"test_identification_accuracy: PASS ({correct}/{len(labels)} correct)")

def test_clustering_time():
    config = get_config()
    df = get_sample_df(config, n=30)
    paths = df["path"].tolist()
    labels = df["client_id"].astype("category").cat.codes.values
    model = load_model_ecapa_from_speechbrain(config)
    embeddings = extract_batch_embeddings_train(model, config["audio_dir"], paths).cpu().numpy()
    n_clusters = len(np.unique(labels))
    start = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pred_labels = kmeans.fit_predict(embeddings)
    elapsed = time.time() - start
    assert elapsed < 30, f"Clustering took too long: {elapsed}s"
    print(f"test_clustering_time: PASS (Elapsed={elapsed:.2f}s)")

if __name__ == "__main__":
    test_voice_processing()
    test_clustering()
    test_processing_time()
    test_eer()
    test_identification_accuracy()
    test_clustering_time()