import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from model_funcs import *
from sklearn.mixture import GaussianMixture
import tensorflow as tf
from sklearn.cluster import SpectralClustering
import warnings
import hdbscan

def simple_cluster(embedding, n_clusters=5, family_id=-1):
    """
    Perform K-means clustering on the embeddings and return cluster labels and centroids.
    """
    if isinstance(embedding, list):
        embedding = np.array(embedding)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding)

    # Calculate the Silhouette score
    sil_score = silhouette_score(embedding, labels)
    print(f"Silhouette Score: {sil_score}")

    # Calculate the centroids of each cluster (mean of all points in each cluster)
    centroids = kmeans.cluster_centers_

    # Visualize the clustering using PCA (for 2D visualization)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embedding)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral', marker='o')
    plt.title(f"K-means Clustering {family_id} (Silhouette Score: {sil_score:.2f})")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.savefig(f'plots/kmeans_clustering_plot{family_id}.png', dpi=300)
    plt.show()
    plt.close()

    # Return labels and centroids
    return labels, centroids


def spectral_clustering(embedding, n_clusters=5, family_id=-1):
    """
    Perform Spectral Clustering on the embeddings and return cluster labels and centroids.
    """
    if isinstance(embedding, list):
        embedding = np.array(embedding)

    # Ensure n_neighbors is less than the number of samples
    n_neighbors = min(n_clusters, len(embedding) - 1,3) #avrg 4 samples per speaker

    # Apply Spectral Clustering
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors', n_neighbors=n_neighbors)
        #spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='rbf', gamma=0.1)
        labels = spectral.fit_predict(embedding)

    # Calculate the Silhouette score
    sil_score = silhouette_score(embedding, labels)
    print(f"Silhouette Score: {sil_score}")

    # Calculate the centroids of each cluster (mean of all points in each cluster)
    centroids = np.array([embedding[labels == i].mean(axis=0) for i in range(n_clusters)])

    # Visualize the clustering using PCA (for 2D visualization)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embedding)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral', marker='o')
    plt.title(f"Spectral Clustering {family_id} (Silhouette Score: {sil_score:.2f})")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    #plt.savefig(f'plots/spectral_clustering_plot{family_id}.png', dpi=300)
    plt.show()
    plt.close()

    # Return labels and centroids
    return labels, centroids


def count_unique_speakers(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, delimiter="\t")
    
    # Extract the number from the 'client_id' field (after the last underscore)
    df['speaker_number'] = df['client_id'].apply(lambda x: int(x.split('_')[-1]))
    
    # Count the number of unique speakers based on the extracted speaker numbers
    unique_speakers = df['speaker_number'].nunique()
    
    return unique_speakers



