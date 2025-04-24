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


def gmm_cluster(embedding, n_clusters=5, family_id=-1):
    """
    Perform Gaussian Mixture Model clustering on the embeddings and return cluster labels and centroids.
    """
    if isinstance(embedding, list):
        embedding = np.array(embedding)

    # Apply Gaussian Mixture Model clustering
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(embedding)

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
    plt.title(f"GMM Clustering {family_id} (Silhouette Score: {sil_score:.2f})")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.savefig(f'plots/gmm_clustering_plot{family_id}.png', dpi=300)
    plt.show()
    plt.close()

    # Return labels and centroids
    return labels, centroids


def dec_cluster(embedding, n_clusters=5, family_id=-1):
    """
    Perform Deep Embedded Clustering (DEC) on the embeddings and return cluster labels and centroids.
    """
    if isinstance(embedding, list):
        embedding = np.array(embedding)

    # Step 1: Initialize KMeans to get initial cluster centroids
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    initial_labels = kmeans.fit_predict(embedding)
    
    # Step 2: Create a deep neural network model for DEC
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(embedding.shape[1],)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_clusters, activation='softmax')
    ])
    
    # Step 3: Compile and train the DEC model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # One-hot encode the labels for DEC training
    labels_one_hot = tf.keras.utils.to_categorical(initial_labels, num_classes=n_clusters)
    model.fit(embedding, labels_one_hot, epochs=20, batch_size=64)
    
    # Step 4: Use the DEC model to predict the cluster labels
    dec_labels = np.argmax(model.predict(embedding), axis=1)
    
    # Calculate the Silhouette score
    sil_score = silhouette_score(embedding, dec_labels)
    print(f"Silhouette Score: {sil_score}")

    # Calculate the centroids of each cluster (mean of all points in each cluster)
    centroids = np.array([embedding[dec_labels == i].mean(axis=0) for i in range(n_clusters)])

    # Visualize the clustering using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embedding)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=dec_labels, cmap='Spectral', marker='o')
    plt.title(f"DEC Clustering {family_id} (Silhouette Score: {sil_score:.2f})")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.savefig(f'plots/dec_clustering_plot{family_id}.png', dpi=300)
    plt.show()
    plt.close()

    # Return labels and centroids
    return dec_labels, centroids


def spectral_clustering(embedding, n_clusters=5, family_id=-1):
    """
    Perform Spectral Clustering on the embeddings and return cluster labels and centroids.
    """
    if isinstance(embedding, list):
        embedding = np.array(embedding)

    # Ensure n_neighbors is less than the number of samples
    n_neighbors = min(n_clusters, len(embedding) - 1)

    # Apply Spectral Clustering
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors', n_neighbors=n_neighbors)
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



