import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from model_funcs import *
from sklearn.mixture import GaussianMixture
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

def simple_cluster():


    config = load_config()

    # Step 1: Prepare your embeddings (assuming `embeddings` is a NumPy array)
    # Example: Let's assume `embeddings` is a 2D NumPy array where each row is an embedding vector.
    # `embeddings` should have shape (n_samples, n_features)
    # For the sake of this example, let's assume you already have embeddings loaded:

    file_path = "/media/ACLP-Nimble/Users/shakedb/final_project/speech_brain_valid_sound/Arabic/embeddig.npy"
    mod_embedding = np.load(file_path)

    train_csv_path = config['train_audio_list_file']
    n_clusters = count_unique_speakers(train_csv_path)
    #n_clusters = 10
    print(f"Number of unique speakers: {n_clusters}")
    print(mod_embedding.shape)  # This will print the shape of the embedding array

    mod_embedding = mod_embedding.squeeze(1).squeeze(1)

    # Step 2: Apply K-means clustering
    #n_clusters = 3  # Specify the number of clusters (change this value as needed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(mod_embedding)

    # Get the cluster labels
    labels = kmeans.labels_

    # Step 3: Calculate the Silhouette score
    sil_score = silhouette_score(mod_embedding, labels)
    print(f"Silhouette Score: {sil_score}")

    # Step 4: Visualize the clustering using PCA (Dimensionality reduction for 2D or 3D visualization)
    # PCA to reduce the embeddings to 2D for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(mod_embedding)

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral', marker='o')
    plt.title(f"K-means Clustering (Silhouette Score: {sil_score:.2f})")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.savefig('clustering_plot.png', dpi=300) 
    plt.show()


def gmm_cluster(n_clusters=5):
    """
    Perform Gaussian Mixture Model clustering on the embeddings.
    """
    
    # Example embeddings: replace this with your actual data
    file_path = "/media/ACLP-Nimble/Users/shakedb/final_project/speech_brain_valid_sound/Arabic/embeddig.npy"
    mod_embedding = np.load(file_path)
    mod_embedding = mod_embedding.squeeze(1).squeeze(1)

    # Apply Gaussian Mixture Model clustering
    config = load_config()
    train_csv_path = config['train_audio_list_file']
    n_clusters = count_unique_speakers(train_csv_path)
    #n_clusters = 10
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(mod_embedding)
    
    # Get the cluster labels
    labels = gmm.predict(mod_embedding)
    
    # Calculate the Silhouette score
    sil_score = silhouette_score(mod_embedding, labels)
    print(f"Silhouette Score: {sil_score}")
    
    # Visualize the clustering using PCA (for 2D visualization)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(mod_embedding)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral', marker='o')
    plt.title(f"GMM Clustering (Silhouette Score: {sil_score:.2f})")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.savefig('gmm_clustering_plot.png', dpi=300)
    plt.show()


def dec_cluster(n_clusters=5):
    """
    Perform Deep Embedded Clustering (DEC) on the embeddings.
    """
    
    # Example embeddings: replace this with your actual data
    file_path = "/media/ACLP-Nimble/Users/shakedb/final_project/speech_brain_valid_sound/Arabic/embeddig.npy"
    mod_embedding = np.load(file_path)
    mod_embedding = mod_embedding.squeeze(1).squeeze(1)

    config = load_config()
    train_csv_path = config['train_audio_list_file']
    n_clusters = count_unique_speakers(train_csv_path)
    #n_clusters = 10

    # Step 1: Initialize KMeans to get initial cluster centroids
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    initial_labels = kmeans.fit_predict(mod_embedding)
    
    # Step 2: Create a deep neural network model for DEC
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(mod_embedding.shape[1],)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_clusters, activation='softmax')
    ])
    
    # Step 3: Compile and train the DEC model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # One-hot encode the labels for DEC training
    labels_one_hot = tf.keras.utils.to_categorical(initial_labels, num_classes=n_clusters)
    model.fit(mod_embedding, labels_one_hot, epochs=20, batch_size=64)
    
    # Step 4: Use the DEC model to predict the cluster labels
    dec_labels = np.argmax(model.predict(mod_embedding), axis=1)
    
    # Calculate the Silhouette score
    sil_score = silhouette_score(mod_embedding, dec_labels)
    print(f"Silhouette Score: {sil_score}")
    
    # Visualize the clustering using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(mod_embedding)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=dec_labels, cmap='Spectral', marker='o')
    plt.title(f"DEC Clustering (Silhouette Score: {sil_score:.2f})")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.savefig('dec_clustering_plot.png', dpi=300)
    plt.show()


def spectral_clustering(n_clusters=5):
    """
    Perform Spectral Clustering on the embeddings and return cluster labels and centroids.
    """
    
    # Example embeddings: replace this with your actual data
    file_path = "/media/ACLP-Nimble/Users/shakedb/final_project/speech_brain_valid_sound/Arabic/embeddig.csv.npy"
    mod_embedding = np.load(file_path)
    mod_embedding = mod_embedding.squeeze(1).squeeze(1)

    # Load configuration and train CSV path
    config = load_config()
    train_csv_path = f"{config['train_audio_list_file']}.csv"
    n_clusters = count_unique_speakers(train_csv_path)
    #n_clusters = 10  # Uncomment this if you want to manually set n_clusters

    # Apply Spectral Clustering
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
    labels = spectral.fit_predict(mod_embedding)

    # Calculate the Silhouette score
    sil_score = silhouette_score(mod_embedding, labels)
    print(f"Silhouette Score: {sil_score}")

    # Calculate the centroids of each cluster (mean of all points in each cluster)
    centroids = np.array([mod_embedding[labels == i].mean(axis=0) for i in range(n_clusters)])

    # Visualize the clustering using PCA (for 2D visualization)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(mod_embedding)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral', marker='o')
    plt.title(f"Spectral Clustering (Silhouette Score: {sil_score:.2f})")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.savefig('spectral_clustering_plot.png', dpi=300)
    plt.show()

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

