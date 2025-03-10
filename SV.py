import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
import torch
import numpy as np
from scipy.spatial.distance import cdist

def identify_cluster(embeddings, labels, new_sample, method="centroid", k=5):
    """
    Identify the best matching cluster for a new sample using different methods.

    :param embeddings: np.array of shape (n_samples, n_features), existing embeddings.
    :param labels: np.array of shape (n_samples,), cluster labels corresponding to embeddings.
    :param new_sample: np.array of shape (n_features,), the new sample embedding.
    :param method: str, the method to use ("centroid", "knn", "cosine", "density").
    :param k: int, number of neighbors for k-NN method.
    :return: int, best matching cluster label.
    """
    
    unique_clusters = np.unique(labels)
    
    if method == "centroid":
        # Compute centroids for each cluster
        centroids = {c: embeddings[labels == c].mean(axis=0) for c in unique_clusters}
        centroids_array = np.array(list(centroids.values()))
        cluster_ids = list(centroids.keys())

        # Find the closest centroid
        distances = cdist([new_sample], centroids_array, metric="euclidean")
        return cluster_ids[np.argmin(distances)]
    
    elif method == "knn":
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(embeddings, labels)
        return knn.predict([new_sample])[0]

    elif method == "cosine":
        # Compute centroids and find the most similar cluster using cosine similarity
        centroids = {c: embeddings[labels == c].mean(axis=0) for c in unique_clusters}
        centroids_array = np.array(list(centroids.values()))
        cluster_ids = list(centroids.keys())

        similarities = 1 - cdist([new_sample], centroids_array, metric="cosine")
        return cluster_ids[np.argmax(similarities)]
    
    else:
        raise ValueError("Invalid method. Choose from 'centroid', 'knn', 'cosine', or 'density'.")


def compute_prototypes(embeddings, labels):
    """
    Compute cluster prototypes as the mean embedding of few-shot samples.

    :param embeddings: np.array of shape (n_samples, n_features)
    :param labels: np.array of shape (n_samples,)
    :return: Dictionary {cluster_id: prototype_vector}
    """
    unique_clusters = np.unique(labels)
    prototypes = {c: embeddings[labels == c].mean(axis=0) for c in unique_clusters}
    return prototypes

def identify_cluster_protonet(embeddings, labels, new_sample):
    """
    Identify the best matching cluster for a new sample using Prototypical Networks.

    :param embeddings: np.array of shape (n_samples, n_features), few-shot support set.
    :param labels: np.array of shape (n_samples,), cluster labels.
    :param new_sample: np.array of shape (n_features,), new sample embedding.
    :return: int, best matching cluster label.
    """
    prototypes = compute_prototypes(embeddings, labels)
    cluster_ids = list(prototypes.keys())
    prototype_vectors = np.array(list(prototypes.values()))

    # Compute Euclidean distance to prototypes
    distances = cdist([new_sample], prototype_vectors, metric="euclidean")
    
    # Assign to nearest cluster prototype
    return cluster_ids[np.argmin(distances)]