import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
import torch
import numpy as np
from scipy.spatial.distance import cdist

# func for close set
def identify_cluster(embeddings, labels, new_sample, method="centroid", k=5):
    """
    Identify the best matching cluster for a new sample.

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

# func for open set
def identify_or_assign_cluster(embeddings, labels, new_sample, method="centroid", k=5, threshold=1.2):
    """
    Identify the best matching cluster for a new sample or assign it to a new cluster.

    :param embeddings: np.array of shape (n_samples, n_features), existing embeddings.
    :param labels: np.array of shape (n_samples,), cluster labels corresponding to embeddings.
    :param new_sample: np.array of shape (n_features,), the new sample embedding.
    :param method: str, the method to use ("centroid", "knn", "cosine", "density").
    :param k: int, number of neighbors for k-NN method.
    :param threshold: float, similarity threshold for assigning to a new cluster.
    :return: int, best matching cluster label or new cluster label.
    """
    
    unique_clusters = np.unique(labels)
    
    if method == "centroid":
        # Compute centroids for each cluster
        centroids = {c: embeddings[labels == c].mean(axis=0) for c in unique_clusters}
        centroids_array = np.array(list(centroids.values()))
        cluster_ids = list(centroids.keys())

        # Find the closest centroid
        distances = cdist([new_sample], centroids_array, metric="euclidean")
        closest_cluster = cluster_ids[np.argmin(distances)]
        closest_centroid = centroids[closest_cluster]

        # Compute the maximum distance from the centroid to any point in the cluster
        cluster_points = embeddings[labels == closest_cluster]
        max_distance = np.max(cdist([closest_centroid], cluster_points, metric="euclidean"))

        # Check if the new sample is within the threshold distance
        new_sample_distance = np.linalg.norm(new_sample - closest_centroid)
        if new_sample_distance <= threshold * max_distance:
            return closest_cluster
        else:
            return max(cluster_ids) + 1  # Assign to a new cluster
    
    elif method == "knn":
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(embeddings, labels)
        predicted_label = knn.predict([new_sample])[0]
        distances, _ = knn.kneighbors([new_sample])
        if np.mean(distances) < threshold:
            return predicted_label
        else:
            return max(unique_clusters) + 1  # Assign to a new cluster

    elif method == "cosine":
        # Compute centroids and find the most similar cluster using cosine similarity
        centroids = {c: embeddings[labels == c].mean(axis=0) for c in unique_clusters}
        centroids_array = np.array(list(centroids.values()))
        cluster_ids = list(centroids.keys())

        similarities = 1 - cdist([new_sample], centroids_array, metric="cosine")
        closest_cluster = cluster_ids[np.argmax(similarities)]
        closest_centroid = centroids[closest_cluster]

        # Compute the maximum similarity from the centroid to any point in the cluster
        cluster_points = embeddings[labels == closest_cluster]
        max_similarity = np.max(1 - cdist([closest_centroid], cluster_points, metric="cosine"))

        # Check if the new sample is within the threshold similarity
        new_sample_similarity = 1 - cdist([new_sample], [closest_centroid], metric="cosine")[0][0]
        if new_sample_similarity >= threshold * max_similarity:
            return closest_cluster
        else:
            return max(cluster_ids) + 1  # Assign to a new cluster
    
    else:
        raise ValueError("Invalid method. Choose from 'centroid', 'knn', 'cosine', or 'density'.")

