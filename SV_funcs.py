
from scipy.spatial.distance import cosine

# Function to verify if two embeddings belong to the same speaker
def verify_speaker(embedding1, embedding2, config):
    similarity = 1 - cosine(embedding1, embedding2)
    if similarity >= config['verification']['threshold']:
        return True
    else:
        return False

def classify_new_example(new_example, centroids):
    """
    Classify a new example based on its proximity to cluster centroids.
    Args:
    - new_example (ndarray): The embedding of the new example.
    - centroids (ndarray): The centroids of the clusters.
    Returns:
    - label (int): The predicted cluster label.
    """
    # Compute the Euclidean distance from the new example to each centroid
    distances = np.linalg.norm(centroids - new_example, axis=1)
    
    # Return the label of the closest centroid
    return np.argmin(distances)