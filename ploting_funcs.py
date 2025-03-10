import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

def plot_embedding(embedding_list, config):
    # Check if the input is a list of embeddings
    if isinstance(embedding_list, list):
        # Concatenate all embeddings in the list into a single 2D array
        embedding = np.concatenate(embedding_list, axis=0)
    else:
        raise ValueError("Input must be a list of embeddings")

    # Ensure embedding is 2D (samples x features)
    if len(embedding.shape) == 3:
        embedding = embedding.squeeze()  # Remove extra dimensions (squeeze)

    # Perform PCA only if there are more than one sample
    pca = PCA(n_components=config['pca']['n_components'])
    reduced_embedding = pca.fit_transform(embedding)
    
    # Plot the reduced embeddings (only works for 2D embeddings)
    plt.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1])
    
    # Compute cosine similarity between embeddings for coloring the area
    for i in range(reduced_embedding.shape[0] - 1):
        for j in range(i + 1, reduced_embedding.shape[0]):
            # Cosine similarity for the embeddings i and j
            sim = 1 - cosine(reduced_embedding[i], reduced_embedding[j])
            
            # Plot the area between embeddings based on the cosine similarity threshold
            if sim >= config['verification']['threshold']:
                # We plot a line (or shaded area) between the two embeddings if the similarity is high
                plt.plot([reduced_embedding[i, 0], reduced_embedding[j, 0]],
                         [reduced_embedding[i, 1], reduced_embedding[j, 1]], 
                         color='yellow', alpha=0.5)

    # Title and labels from the configuration
    plt.title(config['plotting']['title'])
    plt.xlabel(config['plotting']['xlabel'])
    plt.ylabel(config['plotting']['ylabel'])
    
    plt.show()
