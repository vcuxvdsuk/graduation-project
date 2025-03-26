import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cosine
import wandb


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
    plt.close()

def plot_embeddings_with_new_sample(cenroid_label, KNN_label, cosine_label, family_emb, family_num, test_emb,test_num, labels):
    """
    Print the labels and plot all embeddings, highlighting the test embedding.
    """
    print(f"Centroid Label: {cenroid_label}")
    print(f"KNN Label: {KNN_label}")
    print(f"Cosine Label: {cosine_label}")

    # Perform PCA to reduce the embeddings to 2D for visualization
    pca = PCA(n_components=2)
    reduced_family_emb = pca.fit_transform(family_emb)
    reduced_test_emb = pca.transform([test_emb])

    # Plot the family embeddings with cluster labels
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_family_emb[:, 0], reduced_family_emb[:, 1], c=labels, cmap='Spectral', alpha=0.5, label='Family Embeddings')
    
    # Highlight the test embedding
    plt.scatter(reduced_test_emb[:, 0], reduced_test_emb[:, 1], c='red', label='Test Embedding', edgecolors='black', s=100)
    
    plt.title('Embeddings Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.colorbar(scatter, label='Cluster Label')
    plot_path = f'test_emb_plot/new_sample_embeddings_plot_family{family_num}_sample{test_num}.png'
    plt.savefig(plot_path, dpi=300)
    # Log the plot to wandb
    wandb.log({"clustering_plot": wandb.Image(plot_path)})

    plt.show()
