import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cosine
import wandb


def plot_embedding(config, embedding_list, labels):
    
    embedding = check_embedding(embedding_list)

    # Perform PCA only if there are more than one sample
    pca = PCA(n_components=config['pca']['n_components'])
    reduced_embedding = pca.fit_transform(embedding)
    
    # Plot the reduced embeddings (only works for 2D embeddings)
    scatter = plt.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], c=labels, cmap='Spectral', alpha=0.5, label='Family Embeddings')
    
    # Title and labels from the configuration
    plt.title(config['plotting']['title'])
    plt.xlabel(config['plotting']['xlabel'])
    plt.ylabel(config['plotting']['ylabel'])
        
    plot_path = f"{config['familes_plot_path']}_all_data.png"
    plt.title(f'Embeddings Visualization - {plot_path}')
    plt.colorbar(scatter, label='Cluster Label')
    plt.savefig(plot_path, dpi=300)
    # Log the plot to wandb
    wandb.log({"clustering_plot": wandb.Image(plot_path)})

    #plt.show()
    plt.close()


def plot_embeddings_with_new_sample(config, family_emb, family_num, test_emb, test_num, labels):
    family_emb = check_embedding(family_emb)
    assert len(family_emb) == len(labels), "Family embeddings and labels must have the same length"
    # Perform PCA to reduce the embeddings to 2D for visualization
    pca = PCA(n_components=2)
    reduced_family_emb = pca.fit_transform(family_emb)
    reduced_test_emb = pca.transform([test_emb])

    # Plot the family embeddings with cluster labels
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_family_emb[:, 0], reduced_family_emb[:, 1], c=labels, cmap='Spectral', alpha=0.5, label='Family Embeddings')
    
    # Highlight the test embedding
    plt.scatter(reduced_test_emb[:, 0], reduced_test_emb[:, 1], c='red', label='Test Embedding', edgecolors='black', s=100)
    
    plot_path = f"{config['familes_plot_path']}{family_num}_sample_{test_num}.png"

    plt.title(f'Embeddings Visualization - {plot_path}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.colorbar(scatter, label='Cluster Label')
    plt.savefig(plot_path, dpi=300)
    # Log the plot to wandb
    wandb.log({"clustering_plot": wandb.Image(plot_path)})

    #plt.show()
    plt.close()



def check_embedding(embedding_list):
    # Check if the input is a list of embeddings
    if isinstance(embedding_list, list):
        # Concatenate all embeddings in the list into a single 2D array
        embedding = np.concatenate(embedding_list, axis=0)
    elif isinstance(embedding_list, np.ndarray):
        embedding = embedding_list
    else:
        raise ValueError(f"Input must be a list of embeddings or a NumPy array, input is {type(embedding_list)}")

    # Ensure embedding is 2D (samples x features)
    if len(embedding.shape) == 3:
        embedding = embedding.squeeze()  # Remove extra dimensions (squeeze)

    return embedding
