
import numpy as np

# Function to modify the embedding (e.g., adding noise)
def modify_embedding(embedding, config):
    noise = np.random.normal(config['modification']['noise_mean'], 
                             config['modification']['noise_std'], embedding.shape)
    modified_embedding = embedding + noise + 1
    return modified_embedding

def modify_embedding_list(embedding_list, config):
    modified_embedding_list = []
    for embedding in embedding_list:
        modified_embedding_list.append(modify_embedding(embedding, config))
    return modified_embedding_list


def modify_all_embedings(file_path ,config):
    embeddings = np.load(file_path)

    #amazing func

    
    # Convert the list of embeddings to a NumPy array
    mod_embedding = np.array(mod_embedding)

    # Save the embeddings as a .npy file
    np.save(config["embedding_output_file"], mod_embedding)
    print(f"Embeddings saved to {config['embedding_output_file']}")
