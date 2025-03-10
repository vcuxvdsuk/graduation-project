from model_funcs import *
from clustering import *
from meta_data_preproccesing import *

def main(config_file='Users/shakedb/final_project/source code/config.yaml'):
    
    #preproccess_pipline()
#######################################################################################################
#   1) load a single family embeddings as numpy array
    family_emb = np.load("C:/Users/USER/source/repos/vcuxvdsuk/graduation-project/Arabic/embedding/family_0_embedding.npy")
######################################################################################################
#   2) cluster the embeddings
    simple_cluster()
    gmm_cluster()
    dec_cluster()
    labels, centroids = spectral_clustering(148)

#####################################################################################################
#   3) evaluate the clustering
#####################################################################################################
#   4) extract embeddings from the all audio test files of corresponding family
#####################################################################################################
#   5) check extracted embeddings simmelirity to the members of the family embeddings
#####################################################################################################
#   6) varification eveluation
#####################################################################################################
#   7) plot the clustering results
#####################################################################################################
#   8) repeat the process for all the families
#####################################################################################################
#       9) fine-tune and repeat






        # Load configuration settings
    config = load_config(config_file)

        # Load model
    speaker_model = load_model_ASR(config)
    
        # embedding from the audio file
    extract_family_embeddings(speaker_model,
                            config['audio_dir'],
                            config['train_audio_list_file'],
                            config['embedding_output_file'],
                            config['familes_emb'])
    # simple_cluster()
    
    # gmm_cluster()
    # dec_cluster()
    # labels, centroids = spectral_clustering(148)

# Run the pipeline
if __name__ == '__main__':
    main()