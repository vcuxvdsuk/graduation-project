from model_funcs import *
from clustering import *
from meta_data_preproccesing import *

def main(config_file='Users/shakedb/final_project/source code/config.yaml'):
    
    #preproccess_pipline()

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