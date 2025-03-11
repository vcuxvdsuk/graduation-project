from model_funcs import *
from clustering import *
from meta_data_preproccesing import *

def main(config_file='/app/config.yaml'):
    
    #preproccess_pipline()
    
   # Load configuration settings
    config = load_config(config_file)

    # Load model
    speaker_model = load_model_ASR(config)

    # Read the CSV file containing paths to all families
    families_df = pd.read_csv(config['familes_emb'], delimiter="\t")

#######################################################################################################
    # Loop through each family
    for family_id, family_path in families_df.groupby("family_id"):
        #   1) load a single family embeddings as numpy array
        print(f"Processing family {family_id}")
        family_emb = np.load(family_path)

######################################################################################################
        #   2) cluster the embeddings
        num_of_speakers = get_unique_speakers_in_family(config['train_audio_list_file'], family_id)
        #simple_cluster()
        #gmm_cluster()
        #dec_cluster()
        labels, centroids = spectral_clustering(family_emb,num_of_speakers)


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

#####################################################################################################    
        # embedding from the audio file - dosnt need to run every time
    #extract_family_embeddings(speaker_model,
     #                       config['audio_dir'],
      #                      config['train_audio_list_file'],
       #                     config['embedding_output_file'],
        #                    config['familes_emb'])


# Run the pipeline
if __name__ == '__main__':
    main()