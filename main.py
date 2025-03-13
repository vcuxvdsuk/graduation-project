from model_funcs import *
from clustering import *
from meta_data_preproccesing import *
from SV import *
from ploting_funcs import *
from statistics import mode

def main(config_file='/app/config.yaml'):
    
   # Load configuration settings
    config = load_config(config_file)
    
    #preproccess_pipline(config)

    # Load model
    speaker_model = load_model_ecapa(config)

    # Read the CSV file containing paths to all families
    families_df = pd.read_csv(config['familes_emb'], delimiter="\t")

#######################################################################################################
    # Loop through each family
    for index, row in families_df.head(10).iterrows():
        family_id = row['family_number']  
        family_path = row['embedding_path']

        #   1) load a single family embeddings as numpy array
        print(f"Processing family {family_id}")
        family_emb = np.load(family_path)

######################################################################################################
        #   2) cluster the embeddings
        num_of_speakers = get_unique_speakers_in_family(config['train_audio_list_file'], family_id)
        #simple_cluster()
        #gmm_cluster()
        #dec_cluster()
        labels, centroids = spectral_clustering(family_emb,num_of_speakers,family_id)


#####################################################################################################
        #   3) evaluate the clustering
#####################################################################################################
        #   4) extract embeddings from the all audio test files of corresponding family
        # Read the CSV file containing paths to all families test audio files
        test_families_df = pd.read_csv(config['test_audio_list_file'], delimiter="\t")
        family_test_group = test_families_df[test_families_df["family_id"] == family_id]
        test_sample_num = 0
        for _, row in family_test_group.iterrows():
            print(f"--------test1---------")
            #audio_path = os.path.join(config['audio_dir'], row['client_id'], row['path'])
            audio_path = os.path.join("/app/Arabic/combined_voice_dir", row['path'])
            test_emb = extract_single_embedding(speaker_model, audio_path)
            if test_emb is not None:
                test_sample_num += 1
                print(f"--------test2---------")
        #####################################################################################################
        #   5) check extracted embeddings simmelirity to the members of the family embeddings
                cenroid_label = identify_cluster(family_emb, labels, test_emb, method="centroid")
                KNN_label = identify_cluster(family_emb, labels, test_emb, method="knn", k=num_of_speakers)
                cosine_label = identify_cluster(family_emb, labels, test_emb, method="cosine")
                
                # Choose the most common label
                common_label = mode([cenroid_label, KNN_label, cosine_label])
#####################################################################################################
        #   6) varification eveluation
#####################################################################################################
        #   7) plot the clustering results
                plot_embeddings_with_new_sample(cenroid_label, KNN_label, cosine_label, family_emb, family_id, test_emb, test_sample_num, labels)
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