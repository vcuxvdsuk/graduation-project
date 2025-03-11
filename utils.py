
import os
import pandas as pd
import shutil

def count_unique_speakers(df):
    
    # Extract the number from the 'client_id' field (after the last underscore)
    df['speaker_number'] = df['client_id'].apply(lambda x: int(x.split('_')[-1]))
    
    # Count the number of unique speakers based on the extracted speaker numbers
    unique_speakers = df['speaker_number'].nunique()
    
    return unique_speakers

#gets the file and a family_id and return the num of uniqe client id of given family_id assuming the family_id are sorted
def get_unique_speakers_in_family(file_path, family_id):
    # Read the CSV file
    df = pd.read_csv(file_path, delimiter="\t")
    
    # Filter the DataFrame by the given family_id
    family_df = df[df['family_id'] == family_id]
    
    # Get the unique client_id values
    unique_clients = family_df['client_id'].nunique()
    
    return unique_clients

class Evaluations:
    # Method to calculate Equal Error Rate (EER)
    def calculate_eer(self, false_acceptance_rates, false_rejection_rates):
        eer = 0.0
        for far, frr in zip(false_acceptance_rates, false_rejection_rates):
            if far == frr:
                eer = far
                break
        return eer

    # Method to calculate False Acceptance Rate (FAR)
    def calculate_far(self, false_acceptances, total_impostor_attempts):
        if total_impostor_attempts == 0:
            return 0.0
        return false_acceptances / total_impostor_attempts

    # Method to calculate False Rejection Rate (FRR)
    def calculate_frr(self, false_rejections, total_genuine_attempts):
        if total_genuine_attempts == 0:
            return 0.0
        return false_rejections / total_genuine_attempts
