import os
import pandas as pd
import shutil
from utils import *
# נתיבים
data_root = "/app/Arabic"
clips_dir = os.path.join(data_root, "clips")  # תיקיית קבצי האודיו

def preproccess_pipline(config):
    
    files = ["/app/Arabic/test.csv",
             "/app/Arabic/train.csv", 
             "/app/Arabic/dev.csv"]  
    combain_csv = "/app/Arabic/combined.csv"
    family_csv = "/app/Arabic/families.csv"
    family_test_csv = "/app/families_test.csv"

    combine_csv(files, combain_csv)

    add_family_id(combain_csv,family_csv)

    extract_first_record_per_speaker(family_csv,family_test_csv)

    source_dirs = ["/app/Arabic/dev",
                    "/app/Arabic/test",
                    "/app/Arabic/train"]  # List of directories to combine
    target_dir = "/app/Arabic/combined_voice_dir"  # Target directory to combine the contents into

    combine_directories(source_dirs,target_dir)


def extract_first_record_per_speaker(input_csv, output_csv):
    """
    Extract the first recording of each speaker from the input CSV and save it to a new CSV file.
    Args:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to the output CSV file where the first recording of each speaker will be saved.
    """
    # קריאת הקובץ המקורי
    df = pd.read_csv(input_csv, delimiter="\t")

    # חישוב את ההקלטה הראשונה של כל דובר
    first_recordings = df.drop_duplicates(subset='client_id', keep='first')
    
    # שמירת ההקלטות הראשונות בקובץ חדש
    first_recordings.to_csv(output_csv, index=False, sep="\t")

    mask = df[df.columns].isin(first_recordings[df.columns].to_dict(orient='list')).all(axis=1)
    
    # להחזיר את הדאטה-סט החדש עם השורות שהוסרו
    df_filtered = df[~mask]

    # שמירת הדאטאסט לאחר מחיקת ההקלטות הראשונות
    df_filtered.to_csv(input_csv, index=False, sep="\t")
    
    print(f"Extracted first recordings to {output_csv} and updated the original dataset.")


def add_family_id(input_csv, output_csv):
    """
    Add a family_id column to the dataset, ensuring up to 5 members per family.
    Args:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to the output CSV file with the family_id column added.
    """
    # Read the original dataset
    df = pd.read_csv(input_csv, delimiter='\t')    

    # Sort by client_id or any other criteria to group speakers logically (optional)
    df = df.sort_values(by='client_id')
    
    # Load configuration and train CSV path
    family_size = 5
    num_of_speakers = count_unique_speakers(df)
    remainder = num_of_speakers % family_size

    # If there's a remainder, remove the remaining speakers that don't fit into full families
    if remainder > 0:
        df = df[:-remainder]

    # Create a list of family IDs, with up to 5 members per family
    family_ids = []
    current_family_id = 0
    members_in_family = 0
    last_speaker = ""
    
    for index, row in df.iterrows():
        if(last_speaker != row[1]):
            last_speaker = row[1]
            members_in_family += 1

        if members_in_family < family_size:
            family_ids.append(current_family_id)
        else:
            current_family_id += 1
            family_ids.append(current_family_id)
            members_in_family = 1  # Reset for the new family
    
    # Add the family_id column to the DataFrame
    df['family_id'] = family_ids
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False, sep="\t")

import pandas as pd

def combine_csv(files, output_csv):
    """
    Combine multiple CSV files into a single CSV file (stacking by rows).
    Args:
    - files (list of str): List of paths to the CSV files.
    - output_csv (str): Path to the output combined CSV file.
    """
    # Read all the CSV files into a list of DataFrames
    dataframes = [pd.read_csv(file, encoding="utf-16", delimiter="\t") for file in files]
    
    # Concatenate all DataFrames vertically (stack by rows)
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_csv, index=False, sep="\t")


def combine_directories(source_dirs, target_dir):
    """
    Combine the contents of multiple directories into a single target directory.
    Args:
    - source_dirs (list of str): List of source directory paths.
    - target_dir (str): Path to the target directory where files will be combined.
    """
    # Make sure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            # List all files and directories in the source directory
            for speaker in os.listdir(source_dir):
                source_item = os.path.join(source_dir, speaker)

                # Check if it's a file
                if os.path.isfile(source_item):
                    # Generate the path to the target directory
                    target_item = os.path.join(target_dir, speaker)
                    
                    # Copy the file to the target directory
                    shutil.copy(source_item, target_item)
                # Optionally handle subdirectories if needed
                elif os.path.isdir(source_item):
                    # Recursively call combine_directories if you want to merge subdirectories too
                    combine_directories([source_item], target_dir)
        else:
            print(f"Source directory {source_dir} does not exist.")

