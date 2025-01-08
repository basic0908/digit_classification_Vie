import pandas as pd
import numpy as np
import os
import glob

def process_all_eeg_files(directory_path):
    """
    Process all Excel files in the specified directory to extract and stack LEFT and RIGHT EEG signals
    along axis 0 for each dataset type, and organize by filenames.

    Parameters:
        directory_path (str): Path to the directory containing Excel files.

    Returns:
        dict: A dictionary with filenames (based on `filename.split('-')[0]`) as keys and corresponding
              dataset dictionaries. Each dataset dictionary has dataset types ('Colors', 'Numbers',
              'Alphabet', 'Objects') as keys and EEG data arrays of shape (20, 6000) as values.
    """
    all_datasets = {}
    
    # Get all Excel files in the directory
    file_paths = glob.glob(os.path.join(directory_path, "*.csv"))
    
    for file_path in file_paths:
        # Extract filename identifier
        filename = os.path.basename(file_path)
        file_key = filename.split('-')[0]

        # Process the current file
        eeg_data = pd.read_csv(file_path)
        eeg_data['IMG_filename'].fillna(method='ffill', inplace=True)

        eeg_data['Dataset'] = eeg_data['IMG_filename'].apply(
            lambda x: 'Colors' if 'Colors' in x else 
                      'Numbers' if 'MNIST' in x else 
                      'Alphabet' if 'Chars74K' in x else 
                      'Objects' if 'ImageNet' in x else 'Unknown'
        )

        datasets = {}
        for dataset in ['Colors', 'Numbers', 'Alphabet', 'Objects']:
            dataset_eeg_left = eeg_data[eeg_data['Dataset'] == dataset].groupby('IMG_filename')['LEFT'].apply(list)
            dataset_eeg_right = eeg_data[eeg_data['Dataset'] == dataset].groupby('IMG_filename')['RIGHT'].apply(list)
            
            # Stack LEFT and RIGHT EEG data along axis 0
            stacked_data = []
            for left, right in zip(dataset_eeg_left[:10], dataset_eeg_right[:10]):
                stacked_data.append(left[:6000])
                stacked_data.append(right[:6000])
            
            datasets[dataset] = np.array(stacked_data)
        
        # Store the processed datasets under the file key
        all_datasets[file_key] = datasets

    return all_datasets
