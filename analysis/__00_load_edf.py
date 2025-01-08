import numpy as np
import os
import mne

import os
import numpy as np
import mne

def load_alphabet(folder_path):
    """
    Load digit classification data from EEG files in the specified folder.

    Parameters:
    folder_path (str): Path to the folder containing the EEG .edf files.

    Returns:
    tuple: A tuple containing:
        - X (numpy.ndarray): The feature matrix of shape (230, 14, 1280).
        - Y (numpy.ndarray): The labels array of shape (230,).
        - arr (list): List of channel names corresponding to the 14 channels.
    """
    X = np.zeros((230, 14, 1280))
    Y = np.zeros((230,))
    ctr = 0
    arr = []

    for fi in os.listdir(folder_path):
        data = mne.io.read_raw_edf(os.path.join(folder_path, fi), preload=False)
        
        # Extract channel names for the 14 channels
        if not arr:  # Populate only once
            arr = data.ch_names[2:16]
            print("Relevant Channels:", arr)
        
        raw_data = data[2:16][0] * 1000  # Select channels and scale
        raw_data = raw_data[:, 0:1280]  # Truncate to 1280 samples
        
        _, cls = fi.split('_')
        if cls[0] == 'A':
            Y[ctr] = 0
        elif cls[0] == 'C':
            Y[ctr] = 1
        elif cls[0] == 'F':
            Y[ctr] = 2
        elif cls[0] == 'H':
            Y[ctr] = 3
        elif cls[0] == 'J':
            Y[ctr] = 4
        elif cls[0] == 'M':
            Y[ctr] = 5
        elif cls[0] == 'P':
            Y[ctr] = 6
        elif cls[0] == 'S':
            Y[ctr] = 7
        elif cls[0] == 'T':
            Y[ctr] = 8
        elif cls[0] == 'Y':
            Y[ctr] = 9
        
        X[ctr, :, :] = raw_data
        ctr += 1

    return X, Y, arr


def load_digits(folder_path):
    """
    Load digit classification data from EEG files in the specified folder.

    Parameters:
    folder_path (str): Path to the folder containing the EEG .edf files.

    Returns:
    tuple: A tuple containing:
        - X (numpy.ndarray): The feature matrix of shape (230, 14, 1280).
        - Y (numpy.ndarray): The labels array of shape (230,).
    """
    X = np.zeros((230, 14, 1280))
    Y = np.zeros((230,))
    ctr = 0
    
    for fi in os.listdir(folder_path):
        data = mne.io.read_raw_edf(os.path.join(folder_path, fi))
        raw_data = data[2:16][0] * 1000
        raw_data = raw_data[:, 0:1280]
        
        _, cls = fi.split('_')
        Y[ctr] = int(cls[0])
        X[ctr, :, :] = raw_data
        ctr += 1

    return X, Y

def load_objects(folder_path):
    """
    Load object classification data from EEG files in the specified folder.

    Parameters:
    folder_path (str): Path to the folder containing the EEG .edf files.

    Returns:
    tuple: A tuple containing:
        - X (numpy.ndarray): The feature matrix of shape (230, 14, 1280).
        - Y (numpy.ndarray): The labels array of shape (230,).
    """
    X = np.zeros((230, 14, 1280))
    Y = np.zeros((230,))
    ctr = 0
    
    for fi in os.listdir(folder_path):
        data = mne.io.read_raw_edf(os.path.join(folder_path, fi))
        raw_data = data[2:16][0] * 1000
        raw_data = raw_data[:, 0:1280]
        
        _, cls = fi.split('_')
        cls = cls.strip()
        if cls == 'Apple.edf':
            Y[ctr] = 0
        elif cls == 'Car.edf':
            Y[ctr] = 1
        elif cls == 'Dog.edf':
            Y[ctr] = 2
        elif cls == 'Gold.edf':
            Y[ctr] = 3
        elif cls == 'Mobile.edf':
            Y[ctr] = 4
        elif cls == 'Rose.edf':
            Y[ctr] = 5
        elif cls == 'Scooter.edf':
            Y[ctr] = 6
        elif cls == 'Tiger.edf':
            Y[ctr] = 7
        elif cls == 'Wallet.edf':
            Y[ctr] = 8
        elif cls == 'Watch.edf':
            Y[ctr] = 9

        X[ctr, :, :] = raw_data
        ctr += 1

    return X, Y
