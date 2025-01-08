import os
import pandas as pd
import numpy as np



def data_organizer():
    cwd = os.getcwd()
    print(f'cwd : {cwd}')
    folder_path = os.path.join(cwd, 'data\\viewRec')
    output_folder = os.path.join(cwd, 'data\\Vie2Image')

    print(f'folder_path : {os.path.exists(folder_path)}')
    print(f'output_folder exists : {os.path.exists(output_folder)}')
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            
            df = pd.read_csv(file_path, header=0)
            
            subject_name = file_name.split('-')[0]
            
            for i in range(0, len(df), 6000):
                chunk = df.iloc[i:i+6000]

                img_filename = chunk.iloc[0,3]

                img_dataset = img_filename.split('\\')[0]
                img_category = img_filename.split('\\')[1]

                if img_dataset == 'Colors':
                    img_dataset = 'Color'
                elif img_dataset == 'MNIST':
                    img_dataset = 'Digit'
                elif img_dataset == 'Chars74K':
                    img_dataset = 'Char'
                else:
                    img_dataset = 'Image'
                
                output_file = os.path.join(output_folder, img_dataset, f"{subject_name}_{img_category}.csv")
                chunk.iloc[:, :3].to_csv(output_file, index=False)
                print(f"Successfully saved at {output_file}")

def load_alphabet(folder_path):
    # Initialize X and Y
    X = np.zeros((220, 2, 6000))  # Shape (220, 2, 6000)
    Y = np.zeros(220, dtype=int)  # Shape (220,)
    
    # Loop through all CSV files in the folder
    for ctr, file in enumerate(sorted(os.listdir(folder_path))):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, header=0)
            
            # Extract first two columns and reshape for X
            X[ctr] = df.iloc[:, :2].values.T[:, :6000]
            
            # Determine the class for Y based on the file name
            cls = file.split('_')[1]  # Assuming file names follow a specific format
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

    return X, Y

def load_object(folder_path):
    # Initialize X and Y
    X = np.zeros((220, 2, 6000))  # Shape (220, 2, 6000)
    Y = np.zeros(220, dtype=int)  # Shape (220,)
    
    # Loop through all CSV files in the folder
    for ctr, file in enumerate(sorted(os.listdir(folder_path))):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, header=0)  # Skip the header
            
            # Extract first two columns and reshape for X
            X[ctr] = df.iloc[:, :2].values.T[:, :6000]
            
            # Determine the class for Y based on the file name
            _, cls = file.split('_')
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
            else:
                raise ValueError(f"Invalid object class in file name: {file}")
    
    return X, Y

def load_digit(folder_path):
    # Initialize X and Y
    X = np.zeros((220, 2, 6000))  # Shape (220, 2, 6000)
    Y = np.zeros(220, dtype=int)  # Shape (220,)
    
    # Loop through all CSV files in the folder
    for ctr, file in enumerate(sorted(os.listdir(folder_path))):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, header=0)  # Skip the header
            
            # Extract first two columns and reshape for X
            X[ctr] = df.iloc[:, :2].values.T[:, :6000]
            
            # Determine the class for Y based on the file name
            cls = file.split('_')[1]  # Assuming folder name contains the class
            try:
                Y[ctr] = int(cls)  # Parse numeric class directly
            except ValueError:
                raise ValueError(f"Invalid digit class in file name: {file}")
    
    return X, Y
