import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from tqdm import tqdm


def plot_eeg(X, instance=0):
    """
    Plot the first instance of EEG data with two channels, flexible for any time length.

    Parameters:
    X (numpy.ndarray): EEG data of shape (n_instances, n_timepoints, 2).

    Returns:
    None: Displays a plot of the first instance with time points on the x-axis.
    """
    if X.ndim != 3 or X.shape[2] != 2:
        print("Input shape must be (n_instances, n_timepoints, 2).")
        return
    
    first_instance = X[0]  # Shape (n_timepoints, 2)
    time_axis = np.arange(first_instance.shape[0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, first_instance[:, 0], label='Channel 1', marker='o')
    plt.plot(time_axis, first_instance[:, 1], label='Channel 2', marker='o')
    
    plt.xlabel('Timepoints')
    plt.ylabel('Amplitude')
    plt.title('EEG Data Visualization - First Instance')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()



def plot_eeg_1280(X, sample_index=0, channels=(0, 1), lim=600):
    """
    Plot EEG data along the 1280 timepoints for two selected channels of a specific sample.
    
    Parameters:
    X (numpy.ndarray): EEG data of shape (230, 14, 1280).
    sample_index (int): Index of the sample to visualize. Default is 0.
    channels (tuple): Tuple of two channel indices to plot. Default is (0, 1).

    Returns:
    None: Displays a plot for the two selected channels with x-axis limited to lim timepoints.
    """
    n_samples, n_channels, n_timepoints = X.shape
    if sample_index >= n_samples:
        print("Sample index out of range.")
        return
    if any(ch >= n_channels for ch in channels):
        print("Channel index out of range.")
        return
    
    time_axis = np.arange(n_timepoints)
    
    plt.figure(figsize=(12, 6))
    for channel in channels:
        plt.plot(time_axis[:lim], X[sample_index, channel, :lim], label=f'Channel {channel + 1}')
    
    plt.xlabel(f'Timepoints (0-{lim})')
    plt.ylabel('Amplitude')
    plt.title(f'EEG Data Visualization - Sample {sample_index}, Channels {channels[0] + 1} & {channels[1] + 1}')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def downsample(X, original_frequency=600, target_frequency=128):
    """
    Downsample EEG data to a target frequency using resampling, with tqdm for progress tracking.

    Parameters:
    X (numpy.ndarray): EEG data of shape (n_instances, n_channels, n_datapoints).
    original_frequency (int): Original sampling frequency (e.g., 600 Hz).
    target_frequency (int): Target sampling frequency (e.g., 128 Hz).

    Returns:
    numpy.ndarray: Downsampled EEG data with adjusted datapoints.
    """
    n_instances, n_channels, n_datapoints = X.shape
    # Calculate the target number of datapoints
    target_datapoints = int(n_datapoints * target_frequency / original_frequency)
    
    # Initialize an array for the downsampled data
    downsampled_data = np.zeros((n_instances, n_channels, target_datapoints))
    
    # Loop through each instance and channel to perform resampling
    for i in tqdm(range(n_instances), desc="Downsampling EEG Data"):
        for channel in range(n_channels):
            downsampled_data[i, channel, :] = resample(X[i, channel, :], target_datapoints)
    
    return downsampled_data