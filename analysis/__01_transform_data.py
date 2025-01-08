import numpy as np
import mne
import contextlib
import sys
import os
from scipy.signal import butter, filtfilt, iirnotch
import pywt



def transform_eeg_data(X, Y, selected_channels=None, npt=32, stride=8):
    """
    Transform EEG data into overlapping segments using a sliding window.

    Parameters:
    X (numpy.ndarray): EEG data of shape (n_samples, 14, n_timepoints).
    Y (numpy.ndarray): Labels corresponding to each sample of shape (n_samples,).
    selected_channels (list or None): Indices of the channels to use. 
                                       If None, use all 14 channels.
    npt (int): Number of time points in each segment (window size).
    stride (int): Step size for the sliding window.

    Returns:
    tuple: 
        - X_new (numpy.ndarray): Transformed feature matrix of shape 
                                 (n_windows, npt, n_selected_channels).
        - Y_new (numpy.ndarray): Labels corresponding to each window.
    """
    if selected_channels is None:
        selected_channels = list(range(X.shape[1]))  # Use all channels if None

    n_samples, n_channels, n_timepoints = X.shape
    n_selected_channels = len(selected_channels)
    
    # Calculate the number of windows
    n_windows_per_sample = (n_timepoints - npt) // stride + 1
    n_windows = n_samples * n_windows_per_sample
    
    # Initialize new arrays
    X_new = np.zeros((n_windows, npt, n_selected_channels))
    Y_new = np.zeros((n_windows,))
    
    ctr = 0
    for i in range(n_samples):
        y = Y[i]
        a = X[i, selected_channels, :].transpose()  # Select only specified channels
        
        val = 0
        while val <= (len(a) - npt):
            x = a[val:val + npt, :]
            X_new[ctr, :, :] = x
            Y_new[ctr] = y
            val += stride
            ctr += 1

    return X_new, Y_new

def MA_X(X, axis, M=3):
    """
    Apply a moving average filter with a specified window size M along the specified axis of the data,
    ensuring the output has the same shape as the input by padding appropriately.

    Parameters:
    X (numpy.ndarray): Input data, e.g., shape (36110, 32, 2).
    axis (int): Axis along which to apply the moving average filter.
    M (int): Window size for the moving average filter.

    Returns:
    numpy.ndarray: Data after applying the moving average filter,
                   with the same shape as the input.
    """
    pad_width = [(0, 0)] * X.ndim  # Initialize padding for all dimensions
    pad_width[axis] = (M // 2, M // 2)  # Apply padding along the specified axis

    # Pad the array to handle edge effects
    X_padded = np.pad(X, pad_width, mode='edge')

    # Apply the moving average filter
    filtered_X = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(M)/M, mode='valid'), axis=axis, arr=X_padded
    )

    return filtered_X

def MA_X2(X, axis, M=3):
    """
    Apply a moving average filter with a specified window size M along the specified axis of the data,
    using 'valid' mode without padding. The input shape should be adjusted so that the output shape matches the desired size.

    Parameters:
    X (numpy.ndarray): Input data, e.g., shape (36110, 34, 2).
    axis (int): Axis along which to apply the moving average filter.
    M (int): Window size for the moving average filter.

    Returns:
    numpy.ndarray: Data after applying the moving average filter, 
                   with reduced shape based on 'valid' mode.
    """
    # Apply the moving average filter along the specified axis
    filtered_X = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(M)/M, mode='valid'), axis=axis, arr=X
    )
    
    return filtered_X


def process_band(X, Y, l_freq=None, h_freq=4, selected_channels=None, npt=32, stride=8):
    """
    Preprocess EEG data for a specific frequency band and create overlapping segments.

    Parameters:
    X (numpy.ndarray): Raw EEG data of shape (n_samples, 14, n_timepoints).
    Y (numpy.ndarray): Labels for each EEG sample.
    l_freq (float or None): Low cutoff frequency for the filter.
    h_freq (float or None): High cutoff frequency for the filter.
    selected_channels (list or None): Indices of the channels to use. 
                                       If None, use all 14 channels.
    npt (int): Number of time points in each segment.
    stride (int): Step size for the sliding window.

    Returns:
    tuple:
        - X_new (numpy.ndarray): Transformed feature matrix of shape 
                                 (n_windows, npt, n_selected_channels).
        - Y_new (numpy.ndarray): Corresponding labels for each segment.
    """
    if selected_channels is None:
        selected_channels = list(range(X.shape[1]))  # Use all channels if None

    n_samples, n_channels, n_timepoints = X.shape
    n_selected_channels = len(selected_channels)
    X_band = np.zeros_like(X)

    # Suppress printing for the filtering process
    with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
        for i in range(n_samples):
            for j in selected_channels:
                sig = X[i, j, :]
                output_signal_mne = mne.filter.filter_data(sig, sfreq=128, l_freq=l_freq, h_freq=h_freq)
                X_band[i, j, :] = output_signal_mne

    # Create overlapping windows
    n_windows_per_sample = (n_timepoints - npt) // stride + 1
    n_windows = n_samples * n_windows_per_sample

    X_new = np.zeros((n_windows, npt, n_selected_channels))
    Y_new = np.zeros((n_windows,))
    ctr = 0

    for i in range(n_samples):
        y = Y[i]
        a = X_band[i, selected_channels, :].transpose()  # Select specified channels and transpose
        val = 0
        while val <= (len(a) - npt):
            x = a[val:val + npt, :]
            X_new[ctr, :, :] = x
            Y_new[ctr] = y
            val += stride
            ctr += 1

    return X_new, Y_new

def butterworth_highpass_filter(X, cutoff=0.1, fs=128, order=2, axis=1):
    """
    Apply a Butterworth high-pass filter to eliminate low-frequency noise,
    ensuring the output retains the same shape as the input.

    Parameters:
    X (numpy.ndarray): Input data of shape (..., n_timepoints, ...).
    cutoff (float): Cutoff frequency of the high-pass filter in Hz (default is 0.1 Hz).
    fs (int): Sampling frequency in Hz (default is 128 Hz).
    order (int): Order of the Butterworth filter (default is 2).
    axis (int): Axis along which to apply the filter.

    Returns:
    numpy.ndarray: High-pass filtered data of the same shape as the input.
    """
    # Design a Butterworth high-pass filter
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    
    # Apply the filter along the specified axis
    def apply_filter(data_slice):
        return filtfilt(b, a, data_slice, padlen=0)
    
    filtered_X = np.apply_along_axis(apply_filter, axis=axis, arr=X)
    
    return filtered_X


def notch_filter(X, freq=60.0, fs=128, Q=30.0, axis=1):
    """
    Apply a notch filter to remove a specific frequency (e.g., 60 Hz) along the specified axis,
    ensuring the output retains the same shape as the input.

    Parameters:
    X (numpy.ndarray): Input data of shape (..., n_timepoints, ...).
    freq (float): Frequency to be removed in Hz (default is 60 Hz).
    fs (int): Sampling frequency in Hz.
    Q (float): Quality factor of the notch filter (higher Q means narrower notch, default is 30).
    axis (int): Axis along which to apply the filter.

    Returns:
    numpy.ndarray: Filtered data of the same shape as the input.
    """
    # Design a notch filter at the target frequency
    b, a = iirnotch(w0=freq / (0.5 * fs), Q=Q)
    
    # Apply the filter along the specified axis
    def apply_filter(data_slice):
        return filtfilt(b, a, data_slice, padlen=0)
    
    filtered_X = np.apply_along_axis(apply_filter, axis=axis, arr=X)
    
    return filtered_X


def zscore_standardize(X, axis=1):
    """
    Apply Z-score standardization along the specified axis of the input data,
    ensuring the output retains the same shape as the input.
    Handles cases where standard deviation is zero by avoiding division by zero.

    Parameters:
    X (numpy.ndarray): Input data of shape (..., n_timepoints, ...).
    axis (int): Axis along which to standardize (default is 1).

    Returns:
    numpy.ndarray: Z-score standardized data of the same shape as the input.
    """
    # Calculate the mean and standard deviation along the specified axis
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    
    # Avoid division by zero by adding a small epsilon to std where it is zero
    epsilon = 1e-8
    std_safe = np.where(std == 0, epsilon, std)
    
    # Apply Z-score standardization
    standardized_X = (X - mean) / std_safe
    
    return standardized_X



def dwt_reconstruct(X, threshold=0.1):
    """
    Applies Daubechies-4 DWT, thresholds coefficients for denoising, 
    and reconstructs the data along the 32-channel dimension.
    
    Parameters:
        X (numpy.ndarray): Input EEG data of shape (n_samples, n_channels, n_features).
                           Assumes n_channels is 32.
        threshold (float): Threshold value for wavelet coefficient denoising.
                           
    Returns:
        numpy.ndarray: Denoised and reconstructed data with the same shape as input X.
    """
    # Initialize an array to store reconstructed data
    reconstructed_data = np.zeros_like(X)
    
    # Loop through each time point and feature
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):  # Loop over the feature dimension
            # Apply DWT
            coeffs = pywt.wavedec(X[i, :, j], 'db4', mode='symmetric', level=None)
            
            # Apply soft thresholding for denoising
            coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            
            # Reconstruct the signal using thresholded coefficients
            reconstructed_data[i, :, j] = pywt.waverec(coeffs_thresholded, 'db4', mode='symmetric')[:32]
    
    return reconstructed_data

def minmax_scale(X, axis=1):
    """
    Apply Min-Max scaling along the specified axis of the input data,
    ensuring the output retains the same shape as the input.
    Scales the data to the range [0, 1].

    Parameters:
    X (numpy.ndarray): Input data of shape (..., n_timepoints, ...).
    axis (int): Axis along which to scale (default is 1).

    Returns:
    numpy.ndarray: Min-Max scaled data of the same shape as the input.
    """
    # Calculate the minimum and maximum along the specified axis
    min_val = np.min(X, axis=axis, keepdims=True)
    max_val = np.max(X, axis=axis, keepdims=True)
    
    # Avoid division by zero when max_val equals min_val
    range_safe = max_val - min_val
    epsilon = 1e-8
    range_safe = np.where(range_safe == 0, epsilon, range_safe)
    
    # Apply Min-Max scaling
    scaled_X = (X - min_val) / range_safe
    
    return scaled_X

from sklearn.decomposition import FastICA

def apply_ica_eeg(data):
    """
    Apply ICA along the last dimension of EEG data (timepoints).

    Parameters:
    data (numpy.ndarray): Input EEG data of shape (n_trials, n_channels, n_timepoints).

    Returns:
    numpy.ndarray: EEG data with ICA applied, same shape as input.
    """
    n_trials, n_channels, n_timepoints = data.shape
    transformed_data = np.zeros_like(data)  # To store ICA-transformed data
    
    # Apply ICA for each trial and channel independently
    for trial in range(n_trials):
        for channel in range(n_channels):
            # Reshape to (n_timepoints, 1) for ICA
            signal = data[trial, channel, :].reshape(-1, 1)
            
            # Perform ICA
            ica = FastICA(n_components=1, random_state=0)
            transformed_signal = ica.fit_transform(signal)
            
            # Store the transformed signal back into the array
            transformed_data[trial, channel, :] = transformed_signal.ravel()
    
    return transformed_data