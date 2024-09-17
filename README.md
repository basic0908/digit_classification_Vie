# EEG-based classification of imagined digits using low-density EEG
Digit classification using low-density EEG

![framework](https://github.com/user-attachments/assets/67ebcabd-4574-4d10-aeb4-e97151581142)


Aim : to classify the imagined speech of numerical digits from EEG signals by exploiting the past and future temporal characteristics of the signal using several deep learning models  
EEG signal processing : EEG signals were filtered and preprocessed using the discrete wavelet transform to remove artifacts and retrieve feature information  
Feature classification : multiple version sof multilayer bidirectional recurrent neural networks were used

### EEG data
EEG signals from each trial were recorded for 2 seconds. 
- EPOC 14 channels 14 channels 128Hz
- MUSE 4 channels 220 Hz

### Signal Processing
# Butterworth high-pass filter fo order 5 at 0.1 Hz to erase the low-frequencies noise
## Notch filter to remove 60 Hz electrical environment noise.
### Discrete wavelet transform(DWT) using the Daubechies-4 wavelet with two-level decomposition on EPOC and the three-level decompostion on MUSE for denoising and informaiton extraction. 
#### Inverse reconstruction of the original EEG waveform using DWT components

### Results 
![results](https://github.com/user-attachments/assets/4e9681a4-3567-4a79-81b3-cc0d7d1e24e6)


### Data preprocess
Raw -> preprocess(filters) -> reconstruction(wavelet transformation) -> standardization
![raw](plot/3_plot.png)

