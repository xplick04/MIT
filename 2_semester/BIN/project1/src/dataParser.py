from pyedflib import highlevel
import pyedflib as plib
import numpy as np
import matplotlib.pyplot as plt
import os
import mne
import torch


"""
'H' stands for Healthy Controls
'MDD' stands for Major Depressive Disorder 
'EC' stands for eyes closed
'EO' stands for eyes open
'TASK' stands for P300 data
"""
class DataParser:
    dataset = []

    def load_dataset(self):
        path = os.path.join(os.getcwd(), "data")

        for file in os.listdir(path):
            filename = os.fsdecode(file)
            data = mne.io.read_raw_edf(os.path.join(path, filename), preload=True)
            if 'H' in filename:
                self.dataset.append({
                    'Name': filename,
                    'data': data,
                    'label': 0,
                    'label_name': 'Healthy Controls'
                })
            elif 'MDD' in filename:
                self.dataset.append({
                    'Name': filename,
                    'data': data,
                    'label': 1,
                    'label_name': 'MDD Controls'
                })
            break
    
    def preprocess_data(self):
        # Iterate through the loaded dataset
        for d in self.dataset:
            # Iterate through the channels of each data
            for channel in d['data'].info['ch_names']:
                # Access the channel data
                channel_data = d['data'][channel][0][0] # metadata is [1] and data is [0], data inside np.array
                sfreq = int(d['data'].info['sfreq']) # 256Hz

                if 'signals_data' not in d:
                    d['signals_data'] = {}
                window_start = 0
                window_end = sfreq * 2 # 2 seconds window
                overlap = sfreq # 1 second overlap
                channel_bandwidths = []

                while True:
                    if window_end >= len(channel_data):
                        break
                    window_data = channel_data[window_start:window_end]

                    # Compute the PSD of the channel data
                    psd = np.abs(np.fft.fft(window_data))**2 # only positive frequencies
                    frequencies = np.abs(np.fft.fftfreq(len(window_data), d=1/sfreq)) # one sided spectrum
                
                    # Not including frequencies over 50Hz
                    bandwidths = [0,4,8,12,20,30,50]
                    mean_bands_values = []
                    for i in range(len(bandwidths)-1):
                        mean_bands_values.append(np.mean(psd[(frequencies >= bandwidths[i]) & (frequencies < bandwidths[i+1])]))
                    channel_bandwidths.append(np.array(mean_bands_values))
                    # Move the window
                    window_start += overlap
                    window_end += overlap
                    # Add the computed bandwidths to the dataset
                channel_bandwidths = np.array(channel_bandwidths)
                d['signals_data'][f'{channel}'] = channel_bandwidths  # each channel has 6 values each representing avarage bandwidth power of one window (original signal is divided into 2 seconds windows with 1 second overlap)
                
            d.pop('data') # remove original data
            

    
            
