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

    channels_occurrences = {
        'EEG Fp1-LE': 0,
        'EEG F3-LE': 0,
        'EEG C3-LE': 0,
        'EEG P3-LE': 0,
        'EEG O1-LE': 0,
        'EEG F7-LE': 0,
        'EEG T3-LE': 0,
        'EEG T5-LE': 0,
        'EEG Fz-LE': 0,
        'EEG Fp2-LE': 0,
        'EEG F4-LE': 0,
        'EEG C4-LE': 0,
        'EEG P4-LE': 0,
        'EEG O2-LE': 0,
        'EEG F8-LE': 0,
        'EEG T4-LE': 0,
        'EEG T6-LE': 0,
        'EEG Cz-LE': 0,
        'EEG Pz-LE': 0,
        'EEG A2-A1': 0,
        'EEG 23A-23R': 0,
        'EEG 24A-24R': 0
    }

    def load_dataset(self, dPAth):
        path = os.path.join(os.getcwd(), dPAth)

        i = 0
        for file in os.listdir(path):
            if file.endswith(".edf"):
                filename = os.fsdecode(file)
                data = mne.io.read_raw_edf(os.path.join(path, filename), preload=True)
                self.dataset.append({
                    'Name': filename,
                    'data': data,
                    'label': 0 if 'H' in filename else 1,
                    'label_name': 'Healthy' if 'H' in filename else 'MDD'
                })

    
    def preprocess_data(self):
        
        min_band_value = [float('inf') for i in range(6)] # normalize the data
        max_band_value = [float('-inf') for i in range(6)] # normalize the data

        # Iterate through the loaded dataset
        for d in self.dataset:
            # Iterate through the channels of each data
            for channel in d['data'].info['ch_names']:

                self.channels_occurrences[channel] += 1
                channel_data = d['data'][channel][0][0] # metadata is [1] and data is [0], data inside np.array
                sfreq = int(d['data'].info['sfreq']) # 256Hz

                if 'signals_data' not in d:
                    d['signals_data'] = {}

                window_size = sfreq * 60 # 60 seconds window
                channel_bandwidths = []
                i = 0
                flag = False

                while True:
                    if (i + 1) * window_size >= len(channel_data):
                        window_data = channel_data[-window_size:]
                        flag = True
                    else:
                        window_data = channel_data[i * window_size : (i + 1) * window_size]

                    # Compute the PSD of the channel data
                    psd = np.abs(np.fft.fft(window_data))**2
                    frequencies = np.abs(np.fft.fftfreq(len(window_data), d=1/sfreq))
                    
                    # Compute mean band values
                    bandwidths = [0, 4, 8, 12, 20, 30, 50]
                    mean_bands_values = []
                    for j in range(len(bandwidths) - 1):
                        band_value = np.mean( psd[(frequencies >= bandwidths[j]) & (frequencies < bandwidths[j+1])] )
                        mean_bands_values.append(band_value)
                        if band_value < min_band_value[j]:
                            min_band_value[j] = band_value
                        if band_value > max_band_value[j]:
                            max_band_value[j] = band_value

                    # Add the computed bandwidths to the dataset
                    channel_bandwidths.append(mean_bands_values) # shape (n_windows, 6)
                    i += 1
                    if flag:
                        break
                
                d['signals_data'][channel] = np.array(channel_bandwidths)
            d.pop('data') # remove original data

        max_occurrences = max(self.channels_occurrences.values()) # get the maximum number of occurrences of a channel

        # Find channels with occurrences less than the maximum
        channels_to_remove = []
        for key in self.channels_occurrences:
            if self.channels_occurrences[key] < max_occurrences:
                channels_to_remove.append(key)

        # Remove channels that are not common
        for d in self.dataset:
            for key in channels_to_remove:
                if key in d['signals_data']:
                    del d['signals_data'][key]

        """
        # Normalize each band data for remaining channels
        for d in self.dataset: # iterate through the data
            for key in d['signals_data']:
                for i in range(len(d['signals_data'][key])):  # Iterate over the first dimension
                    for j in range(len(d['signals_data'][key][i])):  # Iterate over the second dimension
                        d['signals_data'][key][i][j] = (d['signals_data'][key][i][j] - min_band_value[j]) / (max_band_value[j] - min_band_value[j]) * 100
        """



    def save_data(self, path):
        filename = "processed_data.npy"
        structured_data = []
        for d in self.dataset:
            # Convert signals_data to lists
            for key in d['signals_data']:
                d['signals_data'][key] = d['signals_data'][key].tolist()
            # Create structured array
            dt = np.dtype([('Name', 'U50'), ('label', int), ('label_name', 'U50'), ('signals_data', object)])
            structured_data.append((d['Name'], d['label'], d['label_name'], d['signals_data']))
        
        # Convert to NumPy structured array
        structured_array = np.array(structured_data, dtype=dt)

        # Save the structured array
        np.save(os.path.join(path, filename), structured_array)


    def load_data(self, path):
        filename = "processed_data.npy"
        structured_array = np.load(os.path.join(path, filename), allow_pickle=True)
        data = []
        for d in structured_array:
            data.append({
                'Name': d[0],
                'label': d[1],
                'label_name': d[2],
                'signals_data': d[3]
            })
        return data
            