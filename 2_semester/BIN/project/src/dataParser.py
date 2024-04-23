import numpy as np
import matplotlib.pyplot as plt
import os
import mne
import json


"""
'H' stands for Healthy Controls
'MDD' stands for Major Depressive Disorder 
'EC' stands for eyes closed
'EO' stands for eyes open
'TASK' stands for P300 data
"""
class DataParser:
    dataset = []
    min_duration = 0

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
        min_duration = float('inf')
        path = os.path.join(os.getcwd(), dPAth)
        for file in os.listdir(path):
            if not file.endswith('.edf'):
                continue
            filename = os.fsdecode(file)
            data = mne.io.read_raw_edf(os.path.join(path, filename), preload=True)
            recording_duration = len(data['EEG Fp1-LE'][1]) / data.info['sfreq']

            self.dataset.append({
                'Name': filename,
                'data': data,
                'label': 0,
                'label_name': 'Healthy' if 'H' in filename else 'MDD',
                'recording_duration': recording_duration
            })

            if recording_duration < min_duration: # get the minimum duration of all recordings for normalization
                min_duration = recording_duration

        self.min_duration = min_duration
        

    
    def preprocess_data(self):

        min_band_value = [float('inf') for i in range(6)] # normalize the data
        max_band_value = [float('-inf') for i in range(6)] # normalize the data

        # Iterate through the loaded dataset
        for d in self.dataset:
            # Iterate through the channels of each data
            for channel in d['data'].info['ch_names']:
                self.channels_occurrences[channel] += 1
                # Access the channel data
                channel_data = d['data'][channel][0][0] # metadata is [1] and data is [0], data inside np.array
                sfreq = int(d['data'].info['sfreq']) # 256Hz

                if 'signals_data' not in d:
                    d['signals_data'] = {}

                # Compute the PSD of the channel data
                channel_data = channel_data[:int(self.min_duration * sfreq)] # cut the data to the minimum duration
                psd = np.abs(np.fft.fft(channel_data))**2 # only positive frequencies
                frequencies = np.abs(np.fft.fftfreq(len(channel_data), d=1/sfreq)) # one sided spectrum

                # Not including frequencies over 50Hz
                bandwidths = [0,4,8,12,20,30,50]
                mean_bands_values = []
                for i in range(len(bandwidths)-1):
                    band_value = np.mean( psd[(frequencies >= bandwidths[i]) & (frequencies < bandwidths[i+1])] )
                    mean_bands_values.append(band_value)
                    if band_value < min_band_value[i]:
                        min_band_value[i] = band_value
                    if band_value > max_band_value[i]:
                        max_band_value[i] = band_value

                d['signals_data'][channel] = np.array(mean_bands_values)  # each channel has 6 values each representing avarage bandwidth power of signal
            
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

        # Normalize each band data for remaining channels
        for d in self.dataset:
            for key in d['signals_data']:
                for i in range(6):
                    d['signals_data'][key][i] = (d['signals_data'][key][i] - min_band_value[i]) / (max_band_value[i] - min_band_value[i]) * 100 # normalize to 0-100 range
            

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
            
    
            
