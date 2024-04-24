from pyedflib import highlevel
import pyedflib as plib
import numpy as np
import matplotlib.pyplot as plt
import src.dataParser2 as d
import os
import src.MLP as m
import src.MLPnoBatch as m2
import sys


def convert4MLP(dataset):
    data = []
    for d in dataset:
        merged_channels = None
        for channel in d['signals_data']:
            channel_data = d['signals_data'][channel]
            channel_data = np.array(channel_data)

            if merged_channels is None:
                merged_channels = channel_data  # Initialize merged_channels with the first channel data
            else:
                merged_channels = np.concatenate((merged_channels, channel_data), axis=1)  # Concatenate along axis=1 (bands), data are now in shape (windowSize,channels*bands)

        class_labels = np.array([d['label'] for i in range(merged_channels.shape[0])]).reshape(-1,1)  # Create a column vector of class labels (length = number of windows in the audio file)
        merged_channels = np.concatenate((merged_channels, class_labels), axis=1) # add class labels to the end of the data
        data.append(merged_channels)
    return data



if __name__ == "__main__":
    parser = d.DataParser()
    if "--process_data" in sys.argv:
        parser.load_dataset("data/")
        parser.preprocess_data()
        parser.save_data("data/processed_data/")
        print("Data processed and saved")

    elif "--cross_validation" in sys.argv:
        data = parser.load_data("data/processed_data/") #181 files
        data = convert4MLP(data) # List of files, each file is a numpy array of shape (time, channels*bands + label)
        #m.cross_validation(data)
        m2.cross_validation(data)

    elif "--cgp":
        pass

        

            
