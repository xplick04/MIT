from pyedflib import highlevel
import pyedflib as plib
import numpy as np
import matplotlib.pyplot as plt
import src.dataParser as d
import os
import src.MLP as m
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
                merged_channels = np.concatenate((merged_channels, channel_data), axis=0)  # Concatenate along axis=1 (bands), data are now in shape (channels*bands)
        
        class_labels = np.array([d['label']])  # Create a column vector of class labels
        merged_channels = np.concatenate((merged_channels, class_labels), axis=0) # add class labels to the end of the data
        data.append(merged_channels)
    return np.array(data) 



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
        input_dim = data.shape[1] - 1  # Number of features (channels*bands - 1 for the label column)
        mlp = m.MLP(input_dim)
        mlp.cross_validation(data, input_dim)

        

            
