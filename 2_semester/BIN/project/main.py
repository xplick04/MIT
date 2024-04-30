import numpy as np
import matplotlib.pyplot as plt
import src.dataParser as d
import os
import src.MLP as m
import src.SVM as s
import src.CGP as c
import sys


def convert(dataset):
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

    elif "--svm" in sys.argv:
        data = parser.load_data("data/processed_data/") #181 files
        data = convert(data) # List of files, each file is a numpy array of shape (time, channels*bands + label)
        s.cross_validation(data)

    elif "--cgp" in sys.argv:
        data = parser.load_data("data/processed_data/") #181 files
        data = convert(data) # List of files, each file is a numpy array of shape (time, channels*bands + label)
        c.cross_validation(data, num_generations=100, pop_size=100, MUTATION_MAX=10, lookback=2, x_size=10, y_size=10)
        


            
