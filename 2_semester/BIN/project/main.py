import numpy as np
import matplotlib.pyplot as plt
import src.dataParser as d
import os
import src.others as o
import src.CGP as c
import sys
import matplotlib.pyplot as plt

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


def plot_metrics(metrics, metric_name):
    plt.figure()
    bp = plt.boxplot(metrics, patch_artist=True, showmeans=True, meanline=True, medianprops=dict(color='black', linewidth=1), widths=0.4)

    # Customize boxplot elements
    plt.title(f"{metric_name} for Different Models")
    plt.ylabel(metric_name)
    plt.xlabel("Models")
    
    # Customize x-axis labels and positions
    model_names = ['SVM', 'Logistic Regression', 'Bayes', 'CGP']
    plt.xticks(range(1, len(model_names) + 1), model_names)

    # Add a line from median to left axis with exact number for each boxplot
    for i, column_data in enumerate(metrics):
        median_val = np.median(column_data)
        plt.text(i + 1.22, median_val, f'{median_val:.2f}', ha='left', va='center', color='black')

    # Customize boxplot colors
    colors = ['lightblue'] * len(model_names)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Customize mean line
    plt.setp(bp['means'], color='green', linewidth=1)

    plt.show()



if __name__ == "__main__":
    parser = d.DataParser()
    if "--process_data" in sys.argv:
        parser.load_dataset("data/")
        parser.preprocess_data()
        parser.save_data("data/processed_data/")
        print("Data processed and saved")

    elif "--others" in sys.argv:
        data = parser.load_data("data/processed_data/") #181 files
        data = convert(data) # List of files, each file is a numpy array of shape (time, channels*bands + label)
        o.cross_validation(data)

    elif "--cgp" in sys.argv:
        data = parser.load_data("data/processed_data/") #181 files
        data = convert(data) # List of files, each file is a numpy array of shape (time, channels*bands + label)
        c.cross_validation(data, num_generations=100, pop_size=50, MUTATION_MAX=20, lookback=2, x_size=5, y_size=5) # default values

    elif "--boxplot" in sys.argv:
        acc = []
        sens = []
        spec = []
        accOthers = []
        sensOthers = []
        specOthers = []
        data = parser.load_data("data/processed_data/")
        data = convert(data)
        acc, sens, spec = c.cross_validation(data, num_generations=100, pop_size=500, MUTATION_MAX=20, lookback=1, x_size=7, y_size=7) # best parameters
        """
        accOthers, sensOthers, specOthers = o.cross_validation(data) # SVM, Logistic Regression, Bayes
        accOthers.append(acc)
        sensOthers.append(sens)
        specOthers.append(spec)
        plot_metrics(accOthers, "Accuracy")
        plot_metrics(sensOthers, "Sensitivity")
        plot_metrics(specOthers, "Specificity")"""




        


            
