from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt


config = {
    'learning_rate' : 0.00001,
    'epochs' : 10000,
    'batch_size' : 32,
    'layer_width' : 128
}


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = []

        self.layers.append(nn.Linear(input_dim, config["layer_width"]))
        self.layers.append(nn.Sigmoid())  # ReLU activation
        self.layers.append(nn.Linear(config["layer_width"], config["layer_width"]))
        self.layers.append(nn.Sigmoid())  # ReLU activation
        self.layers.append(nn.Linear(config["layer_width"], config["layer_width"]))
        self.layers.append(nn.Sigmoid())  # ReLU activation
        self.layers.append(nn.Linear(config["layer_width"], 1))
        self.layers.append(nn.Sigmoid())  # Sigmoid for binary classification

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x
    

    def train_model(self, train_data):
        accuracies = []
        loss_values = []
        self.train()  # Set the model to training mode

        train_x, labels = train_data[:, :-1], train_data[:, -1]
        optimizer = optim.Adam(self.parameters(), lr=config["learning_rate"])
        criterion = torch.nn.BCELoss()

        for epoch in range(config["epochs"]):
            epoch_loss = 0
            epoch_accuracy = 0

            optimizer.zero_grad()
            y_batch = self.forward(train_x)
            loss = criterion(y_batch.squeeze(), labels)  # Calculate loss for the entire dataset
            epoch_loss = loss.item()
            loss.backward()
            optimizer.step()

            epoch_accuracy = ((y_batch > 0.5).float() == labels).float().mean().item()
            print(f'Epoch {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy*100:.2f}%')
            loss_values.append(epoch_loss)
            accuracies.append(epoch_accuracy)
        return accuracies, loss_values
    

    def evaluate_model(self, eval_data):
        test_x, labels = eval_data[:, :-1], eval_data[:, -1]
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            y = self.forward(test_x)
            # Compute accuracy
            predictions = (y > 0.5).float()  # Threshold at 0.5
            val_accuracy = (predictions == labels).float().mean().item()
            return val_accuracy


def cross_validation(dataset):
    input_dim = dataset[0].shape[1] - 1  # Number of features (channels*bands - 1 for the label column)
    kf = KFold(n_splits=5, shuffle=True)
    accuracies = []

    for foldID, (train_index, test_index) in enumerate(kf.split(dataset)):
        train_data = get_data(dataset, train_index)
        test_data = get_data(dataset, test_index)
        model = MLP(input_dim)

        train_acc, train_loss = model.train_model(train_data)
        val_accuracy = model.evaluate_model(test_data)
        accuracies.append(val_accuracy)
        plot_data(train_acc, "train_acc")
        plot_data(train_loss, "train_loss")
        print(f'Fold {foldID}, Accuracy: {val_accuracy*100:.2f}%')
        break


def get_data(dataset, idx):
    data = None
    for i, d in enumerate(dataset):
        if i in idx:
            if data is None:
                data = d
            else:
                data = np.concatenate((data, d), axis=0)
    return torch.tensor(data).float()


def plot_data(data, filename):
    plt.plot(data)
    plt.savefig(filename + ".png")
    plt.close()
