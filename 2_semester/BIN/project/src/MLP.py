from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import sys


config = {
    'learning_rate' : 0.001,
    'epochs' : 10,
    'batch_size' : 32
}


class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        y = self.layers(x)
        return y
    

    def train(self, dataset):
        train_x, labels = self.get_features_labels(dataset)
        optimizer = optim.Adam(self.parameters(), lr=config["learning_rate"])
        criterion = torch.nn.BCELoss()

        for epoch in range(config["epochs"]):
            optimizer.zero_grad()
            y = self.forward(train_x)
            loss_value = criterion(y.squeeze(), labels)  # Rename the variable to avoid overwriting the loss function
            loss_value.backward()
            optimizer.step()
            train_accuracy = ((y > 0.5).float() == labels).float().mean().item()
            print(f'Epoch: {epoch}, Loss: {loss_value.item()}, Accuracy: {train_accuracy}')

    
    def evaluate(self, dataset):
        test_x, labels = self.get_features_labels(dataset)
        self.eval()
        with torch.no_grad():
            y = self.forward(test_x)
            loss = torch.nn.BCELoss()
            loss = loss(y, labels)
            # Compute accuracy
            predictions = (y > 0.5).float()  # Threshold at 0.5
            val_accuracy = ((predictions == labels).float().mean().item())
            return val_accuracy, loss.item()


    def cross_validation(self, dataset):
        print(len(dataset[0].shape))
        return
        kf = KFold(n_splits=5, shuffle=True)

        accuracies = []

        for foldID, (train_index, test_index) in enumerate(kf.split(dataset)):
            train_data = dataset[train_index] # get files for training
            test_data = dataset[test_index] # get files for testing
            self.train(train_data)
            val_accuracy, val_loss = self.evaluate(test_data)
            accuracies.append(val_accuracy)
            print(f'Fold: {foldID}, Accuracy: {val_accuracy}, Loss: {val_loss}')

        print(f'Average accuracy: {np.mean(accuracies)}')
    

    def get_features_labels(self, dataset):
        features = []
        labels = []
        for d in dataset:
            features.extend(d[:,:-1])
            labels.extend(d[:,-1])

        return torch.tensor(np.array(features), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32)       