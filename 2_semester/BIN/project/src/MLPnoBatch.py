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
    'learning_rate' : 0.0001,
    'epochs' : 1000,
    'layer_width' : 32
}


class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, config['layer_width']),
            torch.nn.Sigmoid(),
            torch.nn.Linear(config['layer_width'], config['layer_width']),
            torch.nn.Sigmoid(),
            torch.nn.Linear(config['layer_width'], 1),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        y = self.layers(x)
        return y
    

    def train_model(self, train_data):

        accuracies = []
        loss_values = []
        train_x, labels = train_data[:, :-1], train_data[:, -1]
        optimizer = optim.Adam(self.parameters(), lr=config["learning_rate"])
        criterion = torch.nn.BCELoss()

        for epoch in range(config["epochs"]):
            self.train()  # Set the model to training mode
            optimizer.zero_grad()
            y = self.forward(train_x)
            loss_value = criterion(y.squeeze(), labels)  # Rename the variable to avoid overwriting the loss function
            loss_value.backward()
            
            """
            for name, param in self.named_parameters():
                if param.grad is not None:
                    print(f'Layer: {name}, Mean Gradient: {param.grad.mean()}, Max Gradient: {param.grad.abs().max()}')"""

            loss_values.append(loss_value.item())
            optimizer.step()
            train_accuracy = ((y > 0.5).float() == labels).float().mean().item()
            accuracies.append(train_accuracy)
            #print(f'Epoch: {epoch}, Loss: {loss_value.item()}, Accuracy: {train_accuracy*100}%')

        return accuracies, loss_values
    
    def evaluate_model(self, eval_data):
        test_x, labels = eval_data[:, :-1], eval_data[:, -1]
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            y = self.forward(test_x)
            criterion = torch.nn.BCELoss()
            loss = criterion(y.squeeze(), labels)
            # Compute accuracy
            predictions = (y > 0.5).float()  # Threshold at 0.5
            val_accuracy = (predictions == labels).float().mean().item()
            return val_accuracy, loss.item()

    
def get_data(dataset, idx):
    data = None
    for i, d in enumerate(dataset):
        if i in idx:
            if data is None:
                data = d
            else:
                data = np.concatenate((data, d), axis=0)
    return torch.tensor(data).float()




def cross_validation(dataset):
    input_dim = dataset[0].shape[1] - 1  # Number of features (channels*bands - 1 for the label column)
    kf = KFold(n_splits=5, shuffle=True)
    accuracies = []

    for foldID, (train_index, test_index) in enumerate(kf.split(dataset)):
        train_data = get_data(dataset, train_index)
        test_data = get_data(dataset, test_index)
        model = MLP(input_dim)

        train_acc, train_loss = model.train_model(train_data)
        val_accuracy, val_loss = model.evaluate_model(test_data)
        accuracies.append(val_accuracy)
        plot_data(train_acc)
        plot_data(train_loss)

        #print(f'Fold {foldID}, Loss: {val_loss}, Accuracy: {val_accuracy*100}%')
        break


def plot_data(data):
    plt.plot(data)
    plt.show()
