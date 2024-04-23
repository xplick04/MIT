from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import sys
from tqdm import tqdm


config = {
    'learning_rate' : 0.1,
    'epochs' : 10,
    'batch_size' : 16, 
    'layer_width' : 8
}


class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, config['layer_width']),
            torch.nn.ReLU(),
            torch.nn.Linear(config['layer_width'], config['layer_width']),
            torch.nn.ReLU(),
            torch.nn.Linear(config['layer_width'], 1),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        y = self.layers(x)
        return y
    

    def train_model(self, train_data):
        train_x, labels = train_data[:, :-1], train_data[:, -1]
        optimizer = optim.Adam(self.parameters(), lr=config["learning_rate"])
        criterion = torch.nn.BCELoss()

        batch_size = config['batch_size']
        num_samples = train_data.shape[0]
        num_batches = num_samples // batch_size

        for epoch in range(config["epochs"]):
            epoch_loss = 0
            epoch_accuracy = 0
            # Shuffle data for each epoch
            indices = torch.randperm(num_samples)
            train_x_shuffled = train_x[indices]
            labels_shuffled = labels[indices]

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                x_batch = train_x_shuffled[start_idx:end_idx]
                label_batch = labels_shuffled[start_idx:end_idx]

                optimizer.zero_grad()
                y_batch = self.forward(x_batch)
                loss = criterion(y_batch.squeeze(), label_batch)  # Rename the variable to avoid overwriting the loss function
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                epoch_accuracy += ((y_batch > 0.5).float() == label_batch).float().mean().item()

            epoch_loss /= num_batches
            epoch_accuracy /= num_batches

            print(f'Epoch: {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy*100}%')

    
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


    def cross_validation(self, dataset, input_dim):
        kf = KFold(n_splits=5, shuffle=True)
        accuracies = []

        for foldID, (train_index, test_index) in enumerate(kf.split(dataset)):
            self.__init__(input_dim) # Reinitialize the model
            train_data = torch.tensor(dataset[train_index]).to(torch.float32) # Convert to float32
            test_data = torch.tensor(dataset[test_index]).to(torch.float32) # Convert to float32
            self.train_model(train_data)
            val_accuracy, val_loss = self.evaluate_model(test_data)
            accuracies.append(val_accuracy)
            print(f'Fold: {foldID}, Loss: {val_loss}, Accuracy: {val_accuracy*100}%')

        print(f'Average accuracy: {np.mean(accuracies)*100}%')