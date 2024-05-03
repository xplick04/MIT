from sklearn import svm, naive_bayes, ensemble, neural_network, linear_model, metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from sklearn.model_selection import KFold
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class SVM:
    def __init__(self):
        self.clf = svm.SVC(probability=True)

    def train_model(self, train_data):
        train_x, labels = train_data[:, :-1], train_data[:, -1]
        self.clf.fit(train_x, labels)

    def predict(self, test_data):
        return self.clf.predict(test_data)
    
    def evaluate_model(self, test_data):
        test_x, labels = test_data[:, :-1], test_data[:, -1]
        predictions = self.predict(test_x)
        accuracy = self.clf.score(test_x, labels)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return accuracy, sensitivity, specificity
    
class LogisticRegressionModel:
    def __init__(self):
        self.clf = LogisticRegression()

    def train_model(self, train_data):
        train_x, labels = train_data[:, :-1], train_data[:, -1]
        self.clf.fit(train_x, labels)

    def predict(self, test_data):
        return self.clf.predict(test_data)

    def evaluate_model(self, test_data):
        test_x, labels = test_data[:, :-1], test_data[:, -1]
        predictions = self.predict(test_x)
        accuracy = accuracy_score(labels, predictions)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return accuracy, sensitivity, specificity


class Bayes:
    def __init__(self):
        self.clf = naive_bayes.GaussianNB()

    def train_model(self, train_data):
        train_x, labels = train_data[:, :-1], train_data[:, -1]
        self.clf.fit(train_x, labels)

    def predict(self, test_data):
        return self.clf.predict(test_data)
    
    def evaluate_model(self, test_data):
        test_x, labels = test_data[:, :-1], test_data[:, -1]
        predictions = self.predict(test_x)
        accuracy = accuracy_score(labels, predictions)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return accuracy, sensitivity, specificity



def get_data(dataset, idx):
    data = None
    for i, d in enumerate(dataset):
        if i in idx:
            if data is None:
                data = d
            else:
                data = np.concatenate((data, d), axis=0)
    return torch.tensor(data).float()


def cross_validation(data):
    kf = KFold(n_splits=10, shuffle=True)
    model1 = SVM()
    model2 = LogisticRegressionModel()
    model3 = Bayes()
    models = [model1, model2, model3]
    
    all_accuracies = []
    all_sensitivities = []
    all_specificities = []
    
    for model in models:
        accuracies = []
        sensitivities = []
        specificities = []
        for foldID, (train_index, test_index) in enumerate(kf.split(data)):
            train_data = get_data(data, train_index)
            test_data = get_data(data, test_index)

            model.train_model(train_data)
            val_accuracy, sensitivity, specificity = model.evaluate_model(test_data)
            accuracies.append(val_accuracy)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            
        all_accuracies.append(accuracies)
        all_sensitivities.append(sensitivities)
        all_specificities.append(specificities)

    return all_accuracies, all_sensitivities, all_specificities


