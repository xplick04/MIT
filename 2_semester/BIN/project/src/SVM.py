from sklearn import svm
import numpy as np
from sklearn.model_selection import KFold
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class SVM:
    def __init__(self):
        self.clf = svm.SVC()

    def train_model(self, train_data):
        train_x, labels = train_data[:, :-1], train_data[:, -1]
        self.clf.fit(train_x, labels)

    def predict(self, test_data):
        return self.clf.predict(test_data)
    

    def evaluate_model(self, test_data):
        test_x, labels = test_data[:, :-1], test_data[:, -1]
        return self.clf.score(test_x, labels)
    


def cross_validation(data):
    kf = KFold(n_splits=10, shuffle=True)
    accuracies = []

    for foldID, (train_index, test_index) in enumerate(kf.split(data)):
        svm = SVM()
        train_data = get_data(data, train_index)
        test_data = get_data(data, test_index)

        svm.train_model(train_data)

        val_accuracy = svm.evaluate_model(test_data)
        accuracies.append(val_accuracy)
        print(f'Fold {foldID + 1}, Accuracy: {val_accuracy*100}%')

    print(f'Average accuracy: {np.mean(accuracies)*100}%')



def get_data(dataset, idx):
    data = None
    for i, d in enumerate(dataset):
        if i in idx:
            if data is None:
                data = d
            else:
                data = np.concatenate((data, d), axis=0)
    return torch.tensor(data).float()

    
def roc_curve(test_data, svm):
        # Compute ROC curve and ROC area
        probs = svm.predict(test_data[:,:-1])  # Get probabilities of the positive class
        fpr, tpr, _ = roc_curve(test_data[:,-1], probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('SVM - ROC curve')
        plt.legend(loc="lower right")
        plt.show()
