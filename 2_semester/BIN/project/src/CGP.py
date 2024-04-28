import os
import random
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

# CGP parameters
MUTATION_MAX = 10
# CGP dimensions
x_size = 10
y_size = 10
lookback = 2 # must be larger than 0, lookback 1 = can see previous column


class Individual:
    num_inputs = 0
    num_outputs = 0
    chromozome = [] # first column, then second column, etc.
    outputValues = [] # inputs + gene outputs

    def __init__(self, outputValues):
        self.outputValues = outputValues
        # random genes initialization
        self.chromozome = [0 for _ in range((x_size * y_size) * 3 + 1)] # triple for each gate + output element
        for i in range(x_size):  # col idx
            for j in range(y_size):  # row idx

                if i - lookback < 0:
                    #print(0, input_size + y_size * i - 1)
                    i1 = random.randint(0, input_size + y_size * i - 1)  # input_size - 1 for indexing
                    i2 = random.randint(0, input_size + y_size * i - 1)

                elif i - lookback == 0:  # lookback, cannot see input column
                    #print(input_size, input_size + y_size * i - 1)
                    i1 = random.randint(input_size, input_size + y_size * i - 1)  # input_size - 1 for indexing
                    i2 = random.randint(input_size, input_size + y_size * i - 1)

                else:
                    #print(input_size + y_size * (i - lookback), input_size + y_size * i - 1)
                    i1 = random.randint(input_size + y_size * (i - lookback), input_size + y_size * i - 1)  # input_size - 1 for indexing
                    i2 = random.randint(input_size + y_size * (i - lookback), input_size + y_size * i - 1)

                op = random.randint(0, 3)  # 0: add, 1: sub, 2: mul, 3: div
                gate_index = (i * y_size + j) * 3
                self.chromozome[gate_index] = i1
                self.chromozome[gate_index + 1] = i2
                self.chromozome[gate_index + 2] = op

        self.chromozome[-1] = random.randint(input_size + (x_size - lookback) * y_size, self.outputValues.shape[1] - 2) # output can be any of the elements except itself and inputs, -2 for indexing


    def execute(self):
        for i in range(x_size):
            for j in range(y_size):
                for k in range(self.outputValues.shape[0]):
                    gate = (self.chromozome[(i * y_size + j) * 3], self.chromozome[(i * y_size + j) * 3 + 1], self.chromozome[(i * y_size + j) * 3 + 2])

                    if gate[2] == 0: # addition
                        self.outputValues[k][(i * y_size + j) + input_size] = self.outputValues[k][gate[0]] + self.outputValues[k][gate[1]]

                    elif gate[2] == 1: # subtraction
                        self.outputValues[k][(i * y_size + j) + input_size] = self.outputValues[k][gate[0]] - self.outputValues[k][gate[1]]

                    elif gate[2] == 2: # multiplication
                        self.outputValues[k][(i * y_size + j) + input_size] = self.outputValues[k][gate[0]] * self.outputValues[k][gate[1]]

                    elif gate[2] == 3: # division by zero
                        if self.outputValues[k][gate[1]] == 0:
                            self.outputValues[k][(i * y_size + j) + input_size] = 0
                        else:
                            self.outputValues[k][(i * y_size + j) + input_size] = self.outputValues[k][gate[0]] / self.outputValues[k][gate[1]]

        # get thresholded output values
        treshold = 0
        for k in range(self.outputValues.shape[0]):

            if self.outputValues[k][self.chromozome[-1]] > treshold:
                self.outputValues[k][-1] = 1
            else:
                self.outputValues[k][-1] = 0

        return self.outputValues[:,-1] # return vector of outputs


    
    def mutate(self):
        num_mutations = random.randint(0, MUTATION_MAX)
        for _ in range(num_mutations):
            idx = random.randint(0, (x_size * y_size) * 3) # gates + output index

            if idx == (x_size * y_size) * 3 : # output, last element
                self.chromozome[idx] = random.randint(input_size + (x_size - lookback) * y_size, self.outputValues.shape[1] - 2) 
                continue

            if idx % 3 == 2:    # operator
                self.chromozome[idx] = random.randint(0, 3)
            else:   # input
                col_idx = idx // (3 * y_size)
                #print(col_idx)
                if col_idx - lookback < 0:
                    #print(0, input_size + y_size * (col_idx) - 1)
                    self.chromozome[idx] = random.randint(0, input_size + y_size * (col_idx) - 1)
                elif col_idx - lookback == 0: # lookback, cannot see input column
                    #print(input_size, input_size + y_size * (col_idx) - 1)
                    self.chromozome[idx] = random.randint(input_size, input_size + y_size * (col_idx) - 1)
                else:
                    #print(input_size + y_size * ((col_idx - 1) - lookback), input_size + y_size * (col_idx) - 1)
                    self.chromozome[idx] = random.randint(input_size + y_size * ((col_idx) - lookback), input_size + y_size * (col_idx) - 1)

    def copy(self):
        copied_individual = Individual(self.outputValues[:])  # Copying outputValues
        copied_individual.chromozome = self.chromozome[:]     # Copying chromozome
        return copied_individual

    def print_chromozome(self):
        for i in range(x_size):
            for j in range(y_size):
                print(f"({self.chromozome[(i * y_size + j) * 3]},{self.chromozome[(i * y_size + j) * 3 + 1]},{self.chromozome[(i * y_size + j) * 3 + 2]})", end='')
        print(self.chromozome[-1])



class Population:
    population = []
    features = []
    labels = []
    best_individual = None

    def __init__(self, pop_size, features, labels):
        self.population = []
        self.features = features
        self.labels = labels
        outputValues = np.zeros((features.shape[0], x_size * y_size + 1)) # create output values for each individual
        outputValues = np.concatenate((features, outputValues), axis=1)
        # create L + 1 random individuals
        for _ in range(pop_size + 1):
            self.population.append(Individual(outputValues))


    def evaluate(self):
        fitness = []
        for individual in self.population:
            result = individual.execute()  # Assuming execute returns a vector of results (one training dato result per row)
            error = np.linalg.norm(result - self.labels)  # Compute the error between result vector and labels
            fitness.append(error)
        return fitness
    
    def train(self, num_generations):
        for g in tqdm(range(num_generations), desc="Training"):

            fitness = self.evaluate()

            # get best individual
            best_fitness, best_individual_idx = min(fitness), fitness.index(min(fitness))
            best_individual = self.population[best_individual_idx]

            # Replace all individuals in the population with copies of the best individual
            self.population = [best_individual.copy() for _ in range(len(self.population))]

            # Mutate all individuals except the first (best) one
            for individual in self.population[1:]:
                individual.mutate()

            #print(f"Generation {g}, Best fitness: {best_fitness:.2f}")
        self.best_individual = self.population[0]

        
    def test(self, test_features, test_labels):
        test_outputValues = np.zeros((test_features.shape[0], x_size * y_size + 1))
        test_outputValues = np.concatenate((test_features, test_outputValues), axis=1)
        test_individual = Individual(test_outputValues)
        test_individual.chromozome = self.best_individual.chromozome
        test_result = test_individual.execute()
        test_accuracy = (test_labels == test_result).sum() / len(test_labels)
        print(f"Validation accuracy: {test_accuracy*100:.2f}%")


def get_data(data, idx):
    dataOut = np.array([])  # Initialize as an empty array
    for i, d in enumerate(data):
        if i in idx:
            if dataOut.size == 0:  # Check if dataOut is empty
                dataOut = d
            else:
                dataOut = np.concatenate((dataOut, d), axis=0)
    return dataOut



def cross_validation(data, num_generations=10, pop_size=2):
    global input_size

    kf = KFold(n_splits=10, shuffle=True)
    # K fold cross validation
    for foldID, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"---Fold {foldID + 1}---")
        train_data = get_data(data, train_index)
        test_data = get_data(data, test_index)
        train_features, train_labels = train_data[:, :-1], train_data[:, -1]
        test_features, test_labels = test_data[:, :-1], test_data[:, -1]
        input_size = train_features.shape[1]

        pop = Population(pop_size, train_features, train_labels)
        pop.train(num_generations)
        pop.test(test_features, test_labels)




    
            

    




