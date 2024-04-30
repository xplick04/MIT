import os
import random
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import wandb

STATISTICS = False


class Individual:
    num_inputs = 0
    num_outputs = 0
    chromozome = [] # first column, then second column, etc.
    outputValues = [] # inputs + gene outputs

    def __init__(self, outputValues):
        self.outputValues = outputValues
        # random genes initialization
        self.chromozome = [0 for _ in range((config['x_size'] * config['y_size']) * 3 + 1)] # triple for each gate + output element
        for i in range(config['x_size']):  # col idx
            for j in range(config['y_size']):  # row idx

                if i - config['lookback'] < 0:
                    #print(0, input_size + config['y_size'] * i - 1)
                    i1 = random.randint(0, input_size + config['y_size'] * i - 1)  # input_size - 1 for indexing
                    i2 = random.randint(0, input_size + config['y_size'] * i - 1)

                elif i - config['lookback'] == 0:  # lookback, cannot see input column
                    #print(input_size, input_size + config['y_size'] * i - 1)
                    i1 = random.randint(input_size, input_size + config['y_size'] * i - 1)  # input_size - 1 for indexing
                    i2 = random.randint(input_size, input_size + config['y_size'] * i - 1)

                else:
                    #print(input_size + config['y_size'] * (i - config['lookback']), input_size + config['y_size'] * i - 1)
                    i1 = random.randint(input_size + config['y_size'] * (i - config['lookback']), input_size + config['y_size'] * i - 1)  # input_size - 1 for indexing
                    i2 = random.randint(input_size + config['y_size'] * (i - config['lookback']), input_size + config['y_size'] * i - 1)

                op = random.randint(0, 3)  # 0: add, 1: sub, 2: mul, 3: div
                gate_index = (i * config['y_size'] + j) * 3
                self.chromozome[gate_index] = i1
                self.chromozome[gate_index + 1] = i2
                self.chromozome[gate_index + 2] = op

        self.chromozome[-1] = random.randint(input_size + (config['x_size'] - config['lookback']) * config['y_size'], self.outputValues.shape[1] - 2) # output can be any of the elements except itself and inputs, -2 for indexing


    def execute(self):
        for i in range(config['x_size']):
            for j in range(config['y_size']):
                for k in range(self.outputValues.shape[0]):
                    gate = (self.chromozome[(i * config['y_size'] + j) * 3], self.chromozome[(i * config['y_size'] + j) * 3 + 1], self.chromozome[(i * config['y_size'] + j) * 3 + 2])

                    if gate[2] == 0: # addition
                        self.outputValues[k][(i * config['y_size'] + j) + input_size] = self.outputValues[k][gate[0]] + self.outputValues[k][gate[1]]

                    elif gate[2] == 1: # subtraction
                        self.outputValues[k][(i * config['y_size'] + j) + input_size] = self.outputValues[k][gate[0]] - self.outputValues[k][gate[1]]

                    elif gate[2] == 2: # multiplication
                        self.outputValues[k][(i * config['y_size'] + j) + input_size] = self.outputValues[k][gate[0]] * self.outputValues[k][gate[1]]

                    elif gate[2] == 3: # division by zero
                        if self.outputValues[k][gate[1]] == 0:
                            self.outputValues[k][(i * config['y_size'] + j) + input_size] = 0
                        else:
                            self.outputValues[k][(i * config['y_size'] + j) + input_size] = self.outputValues[k][gate[0]] / self.outputValues[k][gate[1]]

        # get thresholded output values
        treshold = 0
        for k in range(self.outputValues.shape[0]):

            if self.outputValues[k][self.chromozome[-1]] > treshold:
                self.outputValues[k][-1] = 1
            else:
                self.outputValues[k][-1] = 0

        return self.outputValues[:,-1] # return vector of outputs



    def mutate(self):
        num_mutations = random.randint(0, config['mut_max'])
        for _ in range(num_mutations):
            idx = random.randint(0, (config['x_size'] * config['y_size']) * 3) # gates + output index

            if idx == (config['x_size'] * config['y_size']) * 3 : # output, last element
                self.chromozome[idx] = random.randint(input_size + (config['x_size'] - config['lookback']) * config['y_size'], self.outputValues.shape[1] - 2) 
                continue

            if idx % 3 == 2:    # operator
                self.chromozome[idx] = random.randint(0, 3)
            else:   # input
                col_idx = idx // (3 * config['y_size'])
                #print(col_idx)
                if col_idx - config['lookback'] < 0:
                    #print(0, input_size + config['y_size'] * (col_idx) - 1)
                    self.chromozome[idx] = random.randint(0, input_size + config['y_size'] * (col_idx) - 1)
                elif col_idx - config['lookback'] == 0: # lookback, cannot see input column
                    #print(input_size, input_size + config['y_size'] * (col_idx) - 1)
                    self.chromozome[idx] = random.randint(input_size, input_size + config['y_size'] * (col_idx) - 1)
                else:
                    #print(input_size + config['y_size'] * ((col_idx - 1) - config['lookback']), input_size + config['y_size'] * (col_idx) - 1)
                    self.chromozome[idx] = random.randint(input_size + config['y_size'] * ((col_idx) - config['lookback']), input_size + config['y_size'] * (col_idx) - 1)

    def copy(self):
        copied_individual = Individual(self.outputValues[:])  # Copying outputValues
        copied_individual.chromozome = self.chromozome[:]     # Copying chromozome
        return copied_individual

    def print_chromozome(self):
        for i in range(config['x_size']):
            for j in range(config['y_size']):
                print(f"({self.chromozome[(i * config['y_size'] + j) * 3]},{self.chromozome[(i * config['y_size'] + j) * 3 + 1]},{self.chromozome[(i * config['y_size'] + j) * 3 + 2]})", end='')
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
        outputValues = np.zeros((features.shape[0], config['x_size'] * config['y_size'] + 1)) # create output values for each individual
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

    def train(self):
        for g in tqdm(range(config['num_generations']), desc="Training"):
            """if g == config['num_generations'] // 2:
                config['mut_max'] //= 2""" # should i divide like in simulating annealing?

            fitness = self.evaluate()

            # get best individual
            best_fitness, best_individual_idx = min(fitness), fitness.index(min(fitness))
            best_individual = self.population[best_individual_idx]

            # Replace all individuals in the population with copies of the best individual
            self.population = [best_individual.copy() for _ in range(len(self.population))]

            # Mutate all individuals except the first (best) one
            for individual in self.population[1:]:
                individual.mutate()

            if STATISTICS:
                wandb.log({"best_fitness": best_fitness})
            else:
                print(f"Generation {g}, Best fitness: {best_fitness:.2f}")

        self.best_individual = self.population[0]


    def test(self, test_features, test_labels):
        test_outputValues = np.zeros((test_features.shape[0], config['x_size'] * config['y_size'] + 1))
        test_outputValues = np.concatenate((test_features, test_outputValues), axis=1)
        test_individual = Individual(test_outputValues)
        test_individual.chromozome = self.best_individual.chromozome
        test_result = test_individual.execute()
        test_accuracy = (test_labels == test_result).sum() / len(test_labels)
        return test_accuracy


def get_data(data, idx):
    dataOut = np.array([])  # Initialize as an empty array
    for i, d in enumerate(data):
        if i in idx:
            if dataOut.size == 0:  # Check if dataOut is empty
                dataOut = d
            else:
                dataOut = np.concatenate((dataOut, d), axis=0)
    return dataOut


def cross_validation(data, num_generations, pop_size, MUTATION_MAX, lookback, x_size, y_size):

    global config
    config = {
        "num_generations": num_generations,
        "pop_size": pop_size,
        "mut_max": MUTATION_MAX,
        "lookback": lookback,
        "x_size": x_size,
        "y_size": y_size
    }

    global input_size
    if STATISTICS:
        run_name = f"CGP_gens{config['num_generations']}_popSize{config['pop_size']}_MUT{config['mut_max']}_lookback{config['lookback']}_dims({config['x_size']},{config['y_size']})"
        wandb.init(project='BIN-CGP', entity='maxim-pl', name=run_name, config=config)
    test_accs = []
    kf = KFold(n_splits=10, shuffle=True)
    # K fold cross validation
    for foldID, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"---Fold {foldID + 1}---")
        train_data = get_data(data, train_index)
        test_data = get_data(data, test_index)
        train_features, train_labels = train_data[:, :-1], train_data[:, -1]
        test_features, test_labels = test_data[:, :-1], test_data[:, -1]
        input_size = train_features.shape[1]

        pop = Population(config['pop_size'], train_features, train_labels)
        pop.train()
        test_acc = pop.test(test_features, test_labels)
        test_accs.append(test_acc)
        if STATISTICS:
            wandb.log({"test_accuracy": test_acc})
        else:
            print(f"Test accuracy: {test_acc*100:.2f}%")

    if STATISTICS:
        wandb.log({"average_test_accuracy": sum(test_accs) / len(test_accs)})
        wandb.finish()