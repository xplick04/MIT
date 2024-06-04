import os
import random
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import scikitplot as skplt
import networkx as nx


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
                #for k in range(self.outputValues.shape[0]):
                gate = (self.chromozome[(i * config['y_size'] + j) * 3], self.chromozome[(i * config['y_size'] + j) * 3 + 1], self.chromozome[(i * config['y_size'] + j) * 3 + 2])

                if gate[2] == 0: # addition
                    self.outputValues[:, (i * config['y_size'] + j) + input_size] = self.outputValues[:, gate[0]] + self.outputValues[:, gate[1]]

                elif gate[2] == 1: # subtraction
                    self.outputValues[:, (i * config['y_size'] + j) + input_size] = self.outputValues[:, gate[0]] - self.outputValues[:, gate[1]]

                elif gate[2] == 2: # multiplication
                    self.outputValues[:, (i * config['y_size'] + j) + input_size] = self.outputValues[:, gate[0]] * self.outputValues[:, gate[1]]

                elif gate[2] == 3: # division 
                    # Define a small constant
                    small_constant = 0.0001
                    # Find the indexes where the value is 0
                    zero_indexes = np.where(self.outputValues[:, gate[1]] == 0)
                    # Replace the values at those indexes with the small constant
                    self.outputValues[zero_indexes, gate[1]] = small_constant
                    self.outputValues[:, (i * config['y_size'] + j) + input_size] = self.outputValues[:, gate[0]] / self.outputValues[:, gate[1]]

        # Apply sigmoid activation function to the output
        self.outputValues[:, -1] = self.sigmoid()
        return self.outputValues[:, -1]  # return vector of outputs
    

    def sigmoid(self):
        larger_numbers = np.where(self.outputValues[:, self.chromozome[-1]] > 50)
        self.outputValues[larger_numbers, self.chromozome[-1]] = 50
        smaller_numbers = np.where(self.outputValues[:, self.chromozome[-1]] < -50)
        self.outputValues[smaller_numbers, self.chromozome[-1]] = -50
        return 1 / (1 + np.exp(-self.outputValues[:, self.chromozome[-1]]))


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
                if col_idx - config['lookback'] < 0:
                    self.chromozome[idx] = random.randint(0, input_size + config['y_size'] * (col_idx) - 1)
                elif col_idx - config['lookback'] == 0: # lookback, cannot see input column
                    self.chromozome[idx] = random.randint(input_size, input_size + config['y_size'] * (col_idx) - 1)
                else:
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

    def plot_chromozome(self):
        G = nx.DiGraph()

        x_size = config['x_size']
        y_size = config['y_size']

        for i in range(x_size):
            for j in range(y_size):
                gate = (self.chromozome[(i * y_size + j) * 3], self.chromozome[(i * y_size + j) * 3 + 1], self.chromozome[(i * y_size + j) * 3 + 2])

                node = f"({gate[0]},{gate[1]},{gate[2]})"
                G.add_node((i,j), label=node, layer=i, subset=i)

                if i != 0:
                    if gate[0] > (input_size - 1) and gate[1] > (input_size - 1): # do not connect to input nodes
                        node_i = (gate[0] - input_size) // y_size # first edge
                        node_j = (gate[0] - input_size) % y_size
                        G.add_edge((node_i, node_j), (i, j))

                        node_i = (gate[1] - input_size) // y_size # second edge
                        node_j = (gate[1] - input_size) % y_size
                        G.add_edge((node_i, node_j), (i, j))

        G.add_node((x_size, 0), label=f"{self.chromozome[-1]}", layer=x_size, subset=x_size) # output node
        node_i = (self.chromozome[-1] - input_size) // y_size
        node_j = (self.chromozome[-1] - input_size) % y_size
        G.add_edge((node_i, node_j), (x_size, 0))

        pos = nx.multipartite_layout(G, subset_key="layer")
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color='lightblue', font_size=8)
        plt.show()


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
        for g in range(config['num_generations']):

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
        return self.best_individual


    def test(self, test_features, test_labels):
        test_outputValues = np.zeros((test_features.shape[0], config['x_size'] * config['y_size'] + 1))
        test_outputValues = np.concatenate((test_features, test_outputValues), axis=1)
        test_individual = Individual(test_outputValues)
        test_individual.chromozome = self.best_individual.chromozome
        test_result = test_individual.execute()
        test_result = np.where(test_result >= 0.5, 1, 0) # thresholding
        test_accuracy = (test_labels == test_result).sum() / len(test_labels)
        test_sensitivity = (test_labels[test_labels == 1] == test_result[test_labels == 1]).sum() / len(test_labels[test_labels == 1])
        test_specificity = (test_labels[test_labels == 0] == test_result[test_labels == 0]).sum() / len(test_labels[test_labels == 0])

        return test_accuracy, test_sensitivity, test_specificity




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

    best_individual = None
    best_test_acc = 0
    test_accs = []
    test_sensis = []
    test_specis = []
    # cross validation, 200 runs on random splits(8:2) to aproximate the average test accuracy, 200 runs seemed to be a good sample size since after 200 runs the average test acc repeated

    number_of_runs = 10
    for n in tqdm(range(number_of_runs), desc="Cross validation runs"):
        train_index = np.random.randint(0, len(data), int(0.8 * len(data)))
        test_index = np.setdiff1d(np.arange(len(data)), train_index)

        train_data = get_data(data, train_index)
        test_data = get_data(data, test_index)
        train_features, train_labels = train_data[:, :-1], train_data[:, -1]
        test_features, test_labels = test_data[:, :-1], test_data[:, -1]
        input_size = train_features.shape[1]

        pop = Population(config['pop_size'], train_features, train_labels)
        ind = pop.train()
        test_acc, test_sensi, test_speci = pop.test(test_features, test_labels)
        test_accs.append(test_acc)
        test_sensis.append(test_sensi)
        test_specis.append(test_speci)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_individual = ind

        if STATISTICS:
            wandb.log({"test_accuracy": test_acc*100})
        else:
            print(f"Test accuracy: {test_acc*100:.2f}%")


    if STATISTICS:
        wandb.log({"average_test_accuracy": sum(test_accs) / len(test_accs) * 100})
        wandb.finish()
    else:
        print(f"Average test accuracy: {sum(test_accs) / len(test_accs) * 100:.2f}%")
        best_individual.plot_chromozome()

    return test_accs, test_sensis, test_specis
