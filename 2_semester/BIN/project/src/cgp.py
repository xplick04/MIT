import os
import random

# CGP parameters
MUTATION_MAX = 5
# CGP dimensions
x_size = 3
y_size = 2
lookback = 1 # must be larger than 0, lookback 1 = can see previous column

input_size = 0


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

        self.chromozome[-1] = random.randint(input_size + (x_size - lookback) * y_size, len(self.outputValues) - 2) # output can be any of the elements except itself and inputs, -2 for indexing


    def execute(self):
        for i in range(x_size):
            for j in range(y_size):
                gate = (self.chromozome[(i * y_size + j) * 3], self.chromozome[(i * y_size + j) * 3 + 1], self.chromozome[(i * y_size + j) * 3 + 2])

                if gate[2] == 0: # addition
                    outputValues[(i * y_size + j) + input_size] = outputValues[gate[0]] + outputValues[gate[1]]

                elif gate[2] == 1: # subtraction
                    outputValues[(i * y_size + j) + input_size] = outputValues[gate[0]] - outputValues[gate[1]]

                elif gate[2] == 2: # multiplication
                    outputValues[(i * y_size + j) + input_size] = outputValues[gate[0]] * outputValues[gate[1]]

                elif gate[2] == 3: # division by zero
                    if outputValues[gate[1]] == 0:
                        outputValues[(i * y_size + j) + input_size] = 0
                    else:
                        outputValues[(i * y_size + j) + input_size] = outputValues[gate[0]] / outputValues[gate[1]]
            
        outputValues[-1] = outputValues[self.chromozome[-1]]


    
    def mutate(self):
        num_mutations = random.randint(0, MUTATION_MAX)
        num_mutations = 10
        for _ in range(num_mutations):
            idx = random.randint(0, (x_size * y_size) * 3) # gates + output index

            if idx == (x_size * y_size) * 3 : # output, last element
                self.chromozome[idx] = random.randint(input_size + (x_size - lookback) * y_size, len(self.outputValues) - 2) 
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


    def print_chromozome(self):
        for i in range(x_size):
            for j in range(y_size):
                print(f"{self.chromozome[(i * y_size + j) * 3]}, {self.chromozome[(i * y_size + j) * 3 + 1]}, {self.chromozome[(i * y_size + j) * 3 + 2]}")
        print(self.chromozome[-1])



if __name__ == "__main__":
    inputs = [0, 1, 2, 3]
    input_size = len(inputs)

    outputValues = inputs + [0 for i in range(x_size * y_size + 1)] # inputs + gene outputs + output
    print(outputValues)

    ind = Individual(outputValues) # each list is a column
    ind.print_chromozome()
    print("-----")
    ind.mutate()
    ind.print_chromozome()




