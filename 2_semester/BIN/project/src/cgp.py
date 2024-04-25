import os
import random


# CGP outputs
outputValues = []

# CGP parameters
MUTATION_MAX = 5
# CGP dimensions
x_size = 2
y_size = 3

input_size = 0

class Gate:
    def __init__(self, x1, x2, op):
        self.x1 = x1
        self.x2 = x2
        self.op = op

    def execute(self):
        i1 = outputValues[self.x1]
        i2 = outputValues[self.x2]
        if self.op == 0:
            return i1 + i2
        elif self.op == 1:
            return i1 - i2
        elif self.op == 2:
            return i1 * i2
        elif self.op == 3:
            if i2 == 0:
                return 0
            return i1 / i2
        
    def __repr__(self):
        return f"{self.x1, self.x2, self.op}"



class Individual:
    x_size = 0
    y_size = 0
    num_inputs = 0
    num_outputs = 0
    gates = []
    lookback = 0
    outputElem = 0


    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        # random genes initialization
        self.gates = [[Gate(0, 0, 0) for _ in range(y_size)] for _ in range(x_size)]
        for i in range(len(self.gates)):
            for j in range(len(self.gates[i])):
                col_len = len(self.gates[0])
                op = random.randint(0, 3)
                i1 = random.randint(0, (input_size-1) + col_len*i) # input_size - 1 for indexing
                i2 = random.randint(0, (input_size-1) + col_len*i) # each column can have inputs from previous columns
                self.gates[i][j] = Gate(i1, i2, op)
        self.outputElem = random.randint((input_size),(input_size) + (x_size-1) * y_size) # output can be any of the elements except inputs

    def fitness(self):
        pass

    def execute(self):
        for i in range(self.x_size):
            for j in range(self.y_size):
                outputValues[(input_size) + (i * self.y_size) + j] = self.gates[i][j].execute()

        return outputValues[self.outputElem]

    def mutate(self):
        num_mutations = random.randint(1, MUTATION_MAX)
        for _ in range(num_mutations):
            rand = random.randint(0, x_size * y_size * 3) # choose a random gene or output element
            if rand < x_size * y_size * 3:
                col_len = len(self.gates[0])
                i = rand // (3 * col_len)
                j = rand // 3
                if j >= col_len:
                    j -= col_len
                
                if rand % 3 == 0:
                    self.gates[i][j].x1 = random.randint(0, (input_size-1) + col_len*i)
                elif rand % 3 == 1:
                    self.gates[i][j].x2 = random.randint(0, (input_size-1) + col_len*i)
                else:
                    self.gates[i][j].op = random.randint(0, 3)
            else:
                self.outputElem = random.randint((input_size),(input_size) + (x_size-1) * y_size)








if __name__ == "__main__":
    inputs = [1, 2, 3, 4]
    input_size = len(inputs)

    outputValues = inputs + ([0 for _ in range(x_size * y_size)]) # inputs + gene outputs
    print(outputValues)

    ind = Individual(x_size, y_size) # each list is a column
    print(ind.execute())
    print(ind.gates, ind.outputElem)
    ind.mutate()
    print(ind.execute())
    print(ind.gates, ind.outputElem)




