# 
# ARGUMENTS: NUM_SAMPLES, NUM_PARAMETERS, SEED -> python3 test-part2.py 150 10 250
#

import csv
import subprocess
import random
import sys
from pathlib import Path

def generate_random_data(filename, num_samples, num_params, seed):
    random.seed(seed)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for _ in range(num_samples):
            params = [round(random.uniform(0, 150), 3) for _ in range(num_params)]
            label = random.choice(['TridaA', 'TridaB', 'TridaC', 'TridaD'])
            writer.writerow(params + [label])

def remove_last_column(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            writer.writerow(row[:-1])

def train_classifier(input_file, output_file):
    with open(output_file, 'w') as f:
        subprocess.run(['./flp-fun', '-2', input_file], stdout=f)

def classify_data(model_file, input_data, output_file):
    with open(output_file, 'w') as f:
        subprocess.run(['./flp-fun', '-1', model_file, input_data], stdout=f)

def compare_results(test_data_file, classification_output_file):
    with open(test_data_file, 'r') as test_file, open(classification_output_file, 'r') as output_file:
        test_reader = csv.reader(test_file)
        output_reader = csv.reader(output_file)       

        for test_row, output_row in zip(test_reader, output_reader):
            expected_label = test_row[-1]
            classified_label = output_row[0]
            
            if expected_label == classified_label:
                print("Classified correctly:", test_row)
            else:
                print("Misclassification detected:", test_row, "(Expected:", expected_label, "Classified:", classified_label, ")")

test_dir = "test-2"
Path(test_dir).mkdir(exist_ok=True)

data_filename = f'{test_dir}/data.csv'
generate_random_data(data_filename, int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])) 

test_filename = f'{test_dir}/test_data.csv'
remove_last_column(data_filename, test_filename)

training_output_file = f'{test_dir}/output_tree.txt'
train_classifier(data_filename, training_output_file)

classification_output_file = f'{test_dir}/classification_output.txt'
classify_data(training_output_file, test_filename, classification_output_file)

compare_results(data_filename, classification_output_file)