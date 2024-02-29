def sort_numbers_from_file(file_path):
    with open(file_path, 'r') as file:
        input_line = file.readline().strip()

    # Split input line into numbers
    numbers = [int(x) for x in input_line.split()]

    # Sort the numbers
    sorted_numbers = sorted(numbers)

    # Write sorted numbers to stdout
    for num in sorted_numbers:
        print(num)

if __name__ == "__main__":
    sort_numbers_from_file('./input')