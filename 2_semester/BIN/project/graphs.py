import pandas as pd
import matplotlib.pyplot as plt

FONT_SIZE = 12

def grid_boxplots(csv_file, num_cols=3):
    # Load data from CSV file
    df = pd.read_csv(csv_file)

    # Drop the first column
    df = df.drop(columns=[df.columns[0]])

    # Filter column names to keep only those without "__MIN" or "__MAX"
    df = df[[col for col in df.columns if "__MIN" not in col and "__MAX" not in col]]
    
    # Selecting only the desired columns
    selected_columns = df[["CGP_gens100_popSize50_MUT10_lookback2_dims(5,5) - test_accuracy", 
                        "CGP_gens100_popSize50_MUT10_lookback2_dims(7,7) - test_accuracy",
                        "CGP_gens100_popSize50_MUT10_lookback2_dims(10,10) - test_accuracy",]]
    
    # Renaming the columns
    selected_columns = selected_columns.rename(columns={
        "CGP_gens100_popSize50_MUT10_lookback2_dims(5,5) - test_accuracy": "5x5",
        "CGP_gens100_popSize50_MUT10_lookback2_dims(7,7) - test_accuracy": "7x7",
        "CGP_gens100_popSize50_MUT10_lookback2_dims(10,10) - test_accuracy": "10x10",
    })

    # Reassigning the DataFrame with only the selected columns
    df = selected_columns

    # Initialize an empty list to store boxplot data for each column
    boxplot_data = []

    # Iterate over each column
    for column in df.columns:
        # Drop rows with NaN values in the current column and append to boxplot_data
        column_data = df[[column]].dropna()
        boxplot_data.append(column_data.values.flatten())

    # Plotting all boxplots in one graph
    plt.figure(figsize=(8, 8))

    bp = plt.boxplot(boxplot_data, patch_artist=True, showmeans=True, meanline=True, medianprops=dict(color='black', linewidth=1), widths=0.58)

    # Add legends for mean and median with matching colors
    plt.legend(['Mean', 'Median'], 
            handles=[bp['means'][0], bp['medians'][0]], 
            labels=['Mean', 'Median'])

    # Customize boxplot elements
    plt.title('Boxplot for different grid sizes', fontweight='bold', fontsize=FONT_SIZE)
    plt.ylabel('Accuracy (%)',fontweight='bold', fontsize=FONT_SIZE)

    # Add a line from median to left axis with exact number for each boxplot
    for i, column_data in enumerate(boxplot_data):
        median_val = df.iloc[:, i].median()
        plt.text(i + 1.32, median_val, f'{median_val:.2f}', ha='left', va='center', color='black', fontsize=FONT_SIZE)

    # Customize boxplot colors
    colors = ['lightblue'] * len(df.columns)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Customize mean line
    plt.setp(bp['means'], color='green', linewidth=1)

    # Set xticks and labels
    plt.xticks(range(1, len(df.columns) + 1), df.columns, fontweight='bold', fontsize=FONT_SIZE)

    plt.tight_layout()
    plt.show()

def gen_boxplots(csv_file, num_cols=3):
    # Load data from CSV file
    df = pd.read_csv(csv_file)

    # Drop the first column
    df = df.drop(columns=[df.columns[0]])

    # Filter column names to keep only those without "__MIN" or "__MAX"
    df = df[[col for col in df.columns if "__MIN" not in col and "__MAX" not in col]]
    
    # Selecting only the desired columns
    selected_columns = df[["CGP_gens1_popSize500_MUT10_lookback2_dims(5,5) - test_accuracy", 
                        "CGP_gens10_popSize50_MUT10_lookback2_dims(5,5) - test_accuracy",
                        "CGP_gens100_popSize5_MUT10_lookback2_dims(5,5) - test_accuracy",]]
    
    # Renaming the columns
    selected_columns = selected_columns.rename(columns={
        "CGP_gens1_popSize500_MUT10_lookback2_dims(5,5) - test_accuracy": "Gens 1, Pop 500",
        "CGP_gens10_popSize50_MUT10_lookback2_dims(5,5) - test_accuracy": "Gens 10, Pop 50",
        "CGP_gens100_popSize5_MUT10_lookback2_dims(5,5) - test_accuracy": "Gens 100, Pop 5",
    })

    # Reassigning the DataFrame with only the selected columns
    df = selected_columns

    # Initialize an empty list to store boxplot data for each column
    boxplot_data = []

    # Iterate over each column
    for column in df.columns:
        # Drop rows with NaN values in the current column and append to boxplot_data
        column_data = df[[column]].dropna()
        boxplot_data.append(column_data.values.flatten())

    # Plotting all boxplots in one graph
    plt.figure(figsize=(8, 8))

    bp = plt.boxplot(boxplot_data, patch_artist=True, showmeans=True, meanline=True, medianprops=dict(color='black', linewidth=1), widths=0.58)

    # Customize boxplot elements
    plt.title('Boxplot for different number of generations', fontweight='bold', fontsize=FONT_SIZE)
    plt.ylabel('Accuracy (%)',fontweight='bold', fontsize=FONT_SIZE)

    # Add a line from median to left axis with exact number for each boxplot
    for i, column_data in enumerate(boxplot_data):
        median_val = df.iloc[:, i].median()
        plt.text(i + 1.32, median_val, f'{median_val:.2f}', ha='left', va='center', color='black', fontsize=FONT_SIZE)

    # Customize boxplot colors
    colors = ['lightblue'] * len(df.columns)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Customize mean line
    plt.setp(bp['means'], color='green', linewidth=1)

    # Set xticks and labels
    plt.xticks(range(1, len(df.columns) + 1), df.columns, fontweight='bold', fontsize=FONT_SIZE)

    plt.tight_layout()
    plt.show()
    

def mut_boxplots(csv_file, num_cols=3):
    # Load data from CSV file
    df = pd.read_csv(csv_file)

    # Drop the first column
    df = df.drop(columns=[df.columns[0]])

    # Filter column names to keep only those without "__MIN" or "__MAX"
    df = df[[col for col in df.columns if "__MIN" not in col and "__MAX" not in col]]
    
    # Selecting only the desired columns
    selected_columns = df[["CGP_gens100_popSize50_MUT5_lookback2_dims(5,5) - test_accuracy", 
                        "CGP_gens100_popSize50_MUT10_lookback2_dims(5,5) - test_accuracy",
                        "CGP_gens100_popSize50_MUT20_lookback2_dims(5,5) - test_accuracy",]]
    
    # Renaming the columns
    selected_columns = selected_columns.rename(columns={
        "CGP_gens100_popSize50_MUT5_lookback2_dims(5,5) - test_accuracy": "Mutations 5",
        "CGP_gens100_popSize50_MUT10_lookback2_dims(5,5) - test_accuracy": "Mutations 10",
        "CGP_gens100_popSize50_MUT20_lookback2_dims(5,5) - test_accuracy": "Mutations 20",
    })

    # Reassigning the DataFrame with only the selected columns
    df = selected_columns

    # Initialize an empty list to store boxplot data for each column
    boxplot_data = []

    # Iterate over each column
    for column in df.columns:
        # Drop rows with NaN values in the current column and append to boxplot_data
        column_data = df[[column]].dropna()
        boxplot_data.append(column_data.values.flatten())

    # Plotting all boxplots in one graph
    plt.figure(figsize=(8, 8))

    bp = plt.boxplot(boxplot_data, patch_artist=True, showmeans=True, meanline=True, medianprops=dict(color='black', linewidth=1), widths=0.58)

    # Customize boxplot elements
    plt.title('Boxplot for different number of mutations', fontweight='bold', fontsize=FONT_SIZE)
    plt.ylabel('Accuracy (%)',fontweight='bold')

    # Add a line from median to left axis with exact number for each boxplot
    for i, column_data in enumerate(boxplot_data):
        median_val = df.iloc[:, i].median()
        plt.text(i + 1.32, median_val, f'{median_val:.2f}', ha='left', va='center', color='black', fontsize=FONT_SIZE)

    # Customize boxplot colors
    colors = ['lightblue'] * len(df.columns)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Customize mean line
    plt.setp(bp['means'], color='green', linewidth=1)

    # Set xticks and labels
    plt.xticks(range(1, len(df.columns) + 1), df.columns, fontweight='bold', fontsize=FONT_SIZE)

    plt.tight_layout()
    plt.show()


def lb_boxplots(csv_file, num_cols=3):
    # Load data from CSV file
    df = pd.read_csv(csv_file)

    # Drop the first column
    df = df.drop(columns=[df.columns[0]])

    # Filter column names to keep only those without "__MIN" or "__MAX"
    df = df[[col for col in df.columns if "__MIN" not in col and "__MAX" not in col]]
    
    # Selecting only the desired columns
    selected_columns = df[["CGP_gens100_popSize50_MUT10_lookback1_dims(5,5) - test_accuracy", 
                        "CGP_gens100_popSize50_MUT10_lookback2_dims(5,5) - test_accuracy",
                        "CGP_gens100_popSize50_MUT10_lookback3_dims(5,5) - test_accuracy",]]
    
    # Renaming the columns
    selected_columns = selected_columns.rename(columns={
        "CGP_gens100_popSize50_MUT10_lookback1_dims(5,5) - test_accuracy": "Lookback 1",
        "CGP_gens100_popSize50_MUT10_lookback2_dims(5,5) - test_accuracy": "Lookback 2",
        "CGP_gens100_popSize50_MUT10_lookback3_dims(5,5) - test_accuracy": "Lookback 3",
    })

    # Reassigning the DataFrame with only the selected columns
    df = selected_columns

    # Initialize an empty list to store boxplot data for each column
    boxplot_data = []

    # Iterate over each column
    for column in df.columns:
        # Drop rows with NaN values in the current column and append to boxplot_data
        column_data = df[[column]].dropna()
        boxplot_data.append(column_data.values.flatten())

    # Plotting all boxplots in one graph
    plt.figure(figsize=(8, 8))

    bp = plt.boxplot(boxplot_data, patch_artist=True, showmeans=True, meanline=True, medianprops=dict(color='black', linewidth=1), widths=0.58)

    # Customize boxplot elements
    plt.title('Boxplot for different lookback', fontweight='bold', fontsize=FONT_SIZE)
    plt.ylabel('Accuracy (%)',fontweight='bold', fontsize=FONT_SIZE)

    # Add a line from median to left axis with exact number for each boxplot
    for i, column_data in enumerate(boxplot_data):
        median_val = df.iloc[:, i].median()
        plt.text(i + 1.32, median_val, f'{median_val:.2f}', ha='left', va='center', color='black', fontsize=FONT_SIZE)

    # Customize boxplot colors
    colors = ['lightblue'] * len(df.columns)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Customize mean line
    plt.setp(bp['means'], color='green', linewidth=1)

    # Set xticks and labels
    plt.xticks(range(1, len(df.columns) + 1), df.columns, fontweight='bold', fontsize=FONT_SIZE)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_file = "./statistics/test_acc.csv"  # Change this to your CSV file path
    #grid_boxplots(csv_file)
    #gen_boxplots(csv_file)
    #mut_boxplots(csv_file)
    lb_boxplots(csv_file)