import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
def plot_with_std(data, labels):
    # Plot the data
    for i, d in enumerate(data):
        plt.plot(d, label=labels[i])

    # Compute the standard deviation
    std = np.std(data, axis=0)

    # Plot the standard deviation transparently behind the run lines
    plt.fill_between(range(len(std)), np.min(data, axis=0)-std, np.max(data, axis=0)+std, alpha=0.2) # alpha is the transparency
    plt.legend()

    plt.savefig(f'./plots/plot_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
    plt.show()
def read_csv_to_df(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # Return the first row 
    return df # Run 0 Wake loss
# Plot each run against each other 
def plot_dataframe(df):
    labels = [f'run {i}' for i in range(len(df))]
    plot_with_std(df.values.tolist(), labels)
    
def main():
    file_path = 'log.csv'
    df = pd.read_csv(file_path)
    columns = df.columns
    # For each row in each column the DataFrame transform the string into a list of floats
    for column in columns:
        df[column] = df[column].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
        plot_dataframe(df[column])

# Run the main function
if __name__ == '__main__':
    main()