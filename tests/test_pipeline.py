


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def plot_with_std(data, labels):
    # Plot the data
    for i, d in enumerate(data):
        plt.plot(d, label=labels[i])

    # Compute the standard deviation
    std = np.std(data, axis=0)

    # Plot the standard deviation transparently behind the run lines
    plt.fill_between(range(len(std)), np.min(data, axis=0)-std, np.max(data, axis=0)+std, alpha=0.2) # alpha is the transparency
    plt.legend()
    plt.show()



# Take each row of a pandas dataframe and plot it take Wake loss column name and print the value
def plot_dataframe(df, column, labels):
    # Get the data from the dataframe
    data = df[column].values
    data = [float(x.strip('[]')) for x in data]
    #print(data)
    # Plot the data
    for i, d in enumerate(data):
        plt.plot(d, label=labels[i])

    # Compute the standard deviation
    std = np.std(data, axis=0)

    # Plot the standard deviation transparently behind the run lines
    plt.fill_between(range(len(std)), np.min(data, axis=0)-std, np.max(data, axis=0)+std, alpha=0.2) # alpha is the transparency
    plt.legend()
    plt.show()
    
# Read dataframe from log.csv
df = pd.read_csv('log.csv')
# Create a nested list with each index being a row of the column Wake accuracy
wake_accuracy_list = []
for i, row in df.iterrows():
    wake_accuracy_list.append(row['Entropy'])
print(wake_accuracy_list)
wake_accuracy_list = [float(x.strip('[]')) for x in wake_accuracy_list]
print(wake_accuracy_list[0])
column = 'Wake accuracy'
labels = ['run1','run2','run3','run4','run5','run6']

plot_dataframe(df, column, labels)
