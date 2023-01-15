import matplotlib.pyplot as plt
import numpy as np

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

data = [[1,2,3,4], [2,4,6,8], [3,6,9,12]]
labels = ['seed1','seed2','seed3']

plot_with_std(data, labels)
