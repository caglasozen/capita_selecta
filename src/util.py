import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

##Visualization Functions

def visualize_drift(acc, drift, warn, nr_of_batches, len_dataset):
    fig, ax = plt.subplots()
    x = np.linspace(nr_of_batches, len_dataset*nr_of_batches, num = nr_of_batches - 1)
    ax.set_xlabel('index')
    ax.set_ylabel('accuracy')

    plt.vlines(x=[value*len_dataset for value in warn], ymin=0, ymax=1, colors='g', linestyles=':', label='warnings')
    plt.vlines(x=[value*len_dataset for value in drift], ymin=0, ymax=1, colors='r', linestyles='-', label='drifts')
    ax.plot(x, acc, lw=2, label='accuracy')

    ax.legend()
    plt.title("Hyperplane - High (10%) Gradual Drift")
    plt.show()
    return fig


def visualize_magnitude(magn):
    fig, ax = plt.subplots()
    ax.plot(magn, label = "drift magnitude")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Magnitude")

    ax.legend()
    plt.title("Hyperplane - High (10%) Gradual Drift")
    plt.show()
    return fig

#Gradual Drift Utility Functions

def prepare_data(org_batch, pair):
    combined = []
    for i in range(pair[0], pair[1]+1):
        combined.append(org_batch[i])
    return pd.concat(combined)

def get_drift_pairs(drift_list, org_batch_size):
     drift_pairs_size = []
     drift_pairs = []
     for i in range(len(drift_list)-1):
         batch_size = drift_list[i+1]-drift_list[i-1] + 1  * org_batch_size
         drift_pairs_size.append(batch_size)
         drift_pairs.append((drift_list[i], drift_list[i+1]))
     return drift_pairs_size, drift_pairs