import numpy as np
from matplotlib import pyplot as plt

colors = ["pink", "firebrick", "orange", "lightblue", "yellowgreen"]

def scatterplot(labels, data, indices):
    num_clusters = len(np.unique(labels))
    for k in range(num_clusters):
        print(k)
        plt.scatter(data[np.argwhere(labels == k)][:, :, indices[0]], data[np.argwhere(labels == k)][:, :, indices[1]], color=colors[k])
    plt.show()
