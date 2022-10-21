import numpy as np
from matplotlib import pyplot as plt

colors = ["gray", "firebrick", "orange", "gold", "darkkhaki", "olive", "lightblue", "midnightblue", "thistle",
          "blue"]


def scatter_plot(labels, data, indices=(0, 1), title=None, x_label=None, y_label=None):

    num_clusters = len(np.unique(labels))

    for k in range(num_clusters):
        plt.scatter(data[np.argwhere(labels == k)][:, :, indices[0]], data[np.argwhere(labels == k)][:, :, indices[1]],
                    color=colors[k])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()