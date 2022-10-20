import numpy as np
from matplotlib import pyplot as plt

colors = ["darkgray", "firebrick", "orange", "gold", "darkkhaki", "olive", "lightblue", "midnightblue", "thistle",
          "blue"]


def scatterplot(labels, data, indices, title=None, x_label=None, y_label=None):
    num_clusters = len(np.unique(labels))

    for k in range(num_clusters):
        plt.scatter(data[np.argwhere(labels == k)][:, :, indices[0]], data[np.argwhere(labels == k)][:, :, indices[1]],
                    color=colors[k])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
