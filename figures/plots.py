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

def plot_metrics(metrics, k_values, dataset_name):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f'Test Performance on {dataset_name} dataset', fontsize=16)

    ax = axes.ravel()

    for key in metrics.keys():
        ax[0].plot(k_values, [metric[0] for metric in metrics[key]], label=key)
        ax[0].set(xticks=k_values, title='Silhouette Scores', xlabel='k', ylabel='score [-1, 1] (higher = better)')
        ax[0].legend(fontsize='xx-small')

        ax[1].plot(k_values, [metric[1] for metric in metrics[key]], label=key)
        ax[1].set(xticks=k_values, title='Davies Bouldin Scores', xlabel='k',
                  ylabel='score (lower = better, 0 is best)')
        ax[1].legend(fontsize='xx-small')

        ax[2].plot(k_values, [metric[2] for metric in metrics[key]], label=key)
        ax[2].set(xticks=k_values, title='Calinski Harabasz Scores', xlabel='k', ylabel='score (higher = better)')
        ax[2].legend(fontsize='xx-small')

        ax[3].plot(k_values, [metric[3] for metric in metrics[key]], label=key)
        ax[3].set(xticks=k_values, title='Adjusted Mutual Info Scores (uses actual labels)', xlabel='k',
                  ylabel='score [0, 1] (higher = better)')
        ax[3].legend(fontsize='xx-small')

    plt.tight_layout()
    plt.savefig('test_performance.png')
    plt.show()
