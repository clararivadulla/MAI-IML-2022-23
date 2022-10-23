import numpy as np
from matplotlib import pyplot as plt

colors = ["gray", "firebrick", "orange", "gold", "darkkhaki", "olive", "lightblue", "midnightblue", "thistle",
          "blue"]

def confusion_matrix_plot(data, dataset_name):
    k_test = len(data)
    if k_test%2 == 0:
        rows = k_test//2
    else:
        rows = k_test//2+1

    fig, axes = plt.subplots(rows, 2, figsize=(10, 7))
    fig.suptitle(f'Confusion matrix of {dataset_name} dataset', fontsize=16)
    ax = axes.ravel()

    for k in range(k_test):
        ax[k].table(np.array(data[k]), loc='center', rowLabels=data[k].index, colLabels=data[k].columns)
        ax[k].axis('off')
        ax[k].set(xlabel=f'k = {data[k].shape[1]}')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

def scatter_plot(labels, data, indices=(0, 1), title=None):

    num_clusters = len(np.unique(labels))

    for k in range(num_clusters):
        plt.scatter(data[np.argwhere(labels == k)][:, :, indices[0]], data[np.argwhere(labels == k)][:, :, indices[1]],
                    color=colors[k])

    plt.title(title)

    plt.savefig(f'scatterplots/{title}.png', dpi=300)
    plt.show()


def plot_metrics(metrics, k_values, dataset_name, x_label='k'):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f'Test Performance on {dataset_name} dataset', fontsize=16)

    ax = axes.ravel()

    for key in metrics.keys():
        ax[0].plot(k_values, [metric[0] for metric in metrics[key]], label=key)
        ax[0].set(xticks=k_values, title='Silhouette Scores', xlabel=x_label, ylabel='score [-1, 1] (higher = better)')
        ax[0].legend(fontsize='xx-small')

        ax[1].plot(k_values, [metric[1] for metric in metrics[key]], label=key)
        ax[1].set(xticks=k_values, title='Davies Bouldin Scores', xlabel=x_label,
                  ylabel='score (lower = better, 0 is best)')
        ax[1].legend(fontsize='xx-small')

        ax[2].plot(k_values, [metric[2] for metric in metrics[key]], label=key)
        ax[2].set(xticks=k_values, title='Calinski Harabasz Scores', xlabel=x_label, ylabel='score (higher = better)')
        ax[2].legend(fontsize='xx-small')

        ax[3].plot(k_values, [metric[3] for metric in metrics[key]], label=key)
        ax[3].set(xticks=k_values, title='Adjusted Mutual Info Scores (uses actual labels)', xlabel=x_label,
                  ylabel='score [0, 1] (higher = better)')
        ax[3].legend(fontsize='xx-small')

    plt.tight_layout()
    plt.savefig('test_performance.png', dpi=300)
    plt.show()

def plot_metrics_p_or_m(algorithm, metrics, pm_values, dataset_name, p=True):

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f'{algorithm} performance on {dataset_name} dataset', fontsize=16)

    if p:
        xlabel = 'p'
        folder = 'k_harmonic_means'

    else:
        xlabel = 'm'
        folder = 'fuzzy_c_means'

    ax = axes.ravel()
    print(metrics)

    for key in metrics.keys():
        ax[0].plot(pm_values, [metric[0] for metric in metrics[key]], label='k = '+ key)
        ax[0].set(xticks=pm_values, title='Silhouette Scores', xlabel=xlabel, ylabel='score [-1, 1] (higher = better)')
        ax[0].legend(fontsize='xx-small')

        ax[1].plot(pm_values, [metric[1] for metric in metrics[key]], label='k = '+ key)
        ax[1].set(xticks=pm_values, title='Davies Bouldin Scores', xlabel=xlabel,
                  ylabel='score (lower = better, 0 is best)')
        ax[1].legend(fontsize='xx-small')

        ax[2].plot(pm_values, [metric[2] for metric in metrics[key]], label='k = '+ key)
        ax[2].set(xticks=pm_values, title='Calinski Harabasz Scores', xlabel=xlabel, ylabel='score (higher = better)')
        ax[2].legend(fontsize='xx-small')

        ax[3].plot(pm_values, [metric[3] for metric in metrics[key]], label='k = '+ key)
        ax[3].set(xticks=pm_values, title='Adjusted Mutual Info Scores (uses actual labels)', xlabel=xlabel,
                  ylabel='score [0, 1] (higher = better)')
        ax[3].legend(fontsize='xx-small')

    plt.tight_layout()
    plt.savefig(f'{folder}.png', dpi=300)
    plt.show()

def plot_agglomerative(dataset_name, testAgg_results, k_values):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Agglomerative clustering parameters with {dataset_name} dataset', fontsize=16)
    
    ax = axes.ravel()
    
    for key in testAgg_results.keys():
        ax[0].plot(k_values, [metric[0] for metric in testAgg_results[key]], marker='o', label=key)
        ax[0].set(xticks=k_values, title='Silhouette Scores', xlabel='k', ylabel='score [-1, 1] (higher = better)')
        ax[0].legend(fontsize='x-small')
        
        ax[1].plot(k_values, [metric[1] for metric in testAgg_results[key]], marker='o', label=key)
        ax[1].set(xticks=k_values, title='Davies Bouldin Scores', xlabel='k',
                  ylabel='score (lower = better, 0 is best)')
        ax[1].legend(fontsize='xx-small')
                  
        ax[2].plot(k_values, [metric[2] for metric in testAgg_results[key]], marker='o', label=key)
        ax[2].set(xticks=k_values, title='Calinski Harabasz Scores', xlabel='k', ylabel='score (higher = better)')
        ax[2].legend(fontsize='xx-small')
                  
        ax[3].plot(k_values, [metric[3] for metric in testAgg_results[key]], marker='o', label=key)
        ax[3].set(xticks=k_values, title='Adjusted Mutual Info Scores (uses actual labels)', xlabel='k',
                            ylabel='score [0, 1] (higher = better)')
        ax[3].legend(fontsize='xx-small')

    plt.tight_layout()
    plt.savefig(f'test_agglomerative_{dataset_name}.pdf', dpi=300)
    plt.show()


def plot_clusters(k_values, all_labels, data, dataset_name, indices=(0, 1)):
    for key in all_labels.keys():
        i = 0
        for labels in all_labels[key]:
            scatter_plot(labels, data, indices, f'Clustering of {dataset_name} with {key} using {k_values[i]} clusters')
            i += 1