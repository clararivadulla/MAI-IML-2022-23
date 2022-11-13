import numpy as np
from matplotlib import pyplot as plt


colors = ["gray", "firebrick", "orange", "gold", "darkkhaki", "olive", "lightblue", "midnightblue", "thistle", "blue"]


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
    plt.savefig('plots/confusion_matrix.png', dpi=300)
    plt.show()


def scatter_plot_3D(labels, data, indices=(0, 1, 2), title=None, show_plot=False):
    num_clusters = len(np.unique(labels))

    ax = plt.axes(projection='3d')

    for k in range(num_clusters):
        ax.scatter3D(data[np.argwhere(labels == k)][:, :, indices[0]], data[np.argwhere(labels == k)][:, :, indices[1]],
                     data[np.argwhere(labels == k)][:, :, indices[2]], color=colors[k]);

    plt.title(title)
    file_name = title.replace(' ', '_').replace('\n', '_')
    plt.savefig(f"figures/scatter-plots/{file_name.lower()}_3D.png", dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.clf()

def scatter_plot(labels, data, indices=(0, 1), title=None, show_plot=False):

    num_clusters = len(np.unique(labels))

    for k in range(num_clusters):
        plt.scatter(data[np.argwhere(labels == k)][:, :, indices[0]], data[np.argwhere(labels == k)][:, :, indices[1]])

    plt.title(title)
    file_name = title.replace(' ', '_').replace('\n', '_')
    plt.savefig(f'figures/scatter-plots/{file_name.lower()}.png', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.clf()


def scatter_plot_data_only(data, indices=(0, 1), title=None, show_plot=False):

    plt.scatter(data[:, indices[0]], data[:, indices[1]])

    plt.title(title)
    file_name = title.replace(' ', '_').replace('\n', '_')
    plt.savefig(f'figures/scatter-plots/{file_name.lower()}.png', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.clf()


def scatter_plot_data_only_3D(data, indices=(0, 1), title=None, show_plot=False):

    ax = plt.axes(projection='3d')
    ax.scatter3D(data[:, indices[0]], data[:, indices[1]], data[:, indices[2]]);

    plt.title(title)
    file_name = title.replace(' ', '_').replace('\n', '_')
    plt.savefig(f'figures/scatter-plots/{file_name.lower()}_3D.png', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.clf()

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
    plt.savefig('plots/test_performance.png', dpi=300)
    plt.show()


def plot_metrics_p_or_m(algorithm, metrics, pm_values, dataset_name, p=True):

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f'{algorithm} performance on {dataset_name} dataset', fontsize=16)

    if p:
        xlabel = 'p'
        alg = 'k_harmonic_means'

    else:
        xlabel = 'm'
        alg = 'fuzzy_c_means'

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
    plt.savefig(f'plots/{alg}_{dataset_name}.png', dpi=300)
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
    plt.savefig(f'plots/test_agglomerative_{dataset_name}.png', dpi=300)
    plt.show()


def plot_meanShift(dataset_name, testMean_results, quantile_values):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Mean Shift clustering parameters with {dataset_name} dataset', fontsize=16)
    
    ax = axes.ravel()
    
    for key in testMean_results.keys():
        
        ax[0].plot(quantile_values, [metric[0] for metric in testMean_results[key]], marker='o', label=key)
        ax[0].set(xticks=quantile_values, title='Silhouette Scores', xlabel='quantile', ylabel='score [-1, 1] (higher = better)')
        ax[0].legend(fontsize='small')
        
        ax[1].plot(quantile_values, [metric[1] for metric in testMean_results[key]], marker='o', label=key)
        ax[1].set(xticks=quantile_values, title='Davies Bouldin Scores', xlabel='quantile', ylabel='score (lower = better, 0 is best)')
        ax[1].legend(fontsize='small')
        
        ax[2].plot(quantile_values, [metric[2] for metric in testMean_results[key]], marker='o', label=key)
        ax[2].set(xticks=quantile_values, title='Calinski Harabasz Scores', xlabel='quantile', ylabel='score (higher = better)')
        ax[2].legend(fontsize='small')
        
        ax[3].plot(quantile_values, [metric[3] for metric in testMean_results[key] ], marker='o', label=key)
        ax[3].set(xticks=quantile_values, title='Adjusted Mutual Info Scores (uses actual labels)', xlabel='quantile', ylabel='score [0, 1] (higher = better)')
        ax[3].legend(fontsize='small')
    
    plt.tight_layout()
    plt.savefig(f'plots/test_meanShift_{dataset_name}.png', dpi=300)
    plt.show()


def plot_clusters(k_values, all_labels, data, dataset_name, indices=(0, 1)):
    for key in all_labels.keys():
        i = 0
        for labels in all_labels[key]:
            scatter_plot(labels, data, indices, f'Clustering of {dataset_name} with {key} using {k_values[i]} clusters')
            i += 1
