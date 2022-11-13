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


def scatter_plot(labels, data, indices=(0, 1, 2), title=None, show_plot=False, plot_3D=False):

    num_clusters = len(np.unique(labels))

    if plot_3D:
        ax = plt.axes(projection='3d')

    for k in range(num_clusters):

        if plot_3D:
            ax.scatter3D(data[np.argwhere(labels == k)][:, :, indices[0]],
                         data[np.argwhere(labels == k)][:, :, indices[1]],
                         data[np.argwhere(labels == k)][:, :, indices[2]])

        else:
            plt.scatter(data[np.argwhere(labels == k)][:, :, indices[0]], data[np.argwhere(labels == k)][:, :, indices[1]])

    plt.title(title)
    file_name = title.replace(' ', '_').replace('\n', '_')
    if plot_3D:
        plt.savefig(f"figures/scatter-plots/{file_name.lower()}_3D.png", dpi=300)
    else:
        plt.savefig(f'figures/scatter-plots/{file_name.lower()}.png', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.clf()


def scatterplot_original(df, data, indices, title=None, show_plot=False, plot_3D=False):
    if plot_3D:
        ax = plt.axes(projection='3d')
        ax.scatter3D(data[:, indices[0]], data[:, indices[1]], data[:, indices[2]])
        ax.set_xlabel(df.columns[indices[0]])
        ax.set_ylabel(df.columns[indices[1]])
        ax.set_zlabel(df.columns[indices[2]])

    else:
        plt.scatter(data[:, indices[0]], data[:, indices[1]])
        plt.xlabel(df.columns[indices[0]])
        plt.ylabel(df.columns[indices[1]])

    plt.title(title)
    file_name = title.replace(' ', '_').replace('\n', '_')
    if plot_3D:
        plt.savefig(f'figures/scatter-plots/{file_name.lower()}_3D.png', dpi=300)
    else:
        plt.savefig(f'figures/scatter-plots/{file_name.lower()}.png', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.clf()


def scatterplot_transformed_or_reconstructed(data, indices=(0, 1, 2), title=None, show_plot=False, plot_3D=False):

    if plot_3D:
        ax = plt.axes(projection='3d')
        ax.scatter3D(data[:, indices[0]], data[:, indices[1]], data[:, indices[2]]);

    else:
        plt.scatter(data[:, indices[0]], data[:, indices[1]])

    plt.title(title)
    file_name = title.replace(' ', '_').replace('\n', '_')
    if plot_3D:
        plt.savefig(f'figures/scatter-plots/{file_name.lower()}_3D.png', dpi=300)
    else:
        plt.savefig(f'figures/scatter-plots/{file_name.lower()}.png', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.clf()


def plot_clusters(k_values, all_labels, data, dataset_name, indices=(0, 1)):
    for key in all_labels.keys():
        i = 0
        for labels in all_labels[key]:
            scatter_plot(labels, data, indices, f'Clustering of {dataset_name} with {key} using {k_values[i]} clusters')
            i += 1
