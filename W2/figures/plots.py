import numpy as np
from matplotlib import pyplot as plt

def scatter_plot_clustered(labels, data, indices=(0, 1, 2), title=None, show_plot=False, plot_3D=False):
    num_clusters = len(np.unique(labels))
    file_name = title.replace(' ', '_').replace('\n', '_')

    if plot_3D:
        file_name = file_name + '_3D'
        if data.shape[1] >= 3:
            ax = plt.axes(projection='3d')
            for k in range(num_clusters):
                ax.scatter3D(data[np.argwhere(labels == k)][:, :, indices[0]],
                             data[np.argwhere(labels == k)][:, :, indices[1]],
                             data[np.argwhere(labels == k)][:, :, indices[2]])

    else:
        for k in range(num_clusters):
            plt.scatter(data[np.argwhere(labels == k)][:, :, indices[0]],
                        data[np.argwhere(labels == k)][:, :, indices[1]])

    plt.title(title)
    plt.savefig(f'figures/scatter-plots/{file_name.lower()}.png', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.clf()

def scatter_plot_data_only(data, indices=(0, 1, 2), label_names=None, title=None, show_plot=False, plot_3D=False):

    file_name = title.replace(' ', '_').replace('\n', '_')

    if plot_3D:
        file_name = file_name + '_3D'
        if data.shape[1] >= 3:
            ax = plt.axes(projection='3d')
            ax.scatter3D(data[:, indices[0]],
                         data[:, indices[1]],
                         data[:, indices[2]])

            if label_names is not None:
                ax.set_xlabel(label_names[indices[0]])
                ax.set_ylabel(label_names[indices[1]])
                ax.set_zlabel(label_names[indices[2]])

    else:
        plt.scatter(data[:, indices[0]],
                    data[:, indices[1]])

    plt.title(title)
    file_name = title.replace(' ', '_').replace('\n', '_')
    plt.savefig(f'figures/scatter-plots/{file_name.lower()}.png', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.clf()
