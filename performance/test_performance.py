from fuzzy_clustering.fuzzy_c_means import FuzzyCMeans
from k_means.bisecting_k_means import BisectingKMeans
from k_means.k_harmonic_means import KHarmonicMeans
from k_means.k_means import KMeans
from pre_processing import read_arff_files, iris_pre_processing, cmc_pre_processing, \
    pima_diabetes_pre_processing  # , vehicle_pre_processing, segment_pre_processing, vowel_pre_processing
from metrics.metrics import calculate_metrics

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def test_performance(dataset_name):
    if dataset_name == 'cmc':
        df, meta = read_arff_files.read_arff_file('./../datasets/cmc.arff')
        data, labels = cmc_pre_processing.main(df)
    elif dataset_name == 'iris':
        df, meta = read_arff_files.read_arff_file('./../datasets/iris.arff')
        data, labels = iris_pre_processing.main(df)
    elif dataset_name == 'pima_diabetes':
        df, meta = read_arff_files.read_arff_file('./../datasets/pima_diabetes.arff')
        data, labels = pima_diabetes_pre_processing.main(df)
    #    elif dataset_name == 'vehicle':
    #        df, meta = read_arff_files.read_arff_file('./../datasets/vehicle.arff')
    #        data, labels = vehicle_pre_processing.main(df)
    #    elif dataset_name == 'segment':
    #        df, meta = read_arff_files.read_arff_file('./../datasets/segment.arff')
    #        data, labels = segment_pre_processing.main(df)
    #    elif dataset_name == 'vowel':
    #        df, meta = read_arff_files.read_arff_file('./../datasets/vowel.arff')
    #        data, labels = vowel_pre_processing.main(df, meta)
    else:
        raise NameError(f'Wrong dataset name: {dataset_name}')

    all_metrics = {
        'BisectingKMeans': [],
        'AgglomerativeClustering': []
    }

    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for k in k_values:
        bisecting_k_means_labels = BisectingKMeans(data, k=k)
        bisecting_k_means_metrics = calculate_metrics(data=data,
                                                      predicted_labels=bisecting_k_means_labels,
                                                      actual_labels=labels)
        all_metrics['BisectingKMeans'].append(bisecting_k_means_metrics)

        agglomerative_clustering = AgglomerativeClustering(n_clusters=k).fit(data)
        agglomerative_clustering_labels = agglomerative_clustering.labels_
        agglomerative_clustering_metrics = calculate_metrics(data=data,
                                                             predicted_labels=agglomerative_clustering_labels,
                                                             actual_labels=labels)
        all_metrics['AgglomerativeClustering'].append(agglomerative_clustering_metrics)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes.ravel()

    for key in all_metrics.keys():
        ax[0].plot(k_values, [metric[0] for metric in all_metrics[key]], label=key)
        ax[0].set(xticks=k_values, title='Silhouette Scores', xlabel='k', ylabel='score [-1, 1] (higher = better)')
        ax[0].legend()

        ax[1].plot(k_values, [metric[1] for metric in all_metrics[key]], label=key)
        ax[1].set(xticks=k_values, title='Davies Bouldin Scores', xlabel='k',
                  ylabel='score (lower = better, 0 is best)')
        ax[1].legend()

        ax[2].plot(k_values, [metric[2] for metric in all_metrics[key]], label=key)
        ax[2].set(xticks=k_values, title='Calinski Harabasz Scores', xlabel='k', ylabel='score (higher = better)')
        ax[2].legend()

        ax[3].plot(k_values, [metric[3] for metric in all_metrics[key]], label=key)
        ax[3].set(xticks=k_values, title='Adjusted Mutual Info Scores (uses actual labels)', xlabel='k',
                  ylabel='score [0, 1] (higher = better)')
        ax[3].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_performance(dataset_name='iris')
