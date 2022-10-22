import sys
sys.path.append('../')
from fuzzy_clustering.fuzzy_c_means import FuzzyCMeans
from k_means.bisecting_k_means import BisectingKMeans
from k_means.k_harmonic_means import KHarmonicMeans
from k_means.k_means import KMeans
from pre_processing import read_arff_files, iris_pre_processing, cmc_pre_processing, pima_diabetes_pre_processing
from metrics.metrics import calculate_metrics
from figures.plots import plot_metrics


from sklearn.cluster import AgglomerativeClustering


def test_performance(dataset_name):
    if dataset_name == 'iris':
        df, meta = read_arff_files.read_arff_file('./../datasets/iris.arff')
        data, labels = iris_pre_processing.main(df)
    elif dataset_name == 'pima_diabetes':
        df, meta = read_arff_files.read_arff_file('./../datasets/pima_diabetes.arff')
        data, labels = pima_diabetes_pre_processing.main(df)
    elif dataset_name == 'cmc':
        df, meta = read_arff_files.read_arff_file('./../datasets/cmc.arff')
        data, labels = cmc_pre_processing.main(df)
    else:
        raise NameError(f'Wrong dataset name: {dataset_name}')

    all_metrics = {
        'KMeans': [],
        'BisectingKMeans': [],
        'AgglomerativeClustering': [],
        'FuzzyCMeans': [],
        'KHarmonicMeans': []
    }

    k_values = [2, 3, 4, 5, 6, 7]
    for k in k_values:
        print(f'Running algorithms with k={k}')
        k_means = KMeans(k=k)
        k_means.train(data)
        k_means_labels, _ = k_means.classify(data)

        k_means_metrics = calculate_metrics(data=data,
                                            predicted_labels=k_means_labels,
                                            actual_labels=labels)
        all_metrics['KMeans'].append(k_means_metrics)

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

        fuzzy_c_means = FuzzyCMeans(n_clusters=k)
        fuzzy_c_means.fcm(data)
        fuzzy_c_means_labels = fuzzy_c_means.cluster_matching(data)
        fuzzy_c_means_metrics = calculate_metrics(data=data,
                                                  predicted_labels=fuzzy_c_means_labels,
                                                  actual_labels=labels)
        all_metrics['FuzzyCMeans'].append(fuzzy_c_means_metrics)

        k_harmonic_means = KHarmonicMeans(n_clusters=k, max_iter=100)
        k_harmonic_means.khm(data)
        k_harmonic_means_labels = k_harmonic_means.cluster_matching(data)

        k_harmonic_means_metrics = calculate_metrics(data=data,
                                                     predicted_labels=k_harmonic_means_labels,
                                                     actual_labels=labels)
        all_metrics['KHarmonicMeans'].append(k_harmonic_means_metrics)

    plot_metrics(metrics=all_metrics, k_values=k_values, dataset_name=dataset_name)

if __name__ == '__main__':
    test_performance(dataset_name='iris')
