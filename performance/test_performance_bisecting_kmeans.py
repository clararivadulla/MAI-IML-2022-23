import sys
sys.path.append('../')
from fuzzy_clustering.fuzzy_c_means import FuzzyCMeans
from k_means.bisecting_k_means import BisectingKMeans
from k_means.k_harmonic_means import KHarmonicMeans
from k_means.k_means import KMeans
from pre_processing import read_arff_files, iris_pre_processing, cmc_pre_processing, pima_diabetes_pre_processing
from validation_metrics.metrics import calculate_metrics
from figures.plots import plot_metrics, plot_clusters


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
#        'KMeans': [],
        'BisectingKMeans': []
#        'AgglomerativeClustering': [],
#        'FuzzyCMeans': [],
#        'KHarmonicMeans': []
    }

    num_trials_values = [1, 5, 10, 15, 20, 25]
    num_k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for n in num_k_values:

        print(f'Running algorithms with num_k_values={n}')
        # k_means = KMeans(k=n, n_repeat=1)
        # k_means.train(data)
        # k_means_labels, _ = k_means.classify(data)
        #
        # k_means_metrics = calculate_metrics(data=data,
        #                                     predicted_labels=k_means_labels,
        #                                     actual_labels=labels)
        # all_metrics['KMeans'].append(k_means_metrics)

        bisecting_k_means_labels = BisectingKMeans(data, k=n)
        bisecting_k_means_metrics = calculate_metrics(data=data,
                                                      predicted_labels=bisecting_k_means_labels,
                                                      actual_labels=labels)
        all_metrics['BisectingKMeans'].append(bisecting_k_means_metrics)

    plot_metrics(metrics=all_metrics, k_values=num_k_values, dataset_name=dataset_name, x_label='k')

if __name__ == '__main__':
    test_performance(dataset_name='cmc')
