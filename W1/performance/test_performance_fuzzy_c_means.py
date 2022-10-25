import sys

from fuzzy_clustering.fuzzy_c_means import FuzzyCMeans

sys.path.append('../')
from k_means.k_harmonic_means import KHarmonicMeans
from pre_processing import read_arff_files, iris_pre_processing, cmc_pre_processing, pima_diabetes_pre_processing
from validation_metrics.metrics import calculate_metrics
from figures.plots import plot_metrics, plot_metrics_p_or_m


def test_performance(dataset_name):
    if dataset_name == 'iris':
        df, meta = read_arff_files.read_arff_file('./../datasets/iris.arff')
        data, labels = iris_pre_processing.main(df)
    elif dataset_name == 'pima_diabetes':
        df, meta = read_arff_files.read_arff_file('./../datasets/pima_diabetes.arff')
        data, labels = pima_diabetes_pre_processing.main(df)
    elif dataset_name == 'cmc':
        df, meta = read_arff_files.read_arff_file('./../datasets/cmc.arff')
        data, labels = cmc_pre_processing.main(df, numerical_only=True)
    else:
        raise NameError(f'Wrong dataset name: {dataset_name}')

    all_metrics = {'2': [], '3': [], '4': [], '5': [], '6': [], '7': []}
    m_values = [1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    for k in all_metrics.keys():
        for m in m_values:
            print(f'Running Fuzzy C-Means with k={k} and m={m}')
            fuzzy_c_means = FuzzyCMeans(n_clusters=int(k), m=m)
            fuzzy_c_means.fcm(data)
            fuzzy_c_means_labels = fuzzy_c_means.cluster_matching(data)
            print(fuzzy_c_means_labels)
            fuzzy_c_means_metrics = calculate_metrics(data=data,
                                                      predicted_labels=fuzzy_c_means_labels,
                                                      actual_labels=labels)
            all_metrics[k].append(fuzzy_c_means_metrics)

    plot_metrics_p_or_m('Fuzzy C-Means', all_metrics, m_values, dataset_name, p=False)

if __name__ == '__main__':
    #test_performance(dataset_name='cmc')
    test_performance(dataset_name='iris')
    #test_performance(dataset_name='pima_diabetes')
