import sys

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

    all_metrics = {'2': [], '3': [], '4': [], '5': [], '6': [], '7': [], }
    p_values = [2, 2.5, 3, 3.5, 4, 4.5]
    for k in all_metrics.keys():
        for p in p_values:
            print(f'Running K-Harmonic Means with k={k} and p={p}')
            k_harmonic_means = KHarmonicMeans(n_clusters=int(k), p=p)
            k_harmonic_means.khm(data)
            k_harmonic_means_labels = k_harmonic_means.cluster_matching(data)

            k_harmonic_means_metrics = calculate_metrics(data=data,
                                                         predicted_labels=k_harmonic_means_labels,
                                                         actual_labels=labels,
                                                         verbose=True)
            all_metrics[k].append(k_harmonic_means_metrics)

    plot_metrics_p_or_m('K-Harmonic Means', all_metrics, p_values, dataset_name, p=True)


if __name__ == '__main__':
    test_performance(dataset_name='cmc')
    #test_performance(dataset_name='iris')
    #test_performance(dataset_name='pima-diabetes')
