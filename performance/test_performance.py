from fuzzy_clustering.fuzzy_c_means import FuzzyCMeans
from k_means.bisecting_k_means import BisectingKMeans
from k_means.k_harmonic_means import KHarmonicMeans
from k_means.k_means import KMeans
from pre_processing import read_arff_files, iris_pre_processing, cmc_pre_processing, pima_diabetes_pre_processing
from metrics.metrics import calculate_metrics


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
    else:
        raise NameError(f'Wrong dataset name: {dataset_name}')

    all_metrics = {}

    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for k in k_values:
        bisecting_k_means_labels = BisectingKMeans(data, k=k)
        bisecting_k_means_metrics = calculate_metrics(data=data,
                                                      predicted_labels=bisecting_k_means_labels,
                                                      actual_labels=labels)

        all_metrics[k] = bisecting_k_means_metrics

    print(all_metrics)


if __name__ == '__main__':
    test_performance(dataset_name='pima_diabetes')
