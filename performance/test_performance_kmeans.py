import sys
sys.path.append('../')
from k_means.k_means import KMeans
from pre_processing import read_arff_files, iris_pre_processing, cmc_pre_processing, pima_diabetes_pre_processing
from validation_metrics.metrics import calculate_metrics
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
        data, labels = cmc_pre_processing.main(df, numerical_only=True)
    else:
        raise NameError(f'Wrong dataset name: {dataset_name}')

    all_metrics = {'KMeans': []}
    k_values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for k in k_values:
        print(f'Running algorithms with k={k}')
        k_means = KMeans(k=k, max_iter=500, n_repeat=10, seed=None)
        k_means.train(data)
        k_means_labels, _ = k_means.classify(data)

        k_means_metrics = calculate_metrics(data=data,
                                            predicted_labels=k_means_labels,
                                            actual_labels=labels,
                                            verbose = True)
        all_metrics['KMeans'].append(k_means_metrics)

    plot_metrics(metrics=all_metrics, k_values=k_values, dataset_name=dataset_name)

if __name__ == '__main__':
    test_performance(dataset_name='cmc')
