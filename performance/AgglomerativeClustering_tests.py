import sys
sys.path.append('../')

from pre_processing import read_arff_files, iris_pre_processing, pima_diabetes_pre_processing, cmc_pre_processing
from metrics.metrics import calculate_metrics
from figures.plots import plot_agglomerative

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def get_agg_metrics(data, n_clusters, labels, affinity, linkage):
    
    aggCluster = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage).fit(data)
    aggLabels = aggCluster.labels_
    aggMetrics = calculate_metrics(data=data,predicted_labels=aggLabels,actual_labels=labels)
    
    return aggMetrics


def test_agglomerative(dataset_name, k_values, numerical_only=True, linkage=True, affinityMan=True, affinityCos=True):
    if dataset_name == 'iris':
        df, meta = read_arff_files.read_arff_file('./../datasets/iris.arff')
        data, labels = iris_pre_processing.main(df)
    elif dataset_name == 'pima_diabetes':
        df, meta = read_arff_files.read_arff_file('./../datasets/pima_diabetes.arff')
        data, labels = pima_diabetes_pre_processing.main(df)
    elif dataset_name == 'cmc':
        df, meta = read_arff_files.read_arff_file('./../datasets/cmc.arff')
        data, labels = cmc_pre_processing.main(df, numerical_only)
    else:
        raise NameError(f'Wrong dataset name: {dataset_name}')

    allMetrics_agg = {
        'Default': []}
    if linkage:
        allMetrics_agg['EuclideanComplete'] = []
        allMetrics_agg['EuclideanSingle'] = []
        allMetrics_agg['EuclideanAverage'] = []
    if affinityMan:
        allMetrics_agg['ManhattanComplete'] = []
        allMetrics_agg['ManhattanSingle'] = []
        allMetrics_agg['ManhattanAverage'] = []
    if affinityCos:
        allMetrics_agg['CosineComplete'] = []
        allMetrics_agg['CosineSingle'] = []
        allMetrics_agg['CosineAverage'] = []

    for k in k_values:
    
            # Default parameters: affinity = euclidean, linkage = ward
        agg_metrics_def = get_agg_metrics(data, n_clusters=k, labels=labels, affinity='euclidean', linkage='ward')
        allMetrics_agg['EuclideanWard (Default)'].append(agg_metrics_def)
        
        
        if linkage:
            
            # Note that the linkage 'ward' is only accepted with euclidean distance
            
                # Adjusted parameters: linkage = complete
            agg_metrics_comp = get_agg_metrics(data, n_clusters=k, labels=labels, affinity='euclidean', linkage='complete')
            allMetrics_agg['EuclideanComplete'].append(agg_metrics_comp)
            
                # Adjusted parameters: linkage = single
            agg_metrics_sing = get_agg_metrics(data, n_clusters=k, labels=labels, affinity='euclidean', linkage='single')
            allMetrics_agg['EuclideanSingle'].append(agg_metrics_sing)
            
                # Adjusted parameters: linkage = average
            agg_metrics_avg = get_agg_metrics(data, n_clusters=k, labels=labels, affinity='euclidean', linkage='average')
            allMetrics_agg['EuclideanAverage'].append(agg_metrics_avg)
    
    
        if affinityMan:
            
            # Manhattan
            
                # Adjusted parameters: affinity = manhattan, linkage = complete
            agg_metrics_compMan = get_agg_metrics(data, n_clusters=k, labels=labels, affinity='manhattan', linkage='complete')
            allMetrics_agg['ManhattanComplete'].append(agg_metrics_compMan)
            
                # Adjusted parameters: affinity = manhattan, linkage = single
            agg_metrics_singMan = get_agg_metrics(data, n_clusters=k, labels=labels, affinity='manhattan', linkage='single')
            allMetrics_agg['ManhattanSingle'].append(agg_metrics_singMan)
            
                # Adjusted parameters: affinity = manhattan, linkage = average
            agg_metrics_avgMan = get_agg_metrics(data, n_clusters=k, labels=labels, affinity='manhattan', linkage='average')
            allMetrics_agg['ManhattanAverage'].append(agg_metrics_avgMan)

        
        
        if affinityCos:
            
            # Cosine
            
                # Adjusted parameters: affinity = cosine, linkage = complete
            agg_metrics_compCos = get_agg_metrics(data, n_clusters=k, labels=labels, affinity='cosine', linkage='complete')
            allMetrics_agg['CosineComplete'].append(agg_metrics_compCos)
            
                # Adjusted parameters: affinity = cosine, linkage = single
            agg_metrics_singCos = get_agg_metrics(data, n_clusters=k, labels=labels, affinity='cosine', linkage='single')
            allMetrics_agg['CosineSingle'].append(agg_metrics_singCos)
            
                # Adjusted parameters: affinity = cosine, linkage = average
            agg_metrics_avgCos = get_agg_metrics(data, n_clusters=k, labels=labels, affinity='cosine', linkage='average')
            allMetrics_agg['CosineAverage'].append(agg_metrics_avgCos)

    return allMetrics_agg



if __name__ == '__main__':
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    testAgg_results_cmc = test_agglomerative(dataset_name='cmc', k_values=k_values, numerical_only=True)
    testAgg_results_diabetes = test_agglomerative(dataset_name='pima_diabetes', k_values=k_values)
    testAgg_results_iris = test_agglomerative(dataset_name='iris', k_values=k_values)

    # plots
    plot_agglomerative(dataset_name='cmc', testAgg_results=testAgg_results_cmc, k_values=k_values)
    plot_agglomerative(dataset_name='pima_diabetes', testAgg_results=testAgg_results_diabetes, k_values=k_values)
    plot_agglomerative(dataset_name='iris', testAgg_results=testAgg_results_iris, k_values=k_values)
