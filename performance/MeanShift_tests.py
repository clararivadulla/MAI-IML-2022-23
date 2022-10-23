import sys
sys.path.append('../')

from pre_processing import read_arff_files, iris_pre_processing, pima_diabetes_pre_processing, cmc_pre_processing
from validation_metrics.meanShift_metrics import calculate_metrics
from figures.plots import plot_meanShift

import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds


def get_mean_metrics(data, labels, quantile, seedDefault, seed_dim, binSeed, allCluster):
    
    if seedDefault:
        calculateSeeds = None
    else:
        seed_dim = seed_dim
        calculateSeeds = get_bin_seeds(data, seed_dim[0], seed_dim[1])
    
    calculateBandwidth = estimate_bandwidth(data, quantile=quantile)

    meanCluster = MeanShift(bandwidth=calculateBandwidth, seeds=calculateSeeds,
                            bin_seeding=binSeed, cluster_all=allCluster).fit(data)
    meanLabels = meanCluster.labels_
    meanCenters = meanCluster.cluster_centers_   # ndarray of shape (n_clusters, n_features)
    meanMetrics = calculate_metrics(data=data,predicted_labels=meanLabels,actual_labels=labels)
    
    return meanMetrics, meanCenters


def test_meanShift(dataset_name, quantile_values, verbose=False):
    
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

    allMetrics_mean = {
        'Default: None, False, True': [],
        'None, False, False': [],
        'None, True, True': [],
        '(0.1,1), False, True': []
        }

    allCenters_mean = {
        'Default: None, False, True': [],
        'None, False, False': [],
        'None, True, True': [],
        '(0.1,1), False, True': []
        }
    
    for quantile in quantile_values:
        
        # Default parameters: seeds=None, bin_seeding=False, cluster_all=True
        meanMetrics_def, meanCenters_def = get_mean_metrics(data, labels, quantile, seedDefault=True,
                                                            seed_dim=None, binSeed=False, allCluster=True)
        allMetrics_mean['Default: None, False, True'].append(meanMetrics_def)
        allCenters_mean['Default: None, False, True'].append(meanCenters_def)

        # effect of cluster_all; Adjusted parameters: cluster_all= False
        meanMetrics_allClusterFalse, meanCenters_allClusterFalse = get_mean_metrics(data, labels, quantile, seedDefault=True,
                                                                                    seed_dim=None, binSeed=False, allCluster=False)
        allMetrics_mean['None, False, False'].append(meanMetrics_allClusterFalse)
        allCenters_mean['None, False, False'].append(meanCenters_allClusterFalse)

        # effect of bin_seeding; Adjusted parameters: bin_seeding= True
        meanMetrics_binSeedTrue, meanCenters_binSeedTrue = get_mean_metrics(data, labels, quantile, seedDefault=True,
                                                                            seed_dim=None, binSeed=True, allCluster=True)
        allMetrics_mean['None, True, True'].append(meanMetrics_binSeedTrue)
        allCenters_mean['None, True, True'].append(meanCenters_binSeedTrue)

        # effect of seed definition; Adjusted parameters: seeds=(0.1,1)
        seed_dim = (0.1,1)
        try:
            meanMetrics_seed011, meanCenters_seed011 = get_mean_metrics(data, labels, quantile,
                                                                        seedDefault=False, seed_dim=seed_dim, binSeed=False, allCluster=True)
        except:
            meanMetrics_seed011, meanCenters_seed011 = None, None
            if verbose:
                print('No point was within the specified bandwidth of any seed. This test will be skipped.', '\n')
        allMetrics_mean['(0.1,1), False, True'].append(meanMetrics_seed011)
        allCenters_mean['(0.1,1), False, True'].append(meanCenters_seed011)

    return allMetrics_mean, allCenters_mean





if __name__ == '__main__':
    quantile_values = [0.2, 0.3, 0.5, 0.75, 1]
    testMean_results_cmc = test_meanShift(dataset_name='cmc', quantile_values=quantile_values)
    testMean_results_diabetes = test_meanShift(dataset_name='pima_diabetes', quantile_values=quantile_values)
    testMean_results_iris = test_meanShift(dataset_name='iris', quantile_values=quantile_values)

    # plots
    plot_meanShift(dataset_name='cmc', testMean_results=testMean_results_cmc[0], quantile_values=quantile_values)
    plot_meanShift(dataset_name='pima_diabetes', testMean_results=testMean_results_diabetes[0], quantile_values=quantile_values)
    plot_meanShift(dataset_name='iris', testMean_results=testMean_results_iris[0], quantile_values=quantile_values)
