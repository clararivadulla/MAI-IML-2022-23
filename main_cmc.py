from figures.plots import scatter_plot
from fuzzy_clustering.fuzzy_c_means import FuzzyCMeans
from k_means.bisecting_k_means import BisectingKMeans
from k_means.k_harmonic_means import KHarmonicMeans
from k_means.k_means import KMeans
from pre_processing import read_arff_files, cmc_pre_processing
import pandas as pd
from metrics.metrics import calculate_metrics



def main():
    print(
        '··················································\nCMC DATASET\n··················································')

    df, meta = read_arff_files.main('cmc.arff')
    data, labels = cmc_pre_processing.main(df)
    scores = []

    # K-Means
    print(
        '**************************************************\nK-Means\n**************************************************')
    k_means = KMeans(k=3)
    k_means.train(data)
    k_means_labels = k_means.classify(data)[0]
    print('Centroids: \n' + str(k_means.centroids))
    print('Actual Labels: ' + str(labels))
    print('Predicted Labels: ' + str(k_means_labels))

    k_means_metrics = calculate_metrics(data=data,
                                        predicted_labels=k_means_labels,
                                        actual_labels=labels,
                                        algorithm_name='K-Means',
                                        verbose=True)
    scores.append(k_means_metrics)

    scatter_plot(k_means_labels, data, (0, 1), title='CMC dataset\nK-Means with 3 clusters', x_label='x', y_label='y')

    # Bisecting K Means
    print(
        '**************************************************\nBisecting K-Means\n**************************************************')
    bisecting_k_means_labels = BisectingKMeans(data, k=3)
    print('Actual Labels: ' + str(labels))
    print('Predicted Labels: ' + str(bisecting_k_means_labels))

    bisecting_k_means_metrics = calculate_metrics(data=data,
                                                  predicted_labels=bisecting_k_means_labels,
                                                  actual_labels=labels,
                                                  algorithm_name='Bisecting K-Means',
                                                  verbose=True)
    scores.append(bisecting_k_means_metrics)

    scatter_plot(bisecting_k_means_labels, data, (0, 1), title='CMC dataset\nBisecting K-Means with 3 clusters',
                 x_label='x', y_label='y')

    # K-Harmonic Means
    print(
        '**************************************************\nK-Harmonic Means\n**************************************************')
    k_harmonic_means = KHarmonicMeans(n_clusters=3, max_iter=100)
    k_harmonic_means.khm(data)
    k_harmonic_means_labels = k_harmonic_means.cluster_matching(data)
    print('Centroids: \n' + str(k_harmonic_means.centroids))
    print('Actual Labels: ' + str(labels))
    print('Predicted Labels: ' + str(k_harmonic_means_labels))

    k_harmonic_means_metrics = calculate_metrics(data=data,
                                                 predicted_labels=k_harmonic_means_labels,
                                                 actual_labels=labels,
                                                 algorithm_name='K-Harmonic Means',
                                                 verbose=True)
    scores.append(k_harmonic_means_metrics)

    scatter_plot(k_harmonic_means_labels, data, (0, 1), title='CMC dataset\nK-Harmonic Means with 3 clusters',
                 x_label='x', y_label='y')

    # Fuzzy C-Means
    print(
        '**************************************************\nFuzzy C-Means\n**************************************************')
    fuzzy_c_means = FuzzyCMeans(n_clusters=3)
    fuzzy_c_means.fcm(data)
    fuzzy_c_means_labels = fuzzy_c_means.cluster_matching(data)
    print('Centroids: \n' + str(fuzzy_c_means.V))
    print('Actual Labels: ' + str(labels))
    print('Predicted Labels: ' + str(fuzzy_c_means_labels))

    fuzzy_c_means_metrics = calculate_metrics(data=data,
                                              predicted_labels=fuzzy_c_means_labels,
                                              actual_labels=labels,
                                              algorithm_name='Fuzzy C-Means',
                                              verbose=True)
    scores.append(fuzzy_c_means_metrics)

    scatter_plot(fuzzy_c_means_labels, data, (0, 1), title='CMC dataset\nFuzzy C-Means with 3 clusters', x_label='x',
                 y_label='y')

    # Save the scores in a dataframe for future graphs
    scores_df = pd.DataFrame(scores, columns=['Algorithm', 'Silhouette Score', 'Davies Bouldin Score',
                                              'Calinski Harabasz Score', 'Adjusted Mutual Info Score'])
    print(scores_df)

if __name__ == '__main__':
    main()