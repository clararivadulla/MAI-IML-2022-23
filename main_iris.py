from figures.plots import scatter_plot
from fuzzy_clustering.fuzzy_c_means import FuzzyCMeans
from k_means.bisecting_k_means import BisectingKMeans
from k_means.k_harmonic_means import KHarmonicMeans
from k_means.k_means import KMeans
from pre_processing import read_arff_files, iris_pre_processing
from validation_metrics.metrics import calculate_metrics
from sklearn.cluster import AgglomerativeClustering, MeanShift
import pandas as pd

def main():
    print(
        '\n\n··················································\nIRIS DATASET\n··················································')

    df, meta = read_arff_files.main('iris.arff')
    data, labels = iris_pre_processing.main(df)
    scores = []

    # Agglomerative clustering
    print(
        '\n**************************************************\nAgglomerative\n**************************************************')

    agglomerative_clustering = AgglomerativeClustering(n_clusters=3).fit(data)
    agglomerative_clustering_labels = agglomerative_clustering.labels_
    agglomerative_clustering_metrics = calculate_metrics(data=data,
                                                         predicted_labels=agglomerative_clustering_labels,
                                                         actual_labels=labels,
                                                         algorithm_name='Agglomerative Clustering',
                                                         verbose=True)
    scores.append(agglomerative_clustering_metrics)

    scatter_plot(agglomerative_clustering_labels, data, (0, 1), title='Iris dataset\nAgglomerative Clustering with 3 clusters')

    # Mean Shift clustering
    print(
        '\n**************************************************\nMean Shift\n**************************************************')

    meanshift_clustering = MeanShift().fit(data)
    mean_shift_clustering_labels = meanshift_clustering.labels_
    mean_shift_clustering_metrics = calculate_metrics(data=data,
                                                      predicted_labels=mean_shift_clustering_labels,
                                                      actual_labels=labels,
                                                      algorithm_name='Mean Shift Clustering',
                                                      verbose=True)
    scores.append(mean_shift_clustering_metrics)

    scatter_plot(mean_shift_clustering_labels, data, (0, 1), title='Iris dataset\nMean Shift Clustering')

    # K-Means
    print(
        '\n**************************************************\nK-Means\n**************************************************')
    k_means = KMeans(k=3)
    k_means.train(data)
    k_means_labels = k_means.classify(data)[0]
    k_means_metrics = calculate_metrics(data=data,
                                        predicted_labels=k_means_labels,
                                        actual_labels=labels,
                                        algorithm_name='K-Means',
                                        verbose=True)
    scores.append(k_means_metrics)
    scatter_plot(k_means_labels, data, (0, 1), title='Iris dataset\nK-Means with 3 clusters')

    # Bisecting K Means
    print(
        '\n**************************************************\nBisecting K-Means\n**************************************************')
    bisecting_k_means_labels = BisectingKMeans(data, k=3)
    bisecting_k_means_metrics = calculate_metrics(data=data,
                                                  predicted_labels=bisecting_k_means_labels,
                                                  actual_labels=labels,
                                                  algorithm_name='Bisecting K-Means',
                                                  verbose=True)
    scores.append(bisecting_k_means_metrics)
    scatter_plot(bisecting_k_means_labels, data, (0, 1), title='Iris dataset\nBisecting K-Means with 3 clusters')

    # K-Harmonic Means
    print(
        '\n**************************************************\nK-Harmonic Means\n**************************************************')
    k_harmonic_means = KHarmonicMeans(n_clusters=3, max_iter=100)
    k_harmonic_means.khm(data)
    k_harmonic_means_labels = k_harmonic_means.cluster_matching(data)
    k_harmonic_means_metrics = calculate_metrics(data=data,
                                                 predicted_labels=k_harmonic_means_labels,
                                                 actual_labels=labels,
                                                 algorithm_name='K-Harmonic Means',
                                                 verbose=True)
    scores.append(k_harmonic_means_metrics)
    scatter_plot(k_harmonic_means_labels, data, (0, 1), title='Iris dataset\nK-Harmonic Means with 3 clusters')

    # Fuzzy C-Means
    print(
        '\n**************************************************\nFuzzy C-Means\n**************************************************')
    fuzzy_c_means = FuzzyCMeans(n_clusters=3)
    fuzzy_c_means.fcm(data)
    fuzzy_c_means_labels = fuzzy_c_means.cluster_matching(data)
    fuzzy_c_means_metrics = calculate_metrics(data=data,
                                              predicted_labels=fuzzy_c_means_labels,
                                              actual_labels=labels,
                                              algorithm_name='Fuzzy C-Means',
                                              verbose=True)
    scores.append(fuzzy_c_means_metrics)
    scatter_plot(fuzzy_c_means_labels, data, (0, 1), title='Iris dataset\nFuzzy C-Means with 3 clusters')

    # Save the scores in a dataframe for future graphs
    scores_df = pd.DataFrame(scores, columns=['Algorithm', 'Silhouette Score', 'Davies Bouldin Score',
                                              'Calinski Harabasz Score', 'Adjusted Mutual Info Score'])

    print("\nAll metrics:")
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           'expand_frame_repr', False
                           ):
        print(scores_df)


if __name__ == '__main__':
    main()
