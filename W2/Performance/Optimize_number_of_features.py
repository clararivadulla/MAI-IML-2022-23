from pca.pca import pca
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.manifold import TSNE
from k_means.k_means import KMeans
from figures.plots import scatter_plot, scatter_plot_3D
from validation_metrics.metrics import calculate_metrics
from sklearn.decomposition import PCA, IncrementalPCA
import time

import pandas as pd


def run(data, labels, dataset_name, k=3, num_features=2, plot_3D=False):
    print(
        f'\n\n··················································\n{dataset_name.upper()} DATASET\n··················································')

    scores = []

    """
    Without any dimensionality reduction
    """
    print(
        '\n**************************************************\nNo Dimensionality Reduction\n**************************************************')

    # K-Means
    st = time.time()
    k_means = KMeans(k=k, max_iter=100, n_repeat=10, seed=12345)
    k_means.train(data)
    k_means_labels = k_means.classify(data)[0]
    et = time.time()
    print('Time elapsed: ', et - st)
    k_means_metrics = calculate_metrics(data=data,
                                        predicted_labels=k_means_labels,
                                        actual_labels=labels,
                                        algorithm_name='K-Means (no reduction)',
                                        verbose=True)
    scores.append(k_means_metrics)

    plot_title = f'{dataset_name} dataset\nK-Means with {k} clusters\nwithout any dimensionality reduction'
    scatter_plot(k_means_labels, data, (0, 1), title=plot_title)
    if plot_3D:
        scatter_plot_3D(k_means_labels, data, (0, 1, 2), title=plot_title)

    # Agglomerative Clustering
    st = time.time()
    agglomerative_clustering = AgglomerativeClustering(n_clusters=k, affinity='manhattan',
                                                       linkage='average').fit(
        data)
    agglomerative_clustering_labels = agglomerative_clustering.labels_
    et = time.time()
    print('Time elapsed: ', et - st)
    agglomerative_clustering_metrics = calculate_metrics(data=data,
                                                         predicted_labels=agglomerative_clustering_labels,
                                                         actual_labels=labels,
                                                         algorithm_name='Agglomerative Clustering (no reduction)',
                                                         verbose=True)
    scores.append(agglomerative_clustering_metrics)

    plot_title = f'{dataset_name} dataset\nAgglomerative Clustering with {k} clusters\nwithout any dimensionality reduction'
    scatter_plot(agglomerative_clustering_labels, data, (0, 1), title=plot_title)
    if plot_3D:
        scatter_plot_3D(agglomerative_clustering_labels, data, (0, 1, 2), title=plot_title)

    """
    Using our own PCA
    """
    print(
        '\n**************************************************\nUsing our PCA\n**************************************************')

    for f in num_features:
        transformed_data, reconstructed_data, f, variance = pca(data, f)

        # K-Means
        st = time.time()
        k_means_pca = KMeans(k=k, max_iter=100, n_repeat=10, seed=12345)
        k_means_pca.train(transformed_data)
        k_means_pca_labels = k_means_pca.classify(transformed_data)[0]
        et = time.time()
        print('Time elapsed: ', et-st)
        k_means_pca_metrics = calculate_metrics(data=transformed_data,
                                            predicted_labels=k_means_pca_labels,
                                            actual_labels=labels,
                                            algorithm_name=f'K-Means with {k} clusters, our PCA and {f} components',
                                            verbose=True)

        scores.append(k_means_pca_metrics)

        plot_title = f'{dataset_name} dataset\nK-Means with {k} clusters\nusing our own PCA implementation'
        scatter_plot(k_means_pca_labels, transformed_data, (0, 1), title=plot_title)
        if plot_3D:
            scatter_plot_3D(k_means_pca_labels, transformed_data, (0, 1, 2), title=plot_title)

        # Agglomerative Clustering
        st = time.time()
        agglomerative_clustering_pca = AgglomerativeClustering(n_clusters=k, affinity='manhattan',
                                                           linkage='average').fit(transformed_data)
        agglomerative_clustering_pca_labels = agglomerative_clustering_pca.labels_
        et = time.time()
        print('Time elapsed: ', et - st)
        agglomerative_clustering_pca_labels_metrics = calculate_metrics(data=transformed_data,
                                                                    predicted_labels=agglomerative_clustering_pca_labels,
                                                                    actual_labels=labels,
                                                                    algorithm_name=f'Agglomerative Clustering with our PCA, {k} clusters and {f} components',
                                                                    verbose=True)
        scores.append(agglomerative_clustering_pca_labels_metrics)

        plot_title = f'{dataset_name} dataset\nAgglomerative Clustering with {k} clusters\nusing our own PCA implementation'
        scatter_plot(agglomerative_clustering_pca_labels, transformed_data, (0, 1), title=plot_title)
        if plot_3D:
            scatter_plot_3D(agglomerative_clustering_pca_labels, transformed_data, (0, 1, 2), title=plot_title)

    """
    Using sklearn's PCA
    """
    print(
        '\n**************************************************\nUsing sklearn\'s PCA\n**************************************************')

    # PCA
    for f in num_features:
        sklearn_pca = PCA(n_components=f)
        principal_components_sklearn_pca = sklearn_pca.fit_transform(data)

        # K-Means
        st = time.time()
        k_means_sklearn_pca = KMeans(k=k, max_iter=100, n_repeat=10, seed=12345)
        k_means_sklearn_pca.train(principal_components_sklearn_pca)
        k_means_sklearn_pca_labels = k_means_sklearn_pca.classify(principal_components_sklearn_pca)[0]
        et = time.time()
        print('Time elapsed: ', et-st)
        k_means_sklearn_pca_metrics = calculate_metrics(data=principal_components_sklearn_pca,
                                                    predicted_labels=k_means_sklearn_pca_labels,
                                                    actual_labels=labels,
                                                    algorithm_name=f'K-Means with sklearn\'s PCA, {k} clusters and {sklearn_pca.n_components_} components',
                                                    verbose=True)
        scores.append(k_means_sklearn_pca_metrics)

        plot_title = f'{dataset_name} dataset\nK-Means with {k} clusters\nusing PCA from sklearn'
        scatter_plot(k_means_sklearn_pca_labels, principal_components_sklearn_pca, (0, 1), title=plot_title)
        if plot_3D:
            scatter_plot_3D(k_means_sklearn_pca_labels, principal_components_sklearn_pca, (0, 1, 2), title=plot_title)

        # Agglomerative Clustering
        st = time.time()
        agglomerative_clustering_sklearn_pca = AgglomerativeClustering(n_clusters=k, affinity='manhattan',
                                                                   linkage='average').fit(
            principal_components_sklearn_pca)
        agglomerative_clustering_sklearn_pca_labels = agglomerative_clustering_sklearn_pca.labels_
        et = time.time()
        print('Time elapsed: ', et-st)
        agglomerative_clustering_sklearn_pca_metrics = calculate_metrics(data=principal_components_sklearn_pca,
                                                                     predicted_labels=agglomerative_clustering_sklearn_pca_labels,
                                                                     actual_labels=labels,
                                                                     algorithm_name=f'Agglomerative Clustering with sklearn\'s PCA, {k} clusters and {sklearn_pca.n_components_} components',
                                                                     verbose=True)
        scores.append(agglomerative_clustering_sklearn_pca_metrics)

        plot_title = f'{dataset_name} dataset\nAgglomerative Clustering with {k} clusters\nusing PCA from sklearn'
        scatter_plot(agglomerative_clustering_sklearn_pca_labels, principal_components_sklearn_pca, (0, 1), title=plot_title)
        if plot_3D:
            scatter_plot_3D(agglomerative_clustering_sklearn_pca_labels, principal_components_sklearn_pca, (0, 1, 2), title=plot_title)


    scores_df = pd.DataFrame(scores, columns=['Algorithm', 'Silhouette Score', 'Davies Bouldin Score',
                                              'Calinski Harabasz Score', 'Adjusted Mutual Info Score'])
    print("\nAll metrics:")
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           'expand_frame_repr', False
                           ):
        print(scores_df)
