from pca.pca import pca
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.manifold import TSNE
from k_means.k_means import KMeans
from figures.plots import scatter_plot_clustered, scatter_plot_data_only
from validation_metrics.metrics import calculate_metrics
from sklearn.decomposition import PCA, IncrementalPCA
import time

import pandas as pd


def run(data, labels, label_names, indices_to_plot=(0,1,2), dataset_name='dataset', k=3, num_features=2, plot_3D=False):
    print(
        f'\n\n··················································\n{dataset_name.upper()} DATASET\n··················································')

    scores = []

    """
    Without any dimensionality reduction
    """
    print(
        '\n**************************************************\nNo Dimensionality Reduction\n**************************************************')

    print('\nRunning K-Means clustering:')
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
                                        algorithm_name=f'K-Means ({k} clusters, no reduction)',
                                        verbose=True)
    scores.append(k_means_metrics)

    plot_title = f'{dataset_name} dataset\nK-Means with {k} clusters\nwithout any dimensionality reduction'
    scatter_plot_clustered(k_means_labels, data, title=plot_title, plot_3D=plot_3D)

    # Agglomerative Clustering
    print('\nRunning Agglomerative clustering:')
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
                                                         algorithm_name=f'Agglomerative Clustering ({k} clusters, no reduction)',
                                                         verbose=True)
    scores.append(agglomerative_clustering_metrics)

    plot_title = f'{dataset_name} dataset\nAgglomerative Clustering with {k} clusters\nwithout any dimensionality reduction'
    scatter_plot_clustered(agglomerative_clustering_labels, data, title=plot_title, plot_3D=plot_3D)

    """
    Using our own PCA
    """
    print(
        '\n**************************************************\nUsing our PCA\n**************************************************')

    for f in num_features:
        transformed_data, reconstructed_data, f, variance = pca(data, f)

        # plotting the original dataset, transformed_dataset, and the dataset after reconstruction
        plot_title = f'{dataset_name} dataset\nwithout PCA reconstruction'
        scatter_plot_data_only(data, indices=indices_to_plot, label_names=label_names, title=plot_title, plot_3D=plot_3D)

        plot_title = f'{dataset_name} dataset\nafter PCA transformation'
        scatter_plot_data_only(transformed_data, title=plot_title, plot_3D=plot_3D)

        plot_title = f'{dataset_name} dataset\nafter PCA reconstruction'
        scatter_plot_data_only(reconstructed_data, title=plot_title, plot_3D=plot_3D)

        # K-Means
        print('\nRunning K-Means clustering:')
        st = time.time()
        k_means_pca = KMeans(k=k, max_iter=100, n_repeat=10, seed=12345)
        k_means_pca.train(transformed_data)
        k_means_pca_labels = k_means_pca.classify(transformed_data)[0]
        et = time.time()
        print('Time elapsed: ', et-st)
        k_means_pca_metrics = calculate_metrics(data=transformed_data,
                                                predicted_labels=k_means_pca_labels,
                                                actual_labels=labels,
                                                algorithm_name=f'K-Means with our PCA ({k} clusters, {f} components)',
                                                verbose=True)

        scores.append(k_means_pca_metrics)

        plot_title = f'{dataset_name} dataset\nK-Means with {k} clusters\nusing our own PCA implementation'
        scatter_plot_clustered(k_means_pca_labels, transformed_data, title=plot_title, plot_3D=plot_3D)

        # Agglomerative Clustering
        print('\nRunning Agglomerative clustering:')
        st = time.time()
        agglomerative_clustering_pca = AgglomerativeClustering(n_clusters=k, affinity='manhattan',
                                                           linkage='average').fit(transformed_data)
        agglomerative_clustering_pca_labels = agglomerative_clustering_pca.labels_
        et = time.time()
        print('Time elapsed: ', et - st)
        agglomerative_clustering_pca_labels_metrics = calculate_metrics(data=transformed_data,
                                                                        predicted_labels=agglomerative_clustering_pca_labels,
                                                                        actual_labels=labels,
                                                                        algorithm_name=f'Agglomerative Clustering with our PCA ({k} clusters, {f} components)',
                                                                        verbose=True)
        scores.append(agglomerative_clustering_pca_labels_metrics)

        plot_title = f'{dataset_name} dataset\nAgglomerative Clustering with {k} clusters\nusing our own PCA implementation'
        scatter_plot_clustered(agglomerative_clustering_pca_labels, transformed_data, title=plot_title, plot_3D=plot_3D)

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
        print('\nRunning K-Means clustering:')
        st = time.time()
        k_means_sklearn_pca = KMeans(k=k, max_iter=100, n_repeat=10, seed=12345)
        k_means_sklearn_pca.train(principal_components_sklearn_pca)
        k_means_sklearn_pca_labels = k_means_sklearn_pca.classify(principal_components_sklearn_pca)[0]
        et = time.time()
        print('Time elapsed: ', et-st)
        k_means_sklearn_pca_metrics = calculate_metrics(data=principal_components_sklearn_pca,
                                                        predicted_labels=k_means_sklearn_pca_labels,
                                                        actual_labels=labels,
                                                        algorithm_name=f'K-Means with sklearn\'s PCA, ({k} clusters, {f} components)',
                                                        verbose=True)
        scores.append(k_means_sklearn_pca_metrics)

        plot_title = f'{dataset_name} dataset\nK-Means with {k} clusters\nusing PCA from sklearn'
        scatter_plot_clustered(k_means_sklearn_pca_labels, principal_components_sklearn_pca, title=plot_title, plot_3D=plot_3D)

        # Agglomerative Clustering
        print('\nRunning Agglomerative clustering:')
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
                                                                     algorithm_name=f'Agglomerative Clustering with sklearn\'s PCA, ({k} clusters, {f} components)',
                                                                     verbose=True)
        scores.append(agglomerative_clustering_sklearn_pca_metrics)

        plot_title = f'{dataset_name} dataset\nAgglomerative Clustering with {k} clusters\nusing PCA from sklearn'
        scatter_plot_clustered(agglomerative_clustering_sklearn_pca_labels, principal_components_sklearn_pca, title=plot_title, plot_3D=plot_3D)

    """
    Using sklearn's Incremental PCA
    """
    print(
        '\n**************************************************\nUsing sklearn\'s Incremental PCA\n**************************************************')

    # Incremental PCA
    for f in num_features:
        incremental_pca = IncrementalPCA(n_components=f)
        principal_components_incremental_pca = incremental_pca.fit_transform(data)

        # K-Means
        print('\nRunning K-Means clustering:')
        st = time.time()
        k_means_incremental_pca = KMeans(k=k, max_iter=100, n_repeat=10, seed=12345)
        k_means_incremental_pca.train(principal_components_incremental_pca)
        k_means_incremental_pca_labels = k_means_incremental_pca.classify(principal_components_incremental_pca)[0]
        et = time.time()
        print('Time elapsed: ', et - st)
        k_means_incremental_pca_metrics = calculate_metrics(data=principal_components_incremental_pca,
                                                        predicted_labels=k_means_incremental_pca_labels,
                                                        actual_labels=labels,
                                                        algorithm_name=f'K-Means with Incremental PCA ({k} clusters, {f} components)',
                                                        verbose=True)
        scores.append(k_means_incremental_pca_metrics)

        plot_title = f'{dataset_name} dataset\nK-Means with {k} clusters\nusing Incremental PCA from sklearn'
        scatter_plot_clustered(k_means_incremental_pca_labels, principal_components_incremental_pca, title=plot_title, plot_3D=plot_3D)

        # Agglomerative Clustering
        print('\nRunning Agglomerative clustering:')
        st = time.time()
        agglomerative_clustering_incremental_pca = AgglomerativeClustering(n_clusters=k, affinity='manhattan',
                                                                       linkage='average').fit(
        principal_components_incremental_pca)
        agglomerative_clustering_incremental_pca_labels = agglomerative_clustering_incremental_pca.labels_
        et = time.time()
        print('Time elapsed: ', et - st)
        agglomerative_clustering_incremental_pca_metrics = calculate_metrics(data=principal_components_incremental_pca,
                                                                         predicted_labels=agglomerative_clustering_incremental_pca_labels,
                                                                         actual_labels=labels,
                                                                         algorithm_name=f'Agglomerative Clustering with Incremental PCA ({k} clusters, {f} components)',
                                                                         verbose=True)
        scores.append(agglomerative_clustering_incremental_pca_metrics)

        plot_title = f'{dataset_name} dataset\nAgglomerative Clustering with {k} clusters\nusing Incremental PCA from sklearn'
        scatter_plot_clustered(agglomerative_clustering_incremental_pca_labels, principal_components_incremental_pca, title=plot_title, plot_3D=plot_3D)

    """
    Using sklearn's Feature Agglomeration
    """
    print(
        '\n**************************************************\nUsing sklearn\'s Feature Agglomeration\n**************************************************')

    # Feature Agglomeration
    for f in num_features:
        f_agglomeration = FeatureAgglomeration(n_clusters=f)
        f_agglomeration.fit(data)
        data_reduced = f_agglomeration.transform(data)

        # K-Means
        print('\nRunning K-Means clustering:')
        st = time.time()
        k_means_f_agglo = KMeans(k=k, max_iter=100, n_repeat=10, seed=12345)
        k_means_f_agglo.train(data_reduced)
        k_means_f_agglo_labels = k_means_f_agglo.classify(data_reduced)[0]
        et = time.time()
        print('Time elapsed: ', et - st)
        k_means_f_agglo_metrics = calculate_metrics(data=data_reduced,
                                                predicted_labels=k_means_f_agglo_labels,
                                                actual_labels=labels,
                                                algorithm_name=f'K-Means with Feature Agglomeration ({k} clusters, {f} components)',
                                                verbose=True)
        scores.append(k_means_f_agglo_metrics)

        plot_title = f'{dataset_name} dataset\nK-Means with {k} clusters\nusing Feature Agglomeration from sklearn'
        scatter_plot_clustered(k_means_f_agglo_labels, data_reduced, title=plot_title, plot_3D=plot_3D)

        # Agglomerative Clustering
        print('\nRunning Agglomerative clustering:')
        st = time.time()
        agglomerative_clustering_f_agglo = AgglomerativeClustering(n_clusters=k, affinity='manhattan',
                                                               linkage='average').fit(
        data_reduced)
        agglomerative_clustering_f_agglo_labels = agglomerative_clustering_f_agglo.labels_
        et = time.time()
        print('Time elapsed: ', et - st)
        agglomerative_clustering_f_agglo_metrics = calculate_metrics(data=data_reduced,
                                                                 predicted_labels=agglomerative_clustering_f_agglo_labels,
                                                                 actual_labels=labels,
                                                                 algorithm_name=f'Agglomerative Clustering with Feature Agglomeration ({k} clusters, {f} components)',
                                                                 verbose=True)
        scores.append(agglomerative_clustering_f_agglo_metrics)

        plot_title = f'{dataset_name} dataset\nAgglomerative Clustering with {k} clusters\nusing Feature Agglomeration from sklearn'
        scatter_plot_clustered(agglomerative_clustering_f_agglo_labels, data_reduced, title=plot_title, plot_3D=plot_3D)

    """
    Using sklearn's t-SNE
    """
    print(
        '\n**************************************************\nUsing sklearn\'s t-SNE\n**************************************************')

    # t-SNE
    for f in [2, 3]:
        data_embedded = TSNE(n_components=f, learning_rate='auto', init='random', perplexity=3).fit_transform(data)

        plot_title = f'{dataset_name} dataset\nafter t-SNE transformation ({f} features)'
        scatter_plot_data_only(data_embedded, title=plot_title, plot_3D=(f >= 3))

        # K-Means
        print('\nRunning K-Means clustering:')
        st = time.time()
        k_means_t_sne = KMeans(k=k, max_iter=100, n_repeat=10, seed=12345)
        k_means_t_sne.train(data_embedded)
        k_means_t_sne_labels = k_means_t_sne.classify(data_embedded)[0]
        et = time.time()
        print('Time elapsed: ', et - st)
        k_means_t_sne_metrics = calculate_metrics(data=data_embedded,
                                              predicted_labels=k_means_t_sne_labels,
                                              actual_labels=labels,
                                              algorithm_name=f'K-Means with t-SNE ({k} clusters, {f} components)',
                                              verbose=True)
        scores.append(k_means_t_sne_metrics)

        plot_title = f'{dataset_name} dataset\nK-Means with {k} clusters\nusing t-SNE from sklearn'
        scatter_plot_clustered(k_means_t_sne_labels, data_embedded, title=plot_title, plot_3D=plot_3D)

        # Agglomerative Clustering
        print('\nRunning Agglomerative clustering:')
        st = time.time()
        agglomerative_clustering_t_sne = AgglomerativeClustering(n_clusters=k, affinity='manhattan',
                                                             linkage='average').fit(data_embedded)
        agglomerative_clustering_t_sne_labels = agglomerative_clustering_t_sne.labels_
        et = time.time()
        print('Time elapsed: ', et - st)
        agglomerative_clustering_t_sne_metrics = calculate_metrics(data=data_embedded,
                                                               predicted_labels=agglomerative_clustering_t_sne_labels,
                                                               actual_labels=labels,
                                                               algorithm_name=f'Agglomerative Clustering with t-SNE ({k} clusters, {f} components)',
                                                               verbose=True)
        scores.append(agglomerative_clustering_t_sne_metrics)

        plot_title = f'{dataset_name} dataset\nAgglomerative Clustering with {k} clusters\nusing t-SNE from sklearn'
        scatter_plot_clustered(agglomerative_clustering_t_sne_labels, data_embedded, title=plot_title, plot_3D=plot_3D)


    scores_df = pd.DataFrame(scores, columns=['Algorithm', 'Silhouette Score', 'Davies Bouldin Score'])
    print("\nAll metrics:")
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           'display.max_colwidth', 1000,
                           'expand_frame_repr', False
                           ):
        print(scores_df)
