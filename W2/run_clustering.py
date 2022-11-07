from pca.pca import pca
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.manifold import TSNE
from k_means.k_means import KMeans
from figures.plots import scatter_plot
from validation_metrics.metrics import calculate_metrics
from sklearn.decomposition import PCA, IncrementalPCA

import pandas as pd

def run(data, labels, dataset_name, k=3, num_features=2):
    print(
        f'\n\n··················································\n{dataset_name.upper()} DATASET\n··················································')

    scores = []

    """
    Without any dimensionality reduction
    """
    print(
        '\n**************************************************\nNo Dimensionality Reduction\n**************************************************')

    # K-Means
    k_means = KMeans(k=k, max_iter=100, n_repeat=20, seed=12345)
    k_means.train(data)
    k_means_labels = k_means.classify(data)[0]
    k_means_metrics = calculate_metrics(data=data,
                                        predicted_labels=k_means_labels,
                                        actual_labels=labels,
                                        algorithm_name='K-Means (no reduction)',
                                        verbose=True)
    scores.append(k_means_metrics)

    scatter_plot(k_means_labels, data, (0, 1),
                 title=f'{dataset_name} dataset\nK-Means with 3 clusters\nwithout any dimensionality reduction')

    # Agglomerative Clustering
    agglomerative_clustering = AgglomerativeClustering(n_clusters=k, affinity='manhattan',
                                                       linkage='average').fit(
        data)
    agglomerative_clustering_labels = agglomerative_clustering.labels_
    agglomerative_clustering_metrics = calculate_metrics(data=data,
                                                         predicted_labels=agglomerative_clustering_labels,
                                                         actual_labels=labels,
                                                         algorithm_name='Agglomerative Clustering (no reduction)',
                                                         verbose=True)
    scores.append(agglomerative_clustering_metrics)

    scatter_plot(agglomerative_clustering_labels, data,
                 title=f'{dataset_name} dataset\nAgglomerative Clustering with 3 clusters\nwithout any dimensionality reduction')

    """
    Using our own PCA
    """
    print(
        '\n**************************************************\nUsing our PCA\n**************************************************')

    subspace = pca(data, num_features)

    # K-Means
    k_means_pca = KMeans(k=k, max_iter=100, n_repeat=20, seed=12345)
    k_means_pca.train(subspace)
    k_means_pca_labels = k_means_pca.classify(subspace)[0]
    k_means_pca_metrics = calculate_metrics(data=data,
                                            predicted_labels=k_means_pca_labels,
                                            actual_labels=labels,
                                            algorithm_name='K-Means (with our PCA)',
                                            verbose=True)
    scores.append(k_means_pca_metrics)
    scatter_plot(k_means_pca_labels, subspace, (0, 1),
                 title=f'{dataset_name} dataset\nK-Means with 3 clusters\nusing our own PCA implementation')

    # Agglomerative Clustering
    agglomerative_clustering_pca = AgglomerativeClustering(n_clusters=k, affinity='manhattan',
                                                           linkage='average').fit(
        subspace)
    agglomerative_clustering_pca_labels = agglomerative_clustering_pca.labels_
    agglomerative_clustering_pca_labels_metrics = calculate_metrics(data=data,
                                                                    predicted_labels=agglomerative_clustering_pca_labels,
                                                                    actual_labels=labels,
                                                                    algorithm_name='Agglomerative Clustering (with our PCA)',
                                                                    verbose=True)
    scores.append(agglomerative_clustering_pca_labels_metrics)
    scatter_plot(agglomerative_clustering_pca_labels, subspace,
                 title=f'{dataset_name} dataset\nAgglomerative Clustering with 3 clusters\nusing our own PCA implementation')


    """
    Using sklearn's PCA
    """
    print(
        '\n**************************************************\nUsing sklearn\'s PCA\n**************************************************')

    # PCA
    sklearn_pca = PCA(n_components=num_features)
    principal_components_sklearn_pca = sklearn_pca.fit_transform(data)

    # K-Means
    k_means_sklearn_pca = KMeans(k=k, max_iter=100, n_repeat=20, seed=12345)
    k_means_sklearn_pca.train(principal_components_sklearn_pca)
    k_means_sklearn_pca_labels = k_means_sklearn_pca.classify(principal_components_sklearn_pca)[0]
    k_means_sklearn_pca_metrics = calculate_metrics(data=data,
                                                    predicted_labels=k_means_sklearn_pca_labels,
                                                    actual_labels=labels,
                                                    algorithm_name='K-Means (with sklearn\'s PCA)',
                                                    verbose=True)
    scores.append(k_means_sklearn_pca_metrics)
    scatter_plot(k_means_sklearn_pca_labels, principal_components_sklearn_pca, (0, 1), title=f'{dataset_name} dataset\nK-Means with 3 clusters\nusing PCA from sklearn')

    # Agglomerative Clustering
    agglomerative_clustering_sklearn_pca = AgglomerativeClustering(n_clusters=k, affinity='manhattan', linkage='average').fit(principal_components_sklearn_pca)
    agglomerative_clustering_sklearn_pca_labels = agglomerative_clustering_sklearn_pca.labels_
    agglomerative_clustering_sklearn_pca_metrics = calculate_metrics(data=data,
                                                                     predicted_labels=agglomerative_clustering_sklearn_pca_labels,
                                                                     actual_labels=labels,
                                                                     algorithm_name='Agglomerative Clustering (with sklearn\'s PCA)',
                                                                     verbose=True)
    scores.append(agglomerative_clustering_sklearn_pca_metrics)
    scatter_plot(agglomerative_clustering_sklearn_pca_labels, principal_components_sklearn_pca, title=f'{dataset_name} dataset\nAgglomerative Clustering with 3 clusters\nusing PCA from sklearn')

    """
    Using sklearn's Incremental PCA
    """
    print(
        '\n**************************************************\nUsing sklearn\'s Incremental PCA\n**************************************************')

    # Incremental PCA
    incremental_pca = IncrementalPCA(n_components=num_features)
    principal_components_incremental_pca = incremental_pca.fit_transform(data)

    # K-Means
    k_means_incremental_pca = KMeans(k=k, max_iter=100, n_repeat=20, seed=12345)
    k_means_incremental_pca.train(principal_components_sklearn_pca)
    k_means_incremental_pca_labels = k_means_incremental_pca.classify(principal_components_incremental_pca)[0]
    k_means_incremental_pca_metrics = calculate_metrics(data=data,
                                                        predicted_labels=k_means_incremental_pca_labels,
                                                        actual_labels=labels,
                                                        algorithm_name='K-Means (with Incremental PCA)',
                                                        verbose=True)
    scores.append(k_means_incremental_pca_metrics)
    scatter_plot(k_means_incremental_pca_labels, principal_components_incremental_pca, (0, 1), title=f'{dataset_name} dataset\nK-Means with 3 clusters\nusing Incremental PCA from sklearn')

    # Agglomerative Clustering
    agglomerative_clustering_incremental_pca = AgglomerativeClustering(n_clusters=k, affinity='manhattan',linkage='average').fit(principal_components_sklearn_pca)
    agglomerative_clustering_incremental_pca_labels = agglomerative_clustering_incremental_pca.labels_
    agglomerative_clustering_incremental_pca_metrics = calculate_metrics(data=data,
                                                                         predicted_labels=agglomerative_clustering_incremental_pca_labels,
                                                                         actual_labels=labels,
                                                                         algorithm_name='Agglomerative Clustering (with Incremental PCA)',
                                                                         verbose=True)
    scores.append(agglomerative_clustering_incremental_pca_metrics)
    scatter_plot(agglomerative_clustering_incremental_pca_labels, principal_components_incremental_pca,
                 title=f'{dataset_name} dataset\nAgglomerative Clustering with 3 clusters\nusing Incremental PCA from sklearn')

    """
    Using sklearn's Feature Agglomeration
    """
    print(
        '\n**************************************************\nUsing sklearn\'s Feature Agglomeration\n**************************************************')

    # Feature Agglomeration
    f_agglomeration = FeatureAgglomeration(n_clusters=num_features)
    f_agglomeration.fit(data)
    data_reduced = f_agglomeration.transform(data)

    # K-Means
    k_means_f_agglo = KMeans(k=k, max_iter=100, n_repeat=20, seed=12345)
    k_means_f_agglo.train(data_reduced)
    k_means_f_agglo_labels = k_means_f_agglo.classify(data_reduced)[0]
    k_means_f_agglo_metrics = calculate_metrics(data=data,
                                                predicted_labels=k_means_f_agglo_labels,
                                                actual_labels=labels,
                                                algorithm_name='K-Means (with Feature Agglomeration)',
                                                verbose=True)
    scores.append(k_means_f_agglo_metrics)
    scatter_plot(k_means_f_agglo_labels, data_reduced, (0, 1),
                 title=f'{dataset_name} dataset\nK-Means with 3 clusters\nusing Feature Agglomeration from sklearn')

    # Agglomerative Clustering
    agglomerative_clustering_f_agglo = AgglomerativeClustering(n_clusters=k, affinity='manhattan',
                                                               linkage='average').fit(
        data_reduced)
    agglomerative_clustering_f_agglo_labels = agglomerative_clustering_f_agglo.labels_
    agglomerative_clustering_f_agglo_metrics = calculate_metrics(data=data,
                                                                 predicted_labels=agglomerative_clustering_f_agglo_labels,
                                                                 actual_labels=labels,
                                                                 algorithm_name='Agglomerative Clustering (with Feature Agglomeration)',
                                                                 verbose=True)
    scores.append(agglomerative_clustering_f_agglo_metrics)
    scatter_plot(agglomerative_clustering_f_agglo_labels, data_reduced,
                 title=f'{dataset_name} dataset\nAgglomerative Clustering with 3 clusters\nusing Feature Agglomeration from sklearn')

    """
    Using sklearn's t-SNE
    """
    print(
        '\n**************************************************\nUsing sklearn\'s t-SNE\n**************************************************')

    # t-SNE
    data_embedded = TSNE(n_components=num_features, learning_rate='auto', init = 'random', perplexity = 3).fit_transform(data)

    # K-Means
    k_means_t_sne = KMeans(k=k, max_iter=100, n_repeat=20, seed=12345)
    k_means_t_sne.train(data_embedded)
    k_means_t_sne_labels = k_means_t_sne.classify(data_embedded)[0]
    k_means_t_sne_metrics = calculate_metrics(data=data,
                                              predicted_labels=k_means_t_sne_labels,
                                              actual_labels=labels,
                                              algorithm_name='K-Means (with t-SNE)',
                                              verbose=True)
    scores.append(k_means_t_sne_metrics)
    scatter_plot(k_means_t_sne_labels, data_embedded, (0, 1),
                 title=f'{dataset_name} dataset\nK-Means with 3 clusters\nusing t-SNE from sklearn')

    # Agglomerative Clustering
    agglomerative_clustering_t_sne = AgglomerativeClustering(n_clusters=k, affinity='manhattan',
                                                             linkage='average').fit(
        data_embedded)
    agglomerative_clustering_t_sne_labels = agglomerative_clustering_t_sne.labels_
    agglomerative_clustering_t_sne_metrics = calculate_metrics(data=data,
                                                               predicted_labels=agglomerative_clustering_t_sne_labels,
                                                               actual_labels=labels,
                                                               algorithm_name='Agglomerative Clustering (with t-SNE)',
                                                               verbose=True)
    scores.append(agglomerative_clustering_t_sne_metrics)
    scatter_plot(agglomerative_clustering_t_sne_labels, data_embedded,
                 title=f'{dataset_name} dataset\nAgglomerative Clustering with 3 clusters\nusing t-SNE from sklearn')


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
    run()
