from pca.pca import pca
from pre_processing import read_arff_files, cmc_pre_processing
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.manifold import TSNE
from k_means.k_means import KMeans
from figures.plots import scatter_plot

def main():

    df, meta = read_arff_files.main('cmc.arff')
    data, labels = cmc_pre_processing.main(df)

    """
    Without any dimensionality reduction
    """

    # K-Means
    k_means = KMeans(k=3, max_iter=300, n_repeat=20, seed=12345)
    k_means.train(data)
    k_means_labels = k_means.classify(data)[0]
    scatter_plot(k_means_labels, data, (0, 1),
                 title='CMC dataset\nK-Means with 3 clusters\nwithout any dimensionality reduction')

    # Agglomerative Clustering
    agglomerative_clustering = AgglomerativeClustering(n_clusters=3, affinity='manhattan',
                                                                   linkage='average').fit(
        data)
    agglomerative_clustering_labels = agglomerative_clustering.labels_
    scatter_plot(agglomerative_clustering_labels, data,
                 title='CMC dataset\nAgglomerative Clustering with 3 clusters\nwithout any dimensionality reduction')

    """
    Using our own PCA
    """

    subspace = pca(data, 2)

    # K-Means
    k_means_pca = KMeans(k=3, max_iter=300, n_repeat=20, seed=12345)
    k_means_pca.train(subspace)
    k_means_pca_labels = k_means_pca.classify(subspace)[0]
    scatter_plot(k_means_pca_labels, subspace, (0, 1),
                 title='CMC dataset\nK-Means with 3 clusters\nusing our own PCA implementation')

    # Agglomerative Clustering
    agglomerative_clustering_pca = AgglomerativeClustering(n_clusters=3, affinity='manhattan',
                                                                   linkage='average').fit(
        subspace)
    agglomerative_clustering_pca_labels = agglomerative_clustering_pca.labels_
    scatter_plot(agglomerative_clustering_pca_labels, subspace,
                 title='CMC dataset\nAgglomerative Clustering with 3 clusters\nusing our own PCA implementation')


    """
    Using sklearn's PCA
    """

    # PCA
    sklearn_pca = PCA(n_components=2)
    principal_components_sklearn_pca = sklearn_pca.fit_transform(data)

    # K-Means
    k_means_sklearn_pca = KMeans(k=3, max_iter=300, n_repeat=20, seed=12345)
    k_means_sklearn_pca.train(principal_components_sklearn_pca)
    k_means_sklearn_pca_labels = k_means_sklearn_pca.classify(principal_components_sklearn_pca)[0]
    scatter_plot(k_means_sklearn_pca_labels, principal_components_sklearn_pca, (0, 1), title='CMC dataset\nK-Means with 3 clusters\nusing PCA from sklearn')

    # Agglomerative Clustering
    agglomerative_clustering_sklearn_pca = AgglomerativeClustering(n_clusters=3, affinity='manhattan', linkage='average').fit(principal_components_sklearn_pca)
    agglomerative_clustering_sklearn_pca_labels = agglomerative_clustering_sklearn_pca.labels_
    scatter_plot(agglomerative_clustering_sklearn_pca_labels, principal_components_sklearn_pca, title='CMC dataset\nAgglomerative Clustering with 3 clusters\nusing PCA from sklearn')

    """
    Using sklearn's Incremental PCA
    """

    # Incremental PCA
    incremental_pca = IncrementalPCA(n_components=2)
    principal_components_incremental_pca = incremental_pca.fit_transform(data)

    # K-Means
    k_means_incremental_pca = KMeans(k=3, max_iter=300, n_repeat=20, seed=12345)
    k_means_incremental_pca.train(principal_components_sklearn_pca)
    k_means_incremental_pca_labels = k_means_incremental_pca.classify(principal_components_incremental_pca)[0]
    scatter_plot(k_means_incremental_pca_labels, principal_components_incremental_pca, (0, 1), title='CMC dataset\nK-Means with 3 clusters\nusing Incremental PCA from sklearn')

    # Agglomerative Clustering
    agglomerative_clustering_incremental_pca = AgglomerativeClustering(n_clusters=3, affinity='manhattan',linkage='average').fit(principal_components_sklearn_pca)
    agglomerative_clustering_incremental_pca_labels = agglomerative_clustering_incremental_pca.labels_
    scatter_plot(agglomerative_clustering_incremental_pca_labels, principal_components_incremental_pca,
                 title='CMC dataset\nAgglomerative Clustering with 3 clusters\nusing Incremental PCA from sklearn')

    """
    Using sklearn's Feature Agglomeration
    """

    # Feature Agglomeration
    f_agglomeration = FeatureAgglomeration(n_clusters=2)
    f_agglomeration.fit(data)
    data_reduced = f_agglomeration.transform(data)

    # K-Means
    k_means_f_agglo = KMeans(k=3, max_iter=300, n_repeat=20, seed=12345)
    k_means_f_agglo.train(data_reduced)
    k_means_f_agglo_labels = k_means_f_agglo.classify(data_reduced)[0]
    scatter_plot(k_means_f_agglo_labels, data_reduced, (0, 1),
                 title='CMC dataset\nK-Means with 3 clusters\nusing Feature Agglomeration from sklearn')

    # Agglomerative Clustering
    agglomerative_clustering_f_agglo = AgglomerativeClustering(n_clusters=3, affinity='manhattan',
                                                                       linkage='average').fit(
    data_reduced)
    agglomerative_clustering_f_agglo_labels = agglomerative_clustering_f_agglo.labels_
    scatter_plot(agglomerative_clustering_f_agglo_labels, data_reduced,
                 title='CMC dataset\nAgglomerative Clustering with 3 clusters\nusing Feature Agglomeration from sklearn')

    """
    Using sklearn's t-SNE
    """

    # t-SNE
    data_embedded = TSNE(n_components=2, learning_rate='auto', init = 'random', perplexity = 3).fit_transform(data)

    # K-Means
    k_means_t_sne = KMeans(k=3, max_iter=300, n_repeat=20, seed=12345)
    k_means_t_sne.train(data_embedded)
    k_means_t_sne_labels = k_means_t_sne.classify(data_embedded)[0]
    scatter_plot(k_means_t_sne_labels, data_embedded, (0, 1),
                 title='CMC dataset\nK-Means with 3 clusters\nusing t-SNE from sklearn')

    # Agglomerative Clustering
    agglomerative_clustering_t_sne = AgglomerativeClustering(n_clusters=3, affinity='manhattan',
                                                               linkage='average').fit(
        data_embedded)
    agglomerative_clustering_t_sne_labels = agglomerative_clustering_t_sne.labels_
    scatter_plot(agglomerative_clustering_t_sne_labels, data_embedded,
                 title='CMC dataset\nAgglomerative Clustering with 3 clusters\nusing t-SNE from sklearn')



if __name__ == '__main__':
    main()
