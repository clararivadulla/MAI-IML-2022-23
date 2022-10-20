from fuzzy_clustering.fuzzy_c_means import FuzzyCMeans
from k_means.bisecting_k_means import BisectingKMeans
from k_means.k_harmonic_means import KHarmonicMeans
from k_means.k_means import KMeans
from pre_processing import read_arff_files, iris_pre_processing
from sklearn import metrics


def main():

    print(
        '··················································\nIRIS DATASET\n··················································')

    df, meta = read_arff_files.main('iris.arff')
    data, labels = iris_pre_processing.main(df)

    # K-Means
    print(
        '**************************************************\nK-Means\n**************************************************')
    k_means = KMeans()
    k_means.train(data)
    k_means_labels = k_means.cluster_matching(data)
    print('Centroids: \n' + str(k_means.centroids))
    print('Actual Labels: ' + str(labels))
    print('Predicted Labels: ' + str(k_means_labels))

    silhouette_score = metrics.silhouette_score(data, k_means_labels)
    davies_bouldin_score = metrics.davies_bouldin_score(data, k_means_labels)
    calinski_harabasz_score = metrics.calinski_harabasz_score(data, k_means_labels)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels, k_means_labels)

    print('\nMetrics:')
    print(f'Silhouette Score: {silhouette_score}')
    print(f'Davies Bouldin Score: {davies_bouldin_score}')
    print(f'Calinski Harabasz Score: {calinski_harabasz_score}')
    print(f'Adjusted Mutual Info Score: {adjusted_mutual_info_score}')

    # Fuzzy C-Means
    print(
        '**************************************************\nFuzzy C-Means\n**************************************************')
    fuzzy_c_means = FuzzyCMeans()
    fuzzy_c_means.fcm(data)
    fuzzy_c_means_labels = fuzzy_c_means.cluster_matching(data)
    print('Centroids: \n' + str(fuzzy_c_means.V))
    print('Actual Labels: ' + str(labels))
    print('Predicted Labels: ' + str(fuzzy_c_means_labels))

    silhouette_score = metrics.silhouette_score(data, fuzzy_c_means_labels)
    davies_bouldin_score = metrics.davies_bouldin_score(data, fuzzy_c_means_labels)
    calinski_harabasz_score = metrics.calinski_harabasz_score(data, fuzzy_c_means_labels)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels, fuzzy_c_means_labels)

    print('\nMetrics:')
    print(f'Silhouette Score: {silhouette_score}')
    print(f'Davies Bouldin Score: {davies_bouldin_score}')
    print(f'Calinski Harabasz Score: {calinski_harabasz_score}')
    print(f'Adjusted Mutual Info Score: {adjusted_mutual_info_score}')

    # K-Harmonic Means
    print(
        '**************************************************\nK-Harmonic Means\n**************************************************')
    k_harmonic_means = KHarmonicMeans(n_clusters=3, max_iter=100)
    k_harmonic_means.khm(data)
    k_harmonic_means_labels = k_harmonic_means.cluster_matching(data)
    print('Centroids: \n' + str(k_harmonic_means.centroids))
    print('Actual Labels: ' + str(labels))
    print('Predicted Labels: ' + str(k_harmonic_means_labels))

    silhouette_score = metrics.silhouette_score(data, k_harmonic_means_labels)
    davies_bouldin_score = metrics.davies_bouldin_score(data, k_harmonic_means_labels)
    calinski_harabasz_score = metrics.calinski_harabasz_score(data, k_harmonic_means_labels)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels, k_harmonic_means_labels)

    print('\nMetrics:')
    print(f'Silhouette Score: {silhouette_score}')
    print(f'Davies Bouldin Score: {davies_bouldin_score}')
    print(f'Calinski Harabasz Score: {calinski_harabasz_score}')
    print(f'Adjusted Mutual Info Score: {adjusted_mutual_info_score}')
    print()

    # Bisecting K Means
    print(
        '**************************************************\nBisecting K-Means\n**************************************************')
    bisecting_k_means_labels = BisectingKMeans(data, k=3)
    print('Actual Labels: ' + str(labels))
    print('Predicted Labels: ' + str(bisecting_k_means_labels))

    silhouette_score = metrics.silhouette_score(data, bisecting_k_means_labels)
    davies_bouldin_score = metrics.davies_bouldin_score(data, bisecting_k_means_labels)
    calinski_harabasz_score = metrics.calinski_harabasz_score(data, bisecting_k_means_labels)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels, bisecting_k_means_labels)

    print('\nMetrics:')
    print(f'Silhouette Score: {silhouette_score}')
    print(f'Davies Bouldin Score: {davies_bouldin_score}')
    print(f'Calinski Harabasz Score: {calinski_harabasz_score}')
    print(f'Adjusted Mutual Info Score: {adjusted_mutual_info_score}')

