
from k_means.bisecting_KMeans import BKM
from k_means.k_harmonic_means import KHarmonicMeans
from k_means.k_means import KMeans
from pre_processing import credit_a_pre_processing, cmc_pre_processing, read_arff_files, pima_diabetes_pre_processing, iris_pre_processing
from sklearn import metrics

if __name__=='__main__':

    # cmc dataset reading and pre-processing
    df, meta = read_arff_files.main('cmc.arff')
    data_cmc, labels_cmc = cmc_pre_processing.main(df)

    # credit-a dataset reading and pre-processing
    df2, meta2 = read_arff_files.main('credit-a.arff')
    data_credit_a, labels_credit_a = credit_a_pre_processing.main(df2, meta2)

    # pima-diabetes dataset reading and pre-processing
    df3, meta3 = read_arff_files.main('pima_diabetes.arff')
    data_pima_diabetes, labels_pima_diabetes = pima_diabetes_pre_processing.main(df3)

    df4, meta4 = read_arff_files.main('iris.arff')
    data_iris, labels_iris = iris_pre_processing.main(df4)

    # K-Harmonic Means
    print('**************************************************\nK-Harmonic Means\n**************************************************')
    k_harmonic_means = KHarmonicMeans(n_clusters=3, max_iter=100)
    k_harmonic_means.khm(data_iris)
    k_harmonic_means_labels = k_harmonic_means.cluster_matching(data_iris)
    print('Centroids: \n' + str(k_harmonic_means.centroids))
    print('Actual Labels: ' + str(labels_iris))
    print('Predicted Labels: ' + str(k_harmonic_means_labels))

    silhouette_score = metrics.silhouette_score(data_iris, k_harmonic_means_labels)
    davies_bouldin_score = metrics.davies_bouldin_score(data_iris, k_harmonic_means_labels)
    calinski_harabasz_score = metrics.calinski_harabasz_score(data_iris, k_harmonic_means_labels)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels_iris, k_harmonic_means_labels)

    print('\nMetrics:')
    print(f'Silhouette Score: {silhouette_score}')
    print(f'Davies Bouldin Score: {davies_bouldin_score}')
    print(f'Calinski Harabasz Score: {calinski_harabasz_score}')
    print(f'Adjusted Mutual Info Score: {adjusted_mutual_info_score}')
    print()

    # Bisecting K Means
    print('**************************************************\nBisecting K-Means\n**************************************************')
    bisecting_k_means_labels = BKM(data_iris, k=3)
    print('Actual Labels: ' + str(labels_iris))
    print('Predicted Labels: ' + str(bisecting_k_means_labels))

    silhouette_score = metrics.silhouette_score(data_iris, bisecting_k_means_labels)
    davies_bouldin_score = metrics.davies_bouldin_score(data_iris, bisecting_k_means_labels)
    calinski_harabasz_score = metrics.calinski_harabasz_score(data_iris, bisecting_k_means_labels)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels_iris, bisecting_k_means_labels)

    print('\nMetrics:')
    print(f'Silhouette Score: {silhouette_score}')
    print(f'Davies Bouldin Score: {davies_bouldin_score}')
    print(f'Calinski Harabasz Score: {calinski_harabasz_score}')
    print(f'Adjusted Mutual Info Score: {adjusted_mutual_info_score}')

    """
    #read_arff_files.main()
    dataframe, meta = read_arff_file('datasets/credit-a.arff')

    print(dataframe.head(10))
    print(dataframe.describe())

    (nominal, numeric) = get_columns_by_type(meta)

    byte_strings_to_strings(dataframe)

    #drop_rows_with_missing_values(dataframe)

    substitute_missing_values_by_mean(dataframe, numeric)
    substitute_missing_values_by_most_common(dataframe, nominal)

    label_encode_columns(dataframe, nominal)
    #dataframe = one_hot_encode_columns(dataframe, nominal)

    min_max_scale_columns(dataframe, numeric)

    train_data = dataframe.iloc[:0.9*dataframe.shapep[0],:].copy()
    test_data = dataframe.iloc[0.9 * dataframe.shapep[0]:, :].copy()

    #print(dataframe.head(10))
    #print(dataframe.describe())

    clustering = AgglomerativeClustering().fit(dataframe)

    kmeans = Kmeans(k = 4, max_iter = 100, seed = None).train(train_data)
    classify = kmeans.classify(test_data)
    print (classify[0], classify[1])

    #print(clustering.labels_)
    """
