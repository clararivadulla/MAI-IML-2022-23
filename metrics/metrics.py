from sklearn import metrics

def confusion_matrix(y_true, y_pred, class_true=None):
    if y_true.shape != y_pred.shape:
        raise Exception ("Shapes do not match")
    cat_true = list(set(y_true))
    cat_pred = list(set(y_pred))
    if len(cat_true) != len(cat_pred):
        raise Exception ("Number of categories are different")
    n_cat = len(cat_true)
    n = len(y_true)
    conf_matrix = pd.DataFrame(np.zeros((n_cat,n_cat), dtype=int))
    df = pd.DataFrame({'true':y_true, 'predicted': y_pred})
    if class_true!=None:
        if len(class_true)!=n_cat:
            raise Exception ("Class labels do not match number of categories")
        else:
            conf.matrix.index=class_true
    for i in range(n_cat):
        for j in range(n_cat):
            conf_matrix.iloc[i,j] = df[df['true']==cat_true[i]][df['predicted']==cat_pred[j]].shape[0]
    return conf_matrix

def calculate_metrics(data, predicted_labels, actual_labels, algorithm_name=None, verbose=False):

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    silhouette_score = metrics.silhouette_score(data, predicted_labels)

    # https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index
    davies_bouldin_score = metrics.davies_bouldin_score(data, predicted_labels)

    # https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index
    calinski_harabasz_score = metrics.calinski_harabasz_score(data, predicted_labels)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(actual_labels, predicted_labels)

    if verbose:
        if algorithm_name is not None:
            print(f'\nMetrics for {algorithm_name}:')
        else:
            print('\nMetrics:')

        print(f'Silhouette Score: {silhouette_score}')
        print(f'Davies Bouldin Score: {davies_bouldin_score}')
        print(f'Calinski Harabasz Score: {calinski_harabasz_score}')
        print(f'Adjusted Mutual Info Score: {adjusted_mutual_info_score}')

    if algorithm_name is not None:
        all_metrics = [algorithm_name, silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_mutual_info_score]
    else:
        all_metrics = [silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_mutual_info_score]

    return all_metrics