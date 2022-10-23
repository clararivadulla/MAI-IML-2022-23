from sklearn import metrics
import pandas as pd
import numpy as np

def confusion_matrix(y_true, y_pred, class_true=None):
    if y_true.shape != y_pred.shape:
        raise Exception ("Shapes do not match")
    cat_true = list(set(y_true))
    cat_pred = list(set(y_pred))
    n_cat_true = len(cat_true)
    n_cat_pred = len(cat_pred)
    conf_matrix = pd.DataFrame(np.zeros((n_cat_true,n_cat_pred), dtype=int))
    df = pd.DataFrame({'true':y_true, 'predicted': y_pred})
    if class_true!=None:
        if len(class_true)!=n_cat_true:
            raise Exception ("Class labels do not match number of categories")
        else:
            conf_matrix.index=class_true
    for i in range(n_cat_true):
        for j in range(n_cat_pred):
            df1 = df[df['true'] == cat_true[i]].copy()
            df2 = df1[df1['predicted']==cat_pred[j]].copy()
            conf_matrix.iloc[i,j] = df2.shape[0]
    return conf_matrix

def calculate_metrics(data, predicted_labels, actual_labels, algorithm_name=None, verbose=True):
    
    if algorithm_name is not None:
        all_metrics = [algorithm_name]
    else:
        all_metrics = []

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    try:
        silhouette_score = metrics.silhouette_score(data, predicted_labels)
        all_metrics.append(silhouette_score)
        if verbose:
            print("silhouette_score: Success")
    except:
        silhouette_score = None
        all_metrics.append(silhouette_score)
        if verbose:
            print("Too few clusters: silhouette_score requires more than one cluster label")

    # https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index
    try:
        davies_bouldin_score = metrics.davies_bouldin_score(data, predicted_labels)
        all_metrics.append(davies_bouldin_score)
        if verbose:
            print("davies_bouldin_score: Success")
    except:
        davies_bouldin_score = None
        all_metrics.append(davies_bouldin_score)
        if verbose:
            print("Too few clusters: davies_bouldin_score requires more than one cluster label")

    # https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index
    try:
        calinski_harabasz_score = metrics.calinski_harabasz_score(data, predicted_labels)
        all_metrics.append(calinski_harabasz_score)
        if verbose:
            print("calinski_harabasz_score: Success")
    except:
        calinski_harabasz_score = None
        all_metrics.append(calinski_harabasz_score)
        if verbose:
            print("Too few clusters: calinski_harabasz_score requires more than one cluster label")

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html
    try:
        adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(actual_labels, predicted_labels)
        all_metrics.append(adjusted_mutual_info_score)
        if verbose:
            print("adjusted_mutual_info_score: Success")
    except:
        if verbose:
            print("Error: adjusted_mutual_info_score")

    c_matrix = confusion_matrix(actual_labels, predicted_labels)

    if verbose:
        if algorithm_name is not None:
            print(f'\nMetrics for {algorithm_name}:')
        else:
            print('\nMetrics:')

        print(f'Silhouette Score: {silhouette_score}')
        print(f'Davies Bouldin Score: {davies_bouldin_score}')
        print(f'Calinski Harabasz Score: {calinski_harabasz_score}')
        print(f'Adjusted Mutual Info Score: {adjusted_mutual_info_score}')
        print(f'Confusion matrix: {c_matrix}')

    return all_metrics
