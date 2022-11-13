from sklearn import metrics
import pandas as pd
import numpy as np

def accuracy(y_true, y_pred):
    c_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.max(c_matrix, axis=0))/len(y_true)


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

def calculate_metrics(data, predicted_labels, actual_labels, algorithm_name=None, verbose=False):

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    silhouette_score = metrics.silhouette_score(data, predicted_labels)

    # https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index
    davies_bouldin_score = metrics.davies_bouldin_score(data, predicted_labels)

    if verbose:
        if algorithm_name is not None:
            print(f'\nMetrics for {algorithm_name}:')
        else:
            print('\nMetrics:')

        print(f'Silhouette Score: {silhouette_score}')
        print(f'Davies Bouldin Score: {davies_bouldin_score}')

    if algorithm_name is not None:
        all_metrics = [algorithm_name, silhouette_score, davies_bouldin_score]
    else:
        all_metrics = [silhouette_score, davies_bouldin_score]

    return all_metrics