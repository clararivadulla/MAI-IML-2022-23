from sklearn import metrics

def calculate_metrics(data, predicted_labels, actual_labels, algorithm_name=None, verbose=False):

    silhouette_score = metrics.silhouette_score(data, predicted_labels)
    davies_bouldin_score = metrics.davies_bouldin_score(data, predicted_labels)
    calinski_harabasz_score = metrics.calinski_harabasz_score(data, predicted_labels)
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