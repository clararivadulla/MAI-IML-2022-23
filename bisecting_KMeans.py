from sklearn.cluster import KMeans
import numpy as np


def BKM(data, k=3):
    """
    Bisecting KMeans Algorithm
    :param data: data for clustering
    :param k: number of clusters
    :return: labels representing the clusters
    """
    X = data.copy()
    num_clusters = 1
    labels = np.zeros(X.shape[0])
    indices_to_split = np.where(labels == 0)[0]
    cluster_to_split = 0

    while num_clusters < k:
        kmeans_bisect = KMeans(n_clusters=2).fit(X)

        if num_clusters == 1:
            labels = kmeans_bisect.labels_
        else:
            # update the labels after split
            labels[indices_to_split] = [cluster_to_split if label == 0 else num_clusters for label in
                                        kmeans_bisect.labels_]

        # pick the largest cluster to split
        cluster_to_split = np.argmax(np.bincount(labels))
        indices_to_split = np.where(labels == cluster_to_split)[0]

        X = data[labels == cluster_to_split]
        num_clusters += 1

    return labels
