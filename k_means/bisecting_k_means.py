from k_means.k_means import KMeans
import numpy as np

def BisectingKMeans(data, k=3, num_trials=10):
    """
    Bisecting KMeans Algorithm
    :param data: data for clustering
    :param k: number of clusters
    :param num_trials: number of trials of k-means to perform per split
    :return: labels representing the clusters
    """
    X = data.copy()
    num_clusters = 1
    labels = np.zeros(X.shape[0])
    indices_to_split = np.where(labels == 0)[0]
    cluster_to_split = 0

    while num_clusters < k:
        trials = []
        for t in range(0, num_trials):
            k_means = KMeans(k=2, n_repeat=1, seed=None)
            k_means.train(X)
            k_means_labels, k_means_centroids = k_means.classify(X)
            trials.append((k_means_labels, k_means_centroids.to_numpy()))

        best_sse = float("inf")

        # perform k-means trials to get the best overall SSE
        for t in trials:
            (trial_labels, trial_cluster_centers) = t

            cur_sse = 0
            for i in range(0, trial_cluster_centers.shape[0]):
                cluster_center = trial_cluster_centers[i]
                idxs = np.where(trial_labels == i)

                for x in X[idxs]:
                    cur_sse += np.square(x - cluster_center).sum()

            if(cur_sse < best_sse):
                best_sse = cur_sse
                kmeans_bisect_labels = trial_labels

        if num_clusters == 1:
            labels = kmeans_bisect_labels
        else:
            # update the labels after split
            labels[indices_to_split] = [cluster_to_split if label == 0 else num_clusters for label in
                                        kmeans_bisect_labels]

        # pick the largest cluster to split
        cluster_to_split = np.argmax(np.bincount(labels))
        indices_to_split = np.where(labels == cluster_to_split)[0]

        X = data[labels == cluster_to_split]
        num_clusters += 1

    return labels
