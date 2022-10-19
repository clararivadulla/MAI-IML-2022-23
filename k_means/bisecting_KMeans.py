from sklearn.cluster import KMeans
#from k_means.k_means import KMeans
import numpy as np

def BKM(data, k=3, num_trials=5):
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
            trials.append(KMeans(n_clusters=2, init='random').fit(X))

        best_sse = float("inf")
        kmeans_bisect = None

        # perform k-means trials to get the best overall SSE
        for t in trials:
            trial_cluster_centers = t.cluster_centers_
            trial_labels = t.labels_

            cur_sse = 0
            for i in range(0, trial_cluster_centers.shape[0]):
                cluster_center = trial_cluster_centers[i]
                idxs = np.where(trial_labels == i)

                for x in X[idxs]:
                    cur_sse += np.square(x - cluster_center).sum()

            if(cur_sse < best_sse):
                best_sse = cur_sse
                kmeans_bisect = t

        #kmeans = KMeans()
        #kmeans.train(X)
        #a = kmeans.classify(X)

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
