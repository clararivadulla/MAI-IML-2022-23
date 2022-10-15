import numpy as np


class KMeans:
    def __init__(self, k=3, max_iter=100, seed=None):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centroids = None

    def train(self, data):
        np.random.seed(self.seed)
        x = data.shape[0]
        y = data.shape[1]
        c_init = np.random.randint(0, x, size=self.k)
        self.centroids = data[c_init, :].copy()
        c_old = np.full((self.k, y), fill_value=np.inf)
        d = np.full((x, self.k), fill_value=np.inf)
        r = np.full((x, self.k), fill_value=0)
        n_iter = 0
        while c_old != self.centroids and n_iter < self.max_iter:
            c_old = self.centroids.copy()
            for i in range(self.k):
                d.iloc[:, i] = np.sum((data - c_old[i, :]) ** 2, axis=1)
            min_d = np.argmin(d, axis=1)
            for i in range(x):
                r[i, min_d[i]] = 1
            for i in range(self.k):
                for j in range(y):
                    self.centroids[i, j] = np.sum(data[r[:, i], j]) / np.sum(r[:, j])
            n_iter += 1

    def classify(self, data):
        x = data.shape[0]
        d = np.full((x, self.k), fill_value=np.inf)
        for i in range(self.k):
            d.iloc[:, i] = np.sum((data - self.centroids) ** 2, axis=1)
        centroids_idx = np.argmin(d, axis=1)
        return centroids_idx, self.centroids[centroids_idx]