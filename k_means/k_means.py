import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, k=3, max_iter=100, n_repeat=10, seed=1234):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centroids = None
        self.n_repeat = n_repeat

    def train(self, data):
        performance_list = []
        centroids_list = []
        for n in range(self.n_repeat):
            np.random.seed(self.seed+n)
            if type(data) == np.ndarray:
                data = pd.DataFrame(data)
            x = data.shape[0]
            y = data.shape[1]
            c_init = np.random.randint(0, x, size=self.k)
            centroids = data.iloc[c_init, :].copy()
            centroids.index = np.arange(0, self.k, 1)
            c_old = pd.DataFrame(np.full((self.k, y), fill_value=np.inf))
            c_old.columns = centroids.columns
            d = np.full((x, self.k), fill_value=np.inf)
            r = np.full((x, self.k), fill_value=0)
            n_iter = 0
            while any(c_old != centroids) and n_iter < self.max_iter:
                c_old = centroids.copy()
                c_old.index = np.arange(0, self.k, 1)
                for i in range(self.k):
                    d[:, i] = np.sum((data - c_old.iloc[i, :]) ** 2, axis=1)
                min_d = np.argmin(d, axis=1)
                for i in range(x):
                    r[i, min_d[i]] = 1
                for i in range(self.k):
                    for j in range(y):
                        mask = r[:, i] == 1
                        centroids.iloc[i, j] = np.sum(data.iloc[mask, j]) / np.sum(r[:, i])
                n_iter += 1
            data_mean = np.average(data, axis=0)
            performance_index = 0
            for i in range(self.k):
                mask = r[:, i] == 1
                performance_index += (np.sum(np.sum((data.iloc[mask, :]-centroids.iloc[i,:])** 2)) - np.sum((centroids.iloc[i,:]-data_mean)**2))
            centroids_list.append(centroids)
            performance_list.append(performance_index)
        self.centroids = centroids_list[np.argmin(performance_list)]



    def classify(self, data):
        if type(data) == np.ndarray:
            data = pd.DataFrame(data)
        x = data.shape[0]
        d = np.full((x, self.k), fill_value=np.inf)
        for i in range(self.k):
            d[:, i] = np.sum((data - self.centroids.iloc[i, :]) ** 2, axis=1)
        centroids_idx = np.argmin(d, axis=1)
        return centroids_idx, self.centroids

