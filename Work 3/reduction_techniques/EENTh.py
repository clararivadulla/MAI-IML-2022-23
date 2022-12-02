import numpy as np
import pandas as pd

from kNN.kNN import kNN


class EENTh:
    def __init__(self, th=0.5, k=1, dist_metric='minkowski', r=2, weights="uniform"):
        self.k = k
        self.dist_metric = dist_metric
        self.r = r
        self.weights = weights
        self.x_train = None
        self.y_train = None
        self.w = None
        self.th = th

    def reduce(self, x_train, y_train):
        S_x = []
        S_y = []
        kNN_config = kNN(k=self.k, dist_metric=self.dist_metric, r=self.r, weights=self.weights)
        labels = np.unique(y_train)
        for i in range(len(x_train)):
            label = y_train[i]
            x_train_new = np.delete(x_train, i, axis=0)
            y_train_new = np.delete(y_train, i, axis=0)
            probabilities = [self.P_i_x(x_train_new, y_train_new, x_train[i], l, kNN_config) for l in
                             range(len(labels))]
            pj = np.max(probabilities)
            k_prob = probabilities.index(max(probabilities))
            if k_prob == label and pj > self.th:  # If δk-prob (x) ≠ θ or pj ≤ μ, do S ← S − {x}
                S_x.append(x_train[i])
                S_y.append(y_train[i])
        return np.array(S_x), np.array(S_y)

    def P_i_x(self, x_train, y_train, x, i, kNN):  # Probability Pi(x) that a sample x belongs to a class i
        Pij = 0
        kNN.fit(x_train, y_train)
        neighbors, distance = kNN.get_neighbors(x)
        for j in range(0, self.k):
            l = neighbors.iloc[[j]]['label'].values[0]
            d = distance.iloc[[j]].values[0]
            if l == i:
                p_i_j = 1
            else:
                p_i_j = 0
            Pij += p_i_j * (1 / (1 + d))
        return Pij
