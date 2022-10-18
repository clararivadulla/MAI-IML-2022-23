import math

import numpy as np


class FuzzyCMeans:

    def __init__(self, n_clusters=3, max_iter=100):
        self.c = n_clusters
        self.max_iter = max_iter
        self.epsilon = np.finfo(np.float32).eps # Termination threshold
        self.m = 2 # m = 1: crisp; m = 2: typical
        self.U = None
        self.V = None

    def guess_initial_centers(self, X):
        indices = np.random.choice(np.size(X[:, 0]), self.c, replace=False)
        self.V = [X[indices[i]] for i in range(self.c)]

    def update_memberships(self, X, V):
        for k in range(np.size(X[:, 0])):
            for i in range(self.c):
                sum = 0
                for j in range(self.c):
                    sum += np.power(np.linalg.norm(X[k] - V[i]) / max(np.linalg.norm(X[k] - V[j]), self.epsilon), 2 / (self.m - 1))
                self.U[i][k] = np.power(max(sum, self.epsilon), -1)

    def calculate_centers(self, X):
        V = []
        for i in range(self.c):
            num = 0
            den = 0
            for k in range(np.size(X[:, 0])):
                num += math.pow(self.U[i][k], self.m) * X[k]
                den += math.pow(self.U[i][k], self.m)
            V.append(num/max(den, self.epsilon))
        return V

    def cluster_matching(self, X):
        matches = np.zeros(np.size(X[:, 0]))
        for i in range(np.size(X[:, 0])):
            norm = np.linalg.norm(X[i] - self.V[0])
            for j in range(1, self.c):
                aux_norm = np.linalg.norm(X[i] - self.V[j])
                if aux_norm < norm:
                    norm = aux_norm
                    matches[i] = j
        return matches

    def fcm(self, data):

        self.guess_initial_centers(data)
        self.U = np.zeros((self.c, np.size(data[:, 0])))
        self.update_memberships(data, self.V)

        iter = 0
        print('Iterations: ', end=' ')
        while iter < self.max_iter:

            if iter + 1 < self.max_iter:
                print(iter + 1, end=' ')
            else:
                print(iter + 1)

            iter += 1

            V_t = self.calculate_centers(data)
            self.update_memberships(data, V_t)

            if np.linalg.norm(np.array(self.V) - np.array(V_t)) <= self.epsilon:
                self.V = V_t

        return self.V
