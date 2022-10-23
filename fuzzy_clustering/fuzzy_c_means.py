import math

import numpy as np


class FuzzyCMeans:

    def __init__(self, n_clusters=3, max_iter=100, m=2):
        self.c = n_clusters
        self.max_iter = max_iter
        self.epsilon = np.finfo(np.float32).eps  # Termination threshold
        self.m = m  # m = 1: crisp; m = 2: typical
        self.V = None

    def guess_initial_centers(self, X):
        indices = np.random.choice(len(X), self.c, replace=False)
        return np.array([X[indices[i]] for i in range(self.c)])

    def calculate_centers(self, X, U, V):
        for i in range(self.c):
            for dim in range(len(X[0])):
                num = 0
                den = 0
                for k in range(len(X)):
                    num += X[k][dim] * math.pow(U[i][k], self.m)
                    den += math.pow(U[i][k], self.m)
                V[i][dim] = num / den
        return V

    def update_memberships(self, X, V, U):
        for k in range(len(X)):
            for i in range(self.c):
                sum = self.compute_sum(X[k], V, i)
                if sum != 0:
                    U[i][k] = math.pow(sum, -1)
                else:
                    if np.array_equal(V[i], X[k]):
                        U[i][k] = 1
                    else:
                        U[i][k] = 0

        return U

    def compute_sum(self, xk, V, i):
        sum = 0
        for j in range(self.c):
            num = np.linalg.norm(xk - V[i])
            den = np.linalg.norm(xk - V[j])
            if den == 0:
                return 0
            sum += math.pow(num / den, 2 / (self.m - 1))
        return sum

    def fcm(self, data):

        U = np.zeros((self.c, len(data)))
        V = self.guess_initial_centers(data)

        iter = 0

        while iter < self.max_iter:

            V_aux = np.copy(V)
            iter += 1
            U = self.update_memberships(data, V, U)
            V = self.calculate_centers(data, U, V)

            if np.linalg.norm(V - V_aux) <= self.epsilon:
                self.U = U
                self.V = V
                return V

        self.U = U
        self.V = V

    def cluster_matching(self, X):
        matches = np.zeros(len(X))
        for i in range(len(X)):
            print(self.U[:, i])
            matches[i] = np.argmax(self.U[:, i])
        return matches