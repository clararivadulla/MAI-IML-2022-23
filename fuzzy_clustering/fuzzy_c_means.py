import numpy as np


class FuzzyCMeans:

    def __init__(self, n_clusters=3, max_iter=100):
        self.c = n_clusters
        self.max_iter = max_iter
        self.epsilon = np.finfo(np.float32).eps
        self.m = 2
        self.U = None
        self.V = None

    def initialize_memberships(self, X):
        return np.zeros((self.c, np.size(X[:, 0]))) # TODO

    def centers(self, X):
        return np.zeros((self.c, np.size(X[:, 0]))) # TODO

    def update_memberships(self, X):
        sum = 0
        # for i in range(np.size(X[:, 0])):
            # for j in range(self.C):
        return np.zeros((self.c, np.size(X[:, 0]))) # TODO

    def fcm(self, data):

        self.U = self.initialize_memberships(data)

        iter = 0

        while iter < self.max_iter:
            V_t = self.centers(data)
            U_t = self.update_memberships(data)
            iter += 1

        if np.linalg.norm(self.V - V_t) <= self.epsilon:
            self.V = V_t
            self.U = U_t

        return self.V
