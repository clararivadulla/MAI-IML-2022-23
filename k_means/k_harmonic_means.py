import numpy as np


class KHarmonicMeans:

    def __init__(self, n_clusters=3, max_iter=100):
        self.k = n_clusters
        self.max_iter = max_iter
        self.p = 3
        self.epsilon = np.finfo(np.float32).eps
        self.centroids = None

    def cluster_matching(self, X):
        matches = np.zeros(np.size(X[:, 0]))
        for i in range(np.size(X[:, 0])):
            norm = np.linalg.norm(X[i] - self.centroids[0])
            for j in range(1, self.k):
                aux_norm = np.linalg.norm(X[i] - self.centroids[j])
                if aux_norm < norm:
                    norm = aux_norm
                    matches[i] = j
        return matches

    # Performance function
    def performance(self, X, C):

        p = 3
        sum1 = 0

        for i in range(np.size(X[:, 0])):
            sum2 = 0
            for j in range(self.k):
                sum2 += (1 / np.power(np.max([np.linalg.norm(X[i] - X[j]), self.epsilon]), p))
            sum1 += (self.k / sum2)

        return sum1

    # Membership function
    def membership(self, C, cj, xi):

        sum = 0

        for j in range(self.k):
            # The implementation of KHM needs to deal with the case where xi = cj
            sum += np.power(np.max([np.linalg.norm(xi - C[j]), self.epsilon]), -self.p - 2)

        return np.power(np.max([np.linalg.norm(xi - cj), self.epsilon]), -self.p - 2) / sum

    # Weight function
    def weight(self, C, xi):

        sum1 = 0
        sum2 = 0

        for j in range(self.k):
            sum1 += np.power(np.max([np.linalg.norm(xi - C[j]), self.epsilon]), -self.p - 2)
            sum2 += np.power(np.max([np.linalg.norm(xi - C[j]), self.epsilon]), -self.p)

        return sum1 / np.power(sum2, 2)

    def cj(self, X, C, cj):

        sum1 = 0
        sum2 = 0

        # For each data point xi, compute its membership m(cj jxi) in each center cj and its weight w(xi)
        for i in range(np.size(X[:, 0])):
            sum1 += self.membership(C, cj, X[i]) * self.weight(C, X[i]) * X[i]
            sum2 += self.membership(C, cj, X[i]) * self.weight(C, X[i])

        return sum1 / sum2

    def guess_centers(self, X, k):

        centers = []
        indices = np.random.choice(np.size(X[:, 0]), k, replace=False)

        for i in range(k):
            centers.append(X[indices[i]])

        return centers

    def khm(self, data):

        X = data.copy()

        # 1. Initialize the algorithm with guessed centers C
        self.centroids = self.guess_centers(X, self.k)
        iter = 0

        # 4. Repeat steps 2 and 3 till convergence
        while iter < self.max_iter:

            C_aux =  self.centroids.copy()
            iter += 1

            # 3. For each center cj, recompute its location from all data points xi,
            # according to their membership and weights
            for j in range(self.k):
                self.centroids[j] = self.cj(X,  self.centroids,  self.centroids[j])

            if np.array_equal(C_aux, self.centroids):
                return self.centroids