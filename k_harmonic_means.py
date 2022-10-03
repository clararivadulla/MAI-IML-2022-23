import numpy as np

''' 
This algorithm is based on the formulae given in the 
"k-Harmonic Means Clustering Algorithm" paper by Henrico Dolfing
'''


class KHarmonicMeans:

    def __init__(self, n_clusters=3):
        self.k = n_clusters
        self.p = 3
        self.epsilon = np.finfo(np.float32).eps

    # Performance function
    def performance(self, X, C):

        p = 3
        sum1 = 0

        for i in range(len(X)):
            sum2 = 0
            for j in range(len(C)):
                sum2 += (1 / np.power(np.linalg.norm(X[i] - X[j]), p))
            sum1 += (len(C) / sum2)

        return sum1

    # Membership function
    def membership(self, C, cj, xi):

        sum = 0

        for j in range(len(C)):
            # The implementation of KHM needs to deal with the case where xi = cj
            sum += np.power(np.max([np.linalg.norm(xi - C[j]), self.epsilon]), -self.p - 2)

        return np.power(np.max([np.linalg.norm(xi - cj), self.epsilon]), -self.p - 2) / sum

    # Weight function
    def weight(self, C, xi):

        sum1 = 0
        sum2 = 0

        for j in range(len(C)):
            sum1 += np.power(np.max([np.linalg.norm(xi - C[j]), self.epsilon]), -self.p - 2)
            sum2 += np.power(np.max([np.linalg.norm(xi - C[j]), self.epsilon]), -self.p)

        return sum1 / np.power(sum2, 2)

    def cj(self, X, C, cj):

        sum1 = 0
        sum2 = 0

        # For each data point xi, compute its membership m(cj jxi) in each center cj and its weight w(xi)
        for i in range(len(X)):
            sum1 += self.membership(C, cj, X[i]) * self.weight(C, X[i]) * X[i]
            sum2 += self.membership(C, cj, X[i]) * self.weight(C, X[i])

        return sum1 / sum2

    def guess_centers(self, X, k):

        centers = []
        indices = np.random.choice(len(X), k, replace=False)

        for i in range(k):
            centers.append(X[indices[i]])

        return centers

    def khm(self, data, k=3):

        X = data.copy()

        # 1. Initialize the algorithm with guessed centers C
        C = self.guess_centers(X, k)

        # 4. Repeat steps 2 and 3 till convergence
        while True:

            C_aux = C

            # 3. For each center cj , recompute its location from all data points xi,
            # according to their membership and weights
            for j in range(len(C)):
                C[j] = self.cj(X, C, C[j])

            if np.array_equal(C_aux, C):
                return C
