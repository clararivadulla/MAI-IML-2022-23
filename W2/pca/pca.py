import numpy as np


def pca(X, k):

    X_mean = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_mean, rowvar = False)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvectors = eigenvectors[:, idx]
    k_eigenvectors = sorted_eigenvectors[:, 0:k]
    subspace = k_eigenvectors.T.dot(X_mean.T).T

    return subspace
