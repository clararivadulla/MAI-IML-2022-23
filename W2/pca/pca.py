import numpy as np


def pca(X, k):

    # Compute the d-dimensional mean vector
    X_mean = X - np.mean(X, axis=0)

    # Compute the covariance matrix of the whole data set
    covariance_matrix = np.cov(X_mean, rowvar = False)

    # Calculate eigenvectors (e1, e2, …, ed) and their corresponding eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvectors by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Choose k eigenvectors with the largest eigenvalues to form a new d x k dimensional matrix
    k_eigenvectors = sorted_eigenvectors[:, 0:k]

    # Derive the new data set. Use this d x k eigenvector matrix to transform the samples onto the new subspace
    subspace = k_eigenvectors.T.dot(X_mean.T).T

    # Plot the new subspace
    print('PCA subspace:')
    print(subspace)

    return subspace
