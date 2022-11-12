import numpy as np


def pca(X, k):

    # Compute the d-dimensional mean vector
    means = np.mean(X, axis=0)
    X_mean = X - means

    # Compute the covariance matrix of the whole data set
    covariance_matrix = np.cov(X_mean, rowvar = False)

    # Calculate eigenvectors (e1, e2, …, ed) and their corresponding eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    #print('Eigenvectors:')
    #print(eigenvectors)

    #print('\nEigenvalues:')
    #print(eigenvalues)

    # Sort the eigenvectors by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = np.sort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Choose k eigenvectors with the largest eigenvalues to form a new d x k dimensional matrix
    k_eigenvectors = sorted_eigenvectors[:, 0:k]

    # Derive the new data set. Use this d x k eigenvector matrix to transform the samples onto the new subspace
    subspace = X_mean.dot(k_eigenvectors)

    # Plot the new subspace
    #print('PCA subspace:')
    #print(subspace)

    total_var = sum(eigenvalues)
    explained_variance = [(i / total_var) for i in eigenvalues]
    print(f'Total explained variance with {k} components: ', sum(explained_variance[0:k]))

    reconstructed_data = subspace.dot(k_eigenvectors.T) + means

    return subspace, reconstructed_data
