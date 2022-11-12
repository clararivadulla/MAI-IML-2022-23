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

    total_var = sum(eigenvalues)
    explained_variance_i = [(i / total_var) for i in eigenvalues]
    explained_variance_cum = np.cumsum(explained_variance_i)
    if k>0 and k<1:
        for e in range(len(eigenvalues)):
            if explained_variance_cum[e] >= k:
                k=e+1
                break


    print(f'Total explained variance with {k} components: ', sum(explained_variance_i[0:k]))

    # Choose k eigenvectors with the largest eigenvalues to form a new d x k dimensional matrix
    k_eigenvectors = sorted_eigenvectors[:, 0:k]

    # Derive the new data set. Use this d x k eigenvector matrix to transform the samples onto the new subspace
    transformed_data = X_mean.dot(k_eigenvectors)

    # Plot the transformed data in the new subspace
    #print('PCA subspace:')
    #print(subspace)

    reconstructed_data = transformed_data.dot(k_eigenvectors.T) + means

    return transformed_data, reconstructed_data, k, sum(explained_variance_i[0:k])
