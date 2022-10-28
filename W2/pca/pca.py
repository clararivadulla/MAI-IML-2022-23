import numpy as np
import pandas as pd

"""
- Step 1. Read the .arff file and take the whole data set consisting of d-dimensional samples
ignoring the class labels. Save the information in a matrix.
- Step 2. Plot the original data set (choose two or three of its features to visualize it).
- Step 3. Compute the d-dimensional mean vector (i.e., the means of every dimension of the
whole data set).
- Step 4. Compute the covariance matrix of the whole data set. Show this information.
- Step 5. Calculate eigenvectors (e1, e2, …, ed) and their corresponding eigenvalues of the
covariance matrix. Use numpy library. Write them in console.
- Step 6. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the
largest eigenvalues to form a new d x k dimensional matrix (where every column
represents an eigenvector). Write the sorted eigenvectors and eigenvalues in console.
- Step 7. Derive the new data set. Use this d x k eigenvector matrix to transform the samples
onto the new subspace.
- Step 8. Plot the new subspace (choose the largest eigenvectors to plot the matrix).
- Step 9. Reconstruct the data set back to the original one. Additionally, plot the data set.
"""

