import numpy as np


def euclidean(a, b):
    return np.linalg.norm(a - b)


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def minkowski(a, b, r):
    return np.sum(np.abs(a - b) ** r) ** (1 / r)
