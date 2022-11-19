import numpy as np
import pandas as pd

# how do we deal with categorical values?

def euclidean(a, b):
    return np.linalg.norm(a - b)

def euclidean_v2(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))


def minkowski(a, b, r):
    return np.sum(np.abs(a - b) ** r, axis=1) ** (1 / r)
