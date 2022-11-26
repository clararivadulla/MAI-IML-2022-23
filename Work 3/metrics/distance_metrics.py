import numpy as np
import pandas as pd

# how do we deal with categorical values?

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))

def minkowski(a, b, r=2):
    return np.sum(np.abs(a - b) ** r, axis=1) ** (1 / r)
