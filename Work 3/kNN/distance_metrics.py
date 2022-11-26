import numpy as np

# how do we deal with categorical values? Is there a need to deal with it?

def cosine(a, b):
    # Check if there is a zero vector and add 1e-5 to be able to calculate the distance
    a_zero_vector = np.where(~a.any(axis=1))[0]
    b_zero_vector = np.where(~b.any())[0]
    if len(a_zero_vector)>0:
        a.iloc[a_zero_vector,:] += 1e-5
    if len(b_zero_vector)>0:
        b += 1e-5
    return np.dot(a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))

def minkowski(a, b, r=2):
    return np.sum(np.abs(a - b) ** r, axis=1) ** (1 / r)

def clark(a,b):
    c=a+b
    c[c==0] += 1e-5
    return np.sum((a-b)**2/(c**2), axis=1)
