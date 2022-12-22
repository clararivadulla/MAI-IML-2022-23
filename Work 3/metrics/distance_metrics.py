import numpy as np

def cosine(a, b, nom_cols=None, num_cols=None):

    nom_dist = 0
    num_dist = 1 - (np.dot(a[:, num_cols], b[num_cols]) / (
                np.linalg.norm(a[:, num_cols], axis=1) * np.linalg.norm(b[num_cols])))
    if nom_cols != None:
        # Calculate share of nominal variables where 2 entries are the same, on scale from 0 to 1
        nom_dist = np.sum(a[:,nom_cols] == b[nom_cols], axis=1) / len(nom_cols)
    return nom_dist + num_dist

    # Check if there is a zero vector and add 1e-5 to be able to calculate the distance
    '''a_zero_vector = np.where(~a.any(axis=1))[0]
    b_zero_vector = np.where(~b.any())[0]
    if len(a_zero_vector)>0:
        a[a_zero_vector,:] += 1e-5
    if len(b_zero_vector)>0:
        b += 1e-5'''

def minkowski(a, b, r=2, nom_cols=None, num_cols=None):
    nom_dist = 0
    b = b.ravel()
    num_dist = np.sum(np.abs(a[:, num_cols] - b[num_cols]) ** r, axis=1) ** (1 / r)
    if nom_cols != None:
        # Calculate share of nominal variables where 2 entries are the same, on scale from 0 to 1
        nom_dist = np.sum(a[:,nom_cols] == b[nom_cols], axis=1) / len(nom_cols)
    return nom_dist + num_dist


def clark(a,b, nom_cols=None, num_cols=None):
    nom_dist = 0
    c = a[:, num_cols] + b[num_cols]
    c[np.where(c == 0)] = 1e-5
    num_dist = np.sum((a[:, num_cols] - b[num_cols]) ** 2 / (c ** 2), axis=1)
    if nom_cols != None:
        # Calculate share of nominal variables where 2 entries are the same, on scale from 0 to 1
        nom_dist = np.sum(a[:,nom_cols] == b[nom_cols], axis=1) / len(nom_cols)
    return num_dist + nom_dist
