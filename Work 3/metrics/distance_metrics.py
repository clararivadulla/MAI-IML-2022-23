import numpy as np

def transform_scale(a):
    max = 1.
    new_a = ( a * 2 ) + 1
    dif = max - new_a
    if dif % 0.5 == 0:
        new_a = dif + 1
    else:
        new_a = new_a + (dif*2)
    return new_a

def cosine(a, b, nom_cols=None, num_cols=None):

    nom_dist = 0
    num_dist = 1 - (np.dot(a.iloc[:, num_cols], b[num_cols]) / (
                np.linalg.norm(a.iloc[:, num_cols], axis=1) * np.linalg.norm(b[num_cols])))
    if nom_cols != None:
        # Calculate share of nominal variables where 2 entries are the same, on scale from 0 to 1
        nom_dist = np.sum(a.iloc[:,nom_cols] == b[nom_cols], axis=1) / len(nom_cols)
        # Convert nominal distance to the scale of the output of cosine function, -1 to 1
#        nom_dist = nom_dist*2-1
        nom_dist = transform_scale(nom_dist)
    return nom_dist + num_dist

    # Check if there is a zero vector and add 1e-5 to be able to calculate the distance
    '''a_zero_vector = np.where(~a.any(axis=1))[0]
    b_zero_vector = np.where(~b.any())[0]
    if len(a_zero_vector)>0:
        a.iloc[a_zero_vector,:] += 1e-5
    if len(b_zero_vector)>0:
        b += 1e-5'''

def minkowski(a, b, r=2, nom_cols=None, num_cols=None):
    nom_dist = 0
    num_dist = np.sum(np.abs(a.iloc[:, num_cols] - b[num_cols]) ** r, axis=1) ** (1 / r)
    if nom_cols != None:
        # Calculate share of nominal variables where 2 entries are the same, on scale from 0 to 1
        nom_dist = np.sum(a.iloc[:,nom_cols] == b[nom_cols], axis=1) / len(nom_cols)
    return nom_dist + num_dist


def clark(a,b, nom_cols=None, num_cols=None):
    '''c=a+b
    c[c==0] += 1e-5'''
    nom_dist = 0
    c = a.iloc[:, num_cols] + b[num_cols]
    num_dist = np.sum((a.iloc[:, num_cols] - b[num_cols]) ** 2 / (c ** 2), axis=1)
    if nom_cols != None:
        # Calculate share of nominal variables where 2 entries are the same, on scale from 0 to 1
        nom_dist = np.sum(a.iloc[:,nom_cols] == b[nom_cols], axis=1) / len(nom_cols)
    return num_dist + nom_dist
