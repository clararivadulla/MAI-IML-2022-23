import numpy as np

p = 3
epsilon = np.finfo(np.float32).eps

# Performance function
def performance(X, C):
    p = 3
    sum1 = 0
    for i in range(len(X)):
        sum2 = 0
        for j in range(len(C)):
            if np.array_equal(X[i], C[j]):
                distance = np.power(epsilon, p)
            else:
                distance = np.power(np.linalg(X[i], X[j]), p)
            sum2 += (1 / distance)
        sum1 += (len(C) / sum2)
    return sum1

# Membership function
def membership(C, cj, xi):
    distance = np.power(np.linalg(xi - cj), -p-2)
    sum = 0
    for j in range(len(C)):
        sum += np.power(np.linalg(xi - C[j]), -p-2)
    return distance / sum

# Weight function
def weight(C, xi):
    sum1 = 0
    sum2 = 0
    for j in range(len(C)):
        sum1 += np.power(np.linalg(xi - C[j]), -p-2)
        sum2 += np.power(np.linalg(xi - C[j]), -p)
    return sum1 / np.power(sum2, 2)

def cj(X, C, cj):
    sum1 = 0
    sum2 = 0
    for i in range(len(X)):
        sum1 += membership(C, cj, X[i]) * weight(C, X[i]) * X[i]
        sum2 += membership(C, cj, X[i]) * weight(C, X[i])
    return sum1 / sum2

def khm(data, k=3):
    X = data.copy()
