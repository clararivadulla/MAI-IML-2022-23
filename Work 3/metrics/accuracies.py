import pandas as pd
from scipy.stats import ttest_rel
import numpy as np

def accuracy(labels, predictions):
    df = {'true labels':labels, 'predictions':predictions}
    df = pd.DataFrame(df)
    correct = sum(df['true labels']==df['predictions'])
    incorrect = len(labels) - correct
    share = (correct / len(labels))
    return correct, incorrect, share

def t_tests(acc1, acc2, acc3):
    matrix = np.ones((len(acc1),len(acc1)))
    results = []
    for i in range(len(acc1)):
        set1 = [acc1[i], acc2[i], acc3[i]]
        for j in range(i+1,len(acc1)):
            set2 = [acc1[j], acc2[j], acc3[j]]
            p=ttest_rel(set1,set2)[1]
            matrix[i,j] = p
            if p<0.05:
                results.append([i,j,set1, set2,p])
    return matrix, results



