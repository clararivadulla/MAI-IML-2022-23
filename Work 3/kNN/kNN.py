import numpy as np
import pandas as pd
from metrics.distance_metrics import minkowski, cosine, clark
import sklearn_relief
from sklearn.feature_selection import mutual_info_classif

class kNN:
    def __init__(self, k=1, dist_metric='minkowski', r=2, voting='majority', weights="uniform"):
        self.k = k
        self.dist_metric = dist_metric
        self.r = r
        self.voting = voting
        self.weights = weights

    def get_neighbors(self, x_train, y_train, x_test):
        x_train = pd.DataFrame(x_train)
        n_cat = len(set(y_train))

        if self.weights == 'relief':
            if n_cat == 2:
                r = sklearn_relief.Relief(n_features=x_train.shape[1])
                x_train = r.fit_transform(x_train, y_train)
            else:
                r = sklearn_relief.ReliefF(n_features=x_train.shape[1])
                x_train = r.fit_transform(x_train, y_train)
        elif self.weights == 'mutual_info_score':
            mic_w = mutual_info_classif(x_train, y_train, n_neighbors=self.k)
            x_train *= mic_w

        if self.dist_metric == 'minkowski':
            distance = minkowski(x_train, x_test, self.r)
        elif self.dist_metric == 'cosine':
            distance = cosine(x_train, x_test)
        elif self.dist_metric == 'clark':
            distance = clark(x_train, x_test)
        else:
            raise Exception("Distance matrix is not recognized")
        x_train['label'] = y_train
        x_train['distance'] = distance
        x_train.sort_values(by=['distance'], inplace=True)
        # Returns k sorted training data points with labels and their distance from the x_test
        return x_train.iloc[:self.k,:-1], x_train.iloc[:self.k,-1]

    def predict(self, x_train, y_train, x_test):
        neighbors, distance = self.get_neighbors(x_train, y_train, x_test)
        labels = neighbors['label']
        if self.voting == 'majority':
            votes_counted = labels.value_counts()
            if votes_counted.shape[0] > 1:
                # Keep reducing k until you break the tie
                while votes_counted.index[0] == votes_counted.index[1]:
                    labels = labels.iloc[:-1]
                    votes_counted = labels.value_counts()
            return int(votes_counted.index[0])
        elif self.voting == 'inverse_distance':
            while True:
                cat = list(set(labels))
                votes = {'category':cat, 'votes':np.zeros(len(cat))}
                votes = pd.DataFrame(data=votes)
                for i in range(len(cat)):
                    votes.iloc[i,1] = sum(labels[labels==cat[i]]*distance[labels==cat[i]])
                votes_sorted = votes.value_counts()
                if votes_sorted.shape[0] > 1:
                    if votes_sorted.index[0] != votes_sorted.index[1]:
                        return int(votes_sorted.index[0])
                else:
                    return int(votes_sorted.index[0])
        elif self.voting == 'sheppard':
            while True:
                cat = list(set(labels))
                votes = {'category':cat, 'votes':np.zeros(len(cat))}
                votes = pd.DataFrame(data=votes)
                for i in range(len(cat)):
                    votes.iloc[i,1] = sum(labels[labels==cat[i]]*np.exp(-distance[labels==cat[i]]))
                votes_sorted = votes.value_counts()
                if votes_sorted.shape[0] > 1:
                    if votes_sorted.index[0] != votes_sorted.index[1]:
                        return int(votes_sorted.index[0])
                else:
                    return int(votes_sorted.index[0])
        else:
            raise Exception("Voting scheme is not recognized")






