import numpy as np
import pandas as pd

from metrics.distance_metrics import minkowski, cosine, clark
from sklearn.feature_selection import mutual_info_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression

class kNN:
    def __init__(self, k=1, dist_metric='minkowski', r=2, voting='majority', weights="uniform"):
        self.k = k
        self.dist_metric = dist_metric
        self.r = r
        self.voting = voting
        self.weights = weights
        self.x_train = None
        self.y_train = None
        self.w = None
        self.nominal = None
        self.numerical = None

    def fit(self, x_train, y_train, numeric_cols, nominal_cols):
        x_train = x_train.copy()
        self.numerical = numeric_cols
        self.nominal = nominal_cols

        if self.weights == 'lasso':
            lasso = SelectFromModel(LogisticRegression(penalty="l2", max_iter=500), max_features=None) # L2: Ridge Regression
            lasso.fit(x_train, y_train)
            self.w = lasso.get_support()
            x_train *= self.w

        elif self.weights == 'mutual_info_score':
            mic_w = mutual_info_classif(x_train, y_train, n_neighbors=self.k)
            x_train *= mic_w
            self.w = mic_w

        elif self.weights == 'uniform':
            self.w = np.ones(x_train.shape[1])

        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        neighbors, labels, distance = self.get_neighbors(x_test)
        # labels = neighbors['label']
        labels = pd.Series(labels)

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
                votes_list = []
                for i in range(len(cat)):
                    vote = sum(labels[labels==cat[i]]*distance[labels==cat[i]])
                    votes_list.append(vote)
                votes_sorted = pd.DataFrame(data={'category': cat, 'votes': votes_list})
                votes_sorted.sort_values(by=['votes'], inplace=True)
                if votes_sorted.shape[0] > 1:
                    if votes_sorted['category'][0] != votes_sorted['category'][1]:
                        return int(votes_sorted['category'][0])
                else:
                    return int(votes_sorted['category'][0])
        elif self.voting == 'sheppard':
            while True:
                cat = list(set(labels))
                votes_list = []
                for i in range(len(cat)):
                    vote = sum(labels[labels==cat[i]]*np.exp(-distance[labels==cat[i]]))
                    votes_list.append(vote)
                votes_sorted = pd.DataFrame(data={'category': cat, 'votes': votes_list})
                votes_sorted.sort_values(by=['votes'], inplace=True)
                if votes_sorted.shape[0] > 1:
                    if votes_sorted['category'][0] != votes_sorted['category'][1]:
                        return int(votes_sorted['category'][0])
                else:
                    return int(votes_sorted['category'][0])
        else:
            raise Exception("Voting scheme is not recognized")


    def get_neighbors(self, x_test):

        x_train = self.x_train.copy()
        y_train = self.y_train.copy()
        x_test = x_test.copy()
        x_test *= self.w

        if self.dist_metric == 'minkowski':
            distance = minkowski(x_train, x_test, self.r, self.nominal, self.numerical)
        elif self.dist_metric == 'cosine':
            distance = cosine(x_train, x_test, self.nominal, self.numerical)
        elif self.dist_metric == 'clark':
            distance = clark(x_train, x_test, self.nominal, self.numerical)
        else:
            raise Exception("Distance matrix is not recognized")

        distance_sorted_idx = np.argsort(distance)
        k_distance_sorted_idx = distance_sorted_idx[:self.k]
        return x_train[k_distance_sorted_idx], y_train[k_distance_sorted_idx], distance[k_distance_sorted_idx]

