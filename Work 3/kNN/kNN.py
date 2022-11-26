import numpy as np
from metrics.distance_metrics import euclidean, minkowski, cosine

class kNN:
    def __init__(self, k=1, dist_metric='minkowski', r=2, voting='majority', weights=None):
        self.k = k
        self.dist_metric = dist_metric
        self.r = r
        self.voting = voting
        self.weights = weights

    def get_neighbors(self, x_train, y_train, x_test):
        x_train = pd.DataFrame(x_train)
        if self.dist_metric == 'minkowski':
            distance = minkowski(x_train, x_test, self.r)
        elif self.dist_metric == 'cosine':
            distance = cosine(x_train, x_test)
        x_train['label'] = y_train
        x_train['distance'] = distance
        x_train.sort_values(by=['distance'])
        return x_train.iloc[:self.k,:-1]

    def predict(self, x_train, y_train, x_test):
        # how do we deal with even number of votes?
        # different voting schemes
        neighbors = self.get_neighbors(x_train, y_train, x_test)
        labels = neighbors['label']
        prediction = max(set(labels), key=labels.count)
        return int(prediction)

