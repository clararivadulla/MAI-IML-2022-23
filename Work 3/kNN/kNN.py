import numpy as np
from metrics.distance_metrics import euclidean, minkowski, cosine

class kNN:
    def __init__(self, k=1, dist_metric='euclidean', r=2):
        self.k = k
        self.dist_metric = dist_metric
        self.r = r

    def get_neighbors(self, train_data, test_data_row):
        distances = []
        for row in train_data:
            if self.dist_metric == 'euclidean':
                distance = euclidean(test_data_row, row[:len(row) - 1])
            elif self.dist_metric == 'minkowski':
                distance = minkowski(test_data_row, row[:len(row) - 1], self.r)
            elif self.dist_metric == 'cosine':
                distance = cosine(test_data_row, row[:len(row) - 1])
            distances.append((row, distance))
        distances.sort(key=lambda x: x[1])
        return [distances[i][0] for i in range(self.k)]

    def predict(self, train_data, test_data_row):
        neighbors = self.get_neighbors(train_data, test_data_row)
        labels = [neighbor[-1] for neighbor in neighbors]
        prediction = max(set(labels), key=labels.count)
        return int(prediction)
