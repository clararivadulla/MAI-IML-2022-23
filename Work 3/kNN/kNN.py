import numpy as np


class kNN:
    def __init__(self, k=1):
        self.k = k

    def get_neighbors(self, train_data, test_data_row):
        distances = []
        for row in train_data:
            distance = np.linalg.norm(test_data_row - row[:len(row) - 1])
            distances.append((row, distance))
        distances.sort(key=lambda x: x[1])
        return [distances[i][0] for i in range(self.k)]

    def predict(self, train_data, test_data_row):
        neighbors = self.get_neighbors(train_data, test_data_row)
        labels = [neighbor[-1] for neighbor in neighbors]
        prediction = max(set(labels), key=labels.count)
        return int(prediction)
