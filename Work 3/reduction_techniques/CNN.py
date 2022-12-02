import numpy as np

from kNN.kNN import kNN

class CNN:
    def __init__(self, k=1, dist_metric='minkowski', r=2, weights="uniform"):
        self.k = k
        self.dist_metric = dist_metric
        self.r = r
        self.weights = weights
        self.x_train = None
        self.y_train = None

    def reduce_rnn(self, x_train, y_train):
        x_train_cnn = x_train.copy()

    def reduce_cnn(self, x_train, y_train):

        idx_cnn = [0]
        errors = True
        while errors:
            errors = False

            kNN_config = kNN(k=self.k, dist_metric=self.dist_metric, r=self.r, weights=self.weights)
            kNN_config.fit(x_train[idx_cnn], y_train[idx_cnn])

            for i in range(len(x_train)):
                y_hat = kNN_config.predict(x_train[[i]])
                if y_hat != y_train[i]:
                    idx_cnn.append(i)
                    errors = True
                    break

        #kNN_config = kNN()
        #kNN_config.fit(x_train[idx_cnn], y_train[idx_cnn])

        #part_predictions = []
        #for j in range(len(x_train)):
        #    prediction = kNN_config.predict(x_train[j, :])
        #    part_predictions.append(prediction)

        #print(sum(part_predictions==y_train))

        return x_train[idx_cnn], y_train[idx_cnn]
