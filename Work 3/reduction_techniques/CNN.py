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

    def reduce_rnn(self, x_train, y_train, numeric_cols, nominal_cols):
        length = len(x_train)
        idx_rnn = [i for i in range(0, length)]

        for i in range(0, length):
            #print(i)
            kNN_config = kNN(k=self.k, dist_metric=self.dist_metric, r=self.r, weights=self.weights)
            trial_idx_rnn = []
            for z in idx_rnn:
                if z != i:
                    trial_idx_rnn.append(z)
            kNN_config.fit(x_train[trial_idx_rnn], y_train[trial_idx_rnn], numeric_cols=numeric_cols, nominal_cols=nominal_cols)

            error = False
            for j in range(len(x_train)):
                y_hat = kNN_config.predict(x_train[[j]])
                if y_hat != y_train[j]:
                    error = True
                    break

            if not error:
                idx_rnn = trial_idx_rnn

        # kNN_config = kNN()
        # kNN_config.fit(x_train[idx_rnn], y_train[idx_rnn])
        #
        # part_predictions = []
        # for j in range(len(x_train)):
        #     prediction = kNN_config.predict(x_train[j, :])
        #     part_predictions.append(prediction)
        #
        # print(sum(part_predictions==y_train))

        return x_train[idx_rnn], y_train[idx_rnn]


    def reduce_cnn(self, x_train, y_train, numeric_cols, nominal_cols):
        idx_cnn = [0]
        errors = True
        while errors:

            errors = False

            kNN_config = kNN(k=self.k, dist_metric=self.dist_metric, r=self.r, weights=self.weights)
            kNN_config.fit(x_train[idx_cnn], y_train[idx_cnn])

            for i in range(len(x_train)):
                #y_hat = kNN_config.get_neighbors(x_train[[i]])[0]['label'].values[0]
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
