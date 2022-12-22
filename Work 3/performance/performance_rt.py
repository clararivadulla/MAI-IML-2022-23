import timeit
from kNN.kNN import kNN
from metrics.accuracies import accuracy
from reduction_techniques.RNN import RNN
from reduction_techniques.DROP3 import DROP3
from reduction_techniques.EENTh import EENTh


def reduce(data, reduction_technique, dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols, nominal_cols,
        verbose=False):
    print(
        f'\n{reduction_technique} with {dataset_name} k={k}, dist_metric={distance_metric}, voting={voting_scheme}, weights={weighting_scheme}')

    times = []
    accuracies = []
    i = 0
    for d in data:
        print(f"Fold {i} ···········································")
        i += 1
        x_train, y_train, x_test, y_test = d
        x_test_len = len(x_test)
        part_predictions = []
        start = timeit.default_timer()
        if reduction_technique == 'RNN':
            RNN_config = RNN(k=k, dist_metric=distance_metric, weights=weighting_scheme)
            reduced_x, reduced_y = RNN_config.reduce(x_train, y_train, numeric_cols, nominal_cols)
        elif reduction_technique == 'DROP3':
            DROP3_config = DROP3(k=k, dist_metric=distance_metric, weights=weighting_scheme)
            reduced_x, reduced_y = DROP3_config.reduce_drop3(x_train, y_train, numeric_cols, nominal_cols)
        elif reduction_technique == 'EENTh':
            EENTh_config = EENTh(k=k, dist_metric=distance_metric, weights=weighting_scheme)
            reduced_x, reduced_y = EENTh_config.reduce(x_train, y_train, numeric_cols, nominal_cols)
        kNN_config = kNN(k=k, dist_metric=distance_metric, voting=voting_scheme, weights=weighting_scheme)
        kNN_config.fit(reduced_x, reduced_y, numeric_cols=numeric_cols, nominal_cols=nominal_cols)

        for j in range(x_test_len):
            prediction = kNN_config.predict(x_test[j, :])
            part_predictions.append(prediction)

        stop = timeit.default_timer()
        time = stop - start
        times.append(time)

        correct, incorrect, acc = accuracy(y_test, part_predictions)
        if verbose:
            print(
                f'Correct: {correct}, Incorrect: {incorrect}, Accuracy: {round(acc * 100, 2)}%, Time: {round(time, 2)}s')
        accuracies.append(acc)

    avg_time = sum(times) / len(times)
    avg_acc = sum(accuracies) / len(accuracies)
    print(f'\nAverage accuracy: {round(avg_acc * 100, 2)}%, Average time: {round(avg_time, 2)}s')
    return ([round(avg_acc * 100, 2), round(avg_time, 2)], [reduction_technique, k, distance_metric, voting_scheme, weighting_scheme])

def reduction_techniques(data, dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols, nominal_cols, verbose=False):

    results = []

    #rnn_results = reduce(data, 'RNN', dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols, nominal_cols, verbose=False)
    eenth_results = reduce(data, 'EENTh', dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols, nominal_cols, verbose=False)
    drop3_results = reduce(data, 'DROP3', dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols, nominal_cols, verbose=False)

    #results.append(rnn_results)
    results.append(eenth_results)
    results.append(drop3_results)

    print(results)

    results_sorted_by_accuracy = sorted(results, key=lambda item: item[0][0], reverse=True)
    return results_sorted_by_accuracy
