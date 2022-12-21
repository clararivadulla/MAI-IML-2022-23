import timeit
from kNN.kNN import kNN
from metrics.accuracies import accuracy
<<<<<<< Updated upstream
from reduction_techniques.CNN import CNN
=======
<<<<<<< Updated upstream
>>>>>>> Stashed changes
from reduction_techniques.EENTh import EENTh


def cnn(data, dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols, nominal_cols,
        verbose=False):
    print(
        f'\nCNN with {dataset_name} k={k}, dist_metric={distance_metric}, voting={voting_scheme}, weights={weighting_scheme}')

    times = []
    accuracies = []
    i = 0
    for d in data:
        i += 1
        print(f"{i}, ", end=' ')
        x_train, y_train, x_test, y_test = d
        x_test_len = len(x_test)
        part_predictions = []
        start = timeit.default_timer()
        CNN_config = CNN(k=k, dist_metric=distance_metric, weights=weighting_scheme)
        reduced_x, reduced_y = CNN_config.reduce_rnn(x_train, y_train, numeric_cols, nominal_cols)
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
    return ([round(avg_acc * 100, 2), round(avg_time, 2)], ['CNN', k, distance_metric, voting_scheme, weighting_scheme])


def eenth(data, dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols, nominal_cols,
          verbose=False):
    print(
        f'\nEENTh with {dataset_name} k={k}, dist_metric={distance_metric}, voting={voting_scheme}, weights={weighting_scheme}')

    times = []
    accuracies = []
    i = 0
    for d in data:
        i += 1
        print(f"{i}, ", end=' ')
        x_train, y_train, x_test, y_test = d
        x_test_len = len(x_test)
        part_predictions = []
        start = timeit.default_timer()
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
    return ([round(avg_acc * 100, 2), round(avg_time, 2)], ['EENTh', k, distance_metric, voting_scheme, weighting_scheme])


def reduction_techniques(data, dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols,
                         nominal_cols, verbose=False):
    results = []
    # ccn_results = cnn(data, dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols, nominal_cols, verbose=False)
    eenth_results = eenth(data, dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols, nominal_cols, verbose=False)
    # drop3_results =

    # results.append(ccn_results)
    results.append(eenth_results)
    print(results)

    results_sorted_by_accuracy = sorted(results, key=lambda item: item[0][0], reverse=True)
    return results_sorted_by_accuracy


"""
def test_performance_rt(data, dataset_name='', verbose=False):

    ks = [1, 3, 5, 7]
    voting_schemes = ['majority', 'inverse_distance', 'sheppard']
    distance_metrics = ['minkowski', 'cosine', 'clark']
    weighting_schemes = ['uniform', 'mutual_info_score']

    results = []
    i = 0
    for k in ks:
        for voting_scheme in voting_schemes:
            for distance_metric in distance_metrics:
                for weighting_scheme in weighting_schemes:
                    i += 1
                    print(f'\nTEST {i}: k={k}, dist_metric={distance_metric}, voting={voting_scheme}, weights={weighting_scheme}')

                    times = []
                    accuracies = []
=======
from reduction_techniques.CNN import CNN
from reduction_techniques.DROP3 import DROP3
from reduction_techniques.EENTh import EENTh


def reduce(data, reduction_technique, dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols,
           nominal_cols,
           verbose=False):
    print(
        f'\n{reduction_technique} with {dataset_name} k={k}, dist_metric={distance_metric}, voting={voting_scheme}, weights={weighting_scheme}')

    times = []
    accuracies = []
    i = 0
    for d in data:
        i += 1
        print(f"{i}, ", end=' ')
        x_train, y_train, x_test, y_test = d
        x_test_len = len(x_test)
        part_predictions = []
        start = timeit.default_timer()
        if reduction_technique == "RNN":
            CNN_config = CNN(k=k, dist_metric=distance_metric, weights=weighting_scheme)
            reduced_x, reduced_y = CNN_config.reduce_rnn(x_train, y_train, numeric_cols, nominal_cols)
        if reduction_technique == "CNN":
            CNN_config = CNN(k=k, dist_metric=distance_metric, weights=weighting_scheme)
            reduced_x, reduced_y = CNN_config.reduce_cnn(x_train, y_train, numeric_cols, nominal_cols)
        elif reduction_technique == "EENTh":
            EENTh_config = EENTh(k=k, dist_metric=distance_metric, weights=weighting_scheme)
            reduced_x, reduced_y = EENTh_config.reduce(x_train, y_train, numeric_cols, nominal_cols)
        elif reduction_technique == "DROP3":
            DROP3_config = DROP3(k=k, dist_metric=distance_metric, weights=weighting_scheme)
            reduced_x, reduced_y = DROP3_config.reduce_drop3(x_train, y_train, numeric_cols, nominal_cols)
        print(f'Original length: {len(x_train)}, Reduced length: {len(reduced_x)}')
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
    return ([round(avg_acc * 100, 2), round(avg_time, 2)], ['CNN', k, distance_metric, voting_scheme, weighting_scheme])


def reduction_techniques(data, dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols,
                         nominal_cols, verbose=False):
    results = []
    # drop3_results = reduce(data, "DROP3", dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols, nominal_cols, verbose=False)
    ccn_results = reduce(data, "CNN", dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols, nominal_cols, verbose=False)
    eenth_results = reduce(data, "EENTh", dataset_name, k, distance_metric, voting_scheme, weighting_scheme, numeric_cols, nominal_cols, verbose=False)
>>>>>>> Stashed changes

    # results.append(drop3_results)

    results.append(eenth_results)
    print(results)

    results_sorted_by_accuracy = sorted(results, key=lambda item: item[0][0], reverse=True)
<<<<<<< Updated upstream

    return results_sorted_by_accuracy
<<<<<<< Updated upstream
"""
=======
=======
    return results_sorted_by_accuracy
>>>>>>> Stashed changes
>>>>>>> Stashed changes
