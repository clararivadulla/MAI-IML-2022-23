import timeit
from kNN.kNN import kNN
from metrics.accuracies import accuracy
from reduction_techniques.EENTh import EENTh


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

                    for d in data:
                        x_train, y_train, x_test, y_test = d
                        x_test_len = len(x_test)
                        part_predictions = []
                        start = timeit.default_timer()
                        EENTh_config = EENTh(k=k, dist_metric=distance_metric, weights=weighting_scheme)
                        reduced_x, reduced_y = EENTh_config.reduce(x_train, y_train)
                        kNN_config = kNN(k=k, dist_metric=distance_metric, voting=voting_scheme, weights=weighting_scheme)
                        kNN_config.fit(reduced_x, reduced_y)

                        for j in range(x_test_len):
                            prediction = kNN_config.predict(x_test[j, :])
                            part_predictions.append(prediction)

                        stop = timeit.default_timer()
                        time = stop - start
                        times.append(time)

                        correct, incorrect, acc = accuracy(y_test, part_predictions)
                        if verbose:
                           print(f'Correct: {correct}, Incorrect: {incorrect}, Accuracy: {round(acc * 100, 2)}%, Time: {round(time, 2)}s')
                        accuracies.append(acc)

                    avg_time = sum(times) / len(times)
                    avg_acc = sum(accuracies) / len(accuracies)
                    results.append(([round(avg_acc * 100, 2), round(avg_time, 2)], [k, distance_metric, voting_scheme, weighting_scheme]))
                    print(f'Average accuracy: {round(avg_acc * 100, 2)}%, Average time: {round(avg_time, 2)}s')

    print(results)
    with open(f'results_rt_{dataset_name}.txt', 'w') as f:
        for res in results:
            f.write(f"{res}\n")

    results_sorted_by_accuracy = sorted(results, key=lambda item: item[0][0], reverse=True)

    return results_sorted_by_accuracy