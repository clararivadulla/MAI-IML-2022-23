import json
import timeit, time

from kNN.kNN import kNN
from metrics.accuracies import accuracy

def test_performance(data, dataset_name=''):

    ks = [1, 3, 5, 7]
    voting_schemes = ['majority', 'inverse_distance', 'sheppard']
    distance_metrics = ['minkowski', 'cosine', 'clark']
    weighting_schemes = ['uniform', 'mutual_info_score']

    results = []
    z = 0
    for k in ks:
        for voting_scheme in voting_schemes:
            for distance_metric in distance_metrics:
                for weighting_scheme in weighting_schemes:
                    z += 1
                    print(f'\nTEST {z}, k={k}, dist_metric={distance_metric}, voting={voting_scheme}, weights={weighting_scheme}')

                    times = []
                    accuracies = []
                    correct = []
                    incorrect = []

                    for i in range(len(data)):

                        x_train, y_train, x_test, y_test = data[i]

                        part_len = len(x_test)
                        part_predictions = []
                        start = timeit.default_timer()
                        kNN_config = kNN(k=k, dist_metric=distance_metric, voting=voting_scheme, weights=weighting_scheme)
                        kNN_config.fit(x_train, y_train)

                        for j in range(part_len):
                            prediction = kNN_config.predict(x_test[j, :])
                            part_predictions.append(prediction)

                        stop = timeit.default_timer()
                        time = stop - start
                        times.append(time)
                        c, i, p = accuracy(y_test, part_predictions)
                        print(f'Correct: {c}, Incorrect: {i}, Accuracy: {round(p * 100, 2)}%, Time: {round(time, 2)}')
                        accuracies.append(p)
                        correct.append(c)
                        incorrect.append(i)


                    """start = timeit.default_timer()
                    kNN_config.predict()
                    time.sleep(.01)
                    stop = timeit.default_timer()"""

                    avg_time = sum(times) / len(times)
                    avg_acc = sum(accuracies) / len(accuracies)
                    avg_correct = sum(correct) / len(accuracies)
                    avg_incorrect = sum(incorrect) / len(accuracies)
                    results.append(([avg_correct, avg_incorrect, avg_acc, avg_time], [k, distance_metric, voting_scheme, weighting_scheme]))

    print(results)
    with open(f'results_{dataset_name}.txt', 'w') as f:
        for res in results:
            f.write(f"{res}\n")
