import timeit, time

def test_performance(x_train=None, y_train=None, x_test=None):

    ks = [1, 3, 5, 7]
    voting_schemes = ['majority', 'inverse_distance_weighted_votes', 'shepards']
    distance_metrics = ['minkowski', 'cosine', 'distance1']
    weighting_schemes = ['equal_weight', 'algorithm1', 'algorithm2']

    i = 0
    results = {}
    for k in ks:
        for voting_scheme in voting_schemes:
            for distance_metric in distance_metrics:
                for weighting_scheme in weighting_schemes:
                    i += 1
                    print(f'Test {i}')
                    print(f'k={k}, dist_metric={distance_metric}, voting={voting_scheme}, weights={weighting_scheme}\n')

                    #kNN = kNN(k=k, dist_metric=distance_metric, voting=voting_scheme, weights=weighting_scheme)

                    start = timeit.default_timer()
                    time.sleep(.01)
                    stop = timeit.default_timer()

                    results[i] = (stop - start, [k, distance_metric, voting_scheme, weighting_scheme])

    print(results)
