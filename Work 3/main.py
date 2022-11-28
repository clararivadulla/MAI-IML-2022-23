from performance.performance import test_performance
from pre_processing import data

if __name__ == '__main__':

    # In this file, we run kNN for every dataset with the best parameters found for that dataset
    print(f'··················································\n\nTESTS FOR VOWEL DATASET\n··················································')
    dataset_name = 'vowel'
    vowel_data = data.get_data(dataset_name=dataset_name)
    vowel_results_sorted_by_accuracy = test_performance(data=vowel_data, dataset_name=dataset_name)

    print(f'··················································\n\nTESTS FOR PEN-BASED DATASET\n··················································')
    dataset_name = 'pen-based'
    pen_based_data = data.get_data(dataset_name=dataset_name)
    pen_based_results_sorted_by_accuracy = test_performance(data=pen_based_data, dataset_name=dataset_name)

    print(f'··················································\n\nTESTS FOR SATIMAGE DATASET\n··················································')
    dataset_name = 'satimage'
    sat_image_data = data.get_data(dataset_name=dataset_name)
    satimage_results_sorted_by_accuracy = test_performance(data=sat_image_data, dataset_name=dataset_name)

    print()
    """
    print(
        f'··················································\nPEN-BASED DATASET\n··················································')

    penbased_times = []

    for i in range(10):

        print(f'pen-based/pen-based.fold.00000{i}', end=' ')
        df_test, meta_test = read_arff_files.main(f'pen-based/pen-based.fold.00000{i}.test.arff')
        df_train, meta_train = read_arff_files.main(f'pen-based/pen-based.fold.00000{i}.train.arff')
        x_test, y_test = penbased_pre_processing.main(df_test, meta_test, norm_type='min_max')
        x_train, y_train = penbased_pre_processing.main(df_train, meta_train, norm_type='min_max')
        
        part_len = len(x_test)
        part_predictions = []
        start = timeit.default_timer()
        kNN_penbased = kNN(k=3)
        kNN_penbased.fit(x_train, y_train)

        for j in range(part_len):
            prediction = kNN_penbased.predict(x_test[j,:])
            part_predictions.append(prediction)

        stop = timeit.default_timer()
        time = stop - start
        penbased_times.append(time)
        c, i, p = accuracy(y_test, part_predictions)
        print(f'Correct: {c}, Incorrect: {i}, Accuracy: {round(p * 100, 2)}%, Time: {round(time, 2)}')


    print(
        f'\n··················································\nSATIMAGE DATASET\n··················································')

    satimage_times = []

    for i in range(10):

        print(f'satimage/satimage.fold.00000{i}', end=' ')
        df_test, meta_test = read_arff_files.main(f'satimage/satimage.fold.00000{i}.test.arff')
        df_train, meta_train = read_arff_files.main(f'satimage/satimage.fold.00000{i}.train.arff')
        x_test, y_test = satimage_pre_processing.main(df_test, meta_test, norm_type='min_max')
        x_train, y_train = satimage_pre_processing.main(df_train, meta_train, norm_type='min_max')

        part_len = len(x_test)
        part_predictions = []
        start = timeit.default_timer()
        kNN_satimage = kNN(k=3)
        kNN_satimage.fit(x_train, y_train)

        for j in range(part_len):
            prediction = kNN_satimage.predict(x_test[j,:])
            part_predictions.append(prediction)

        stop = timeit.default_timer()
        time = stop - start
        satimage_times.append(time)
        c, i, p = accuracy(y_test, part_predictions)
        print(f'Correct: {c}, Incorrect: {i}, Accuracy: {round(p * 100, 2)}%, Time: {round(time, 2)}')

    print(
        f'\n··················································\nVOWEL DATASET\n··················································')

    vowel_times = []

    for i in range(10):

        print(f'vowel/vowel.fold.00000{i}', end = ' ')
        df_test, meta_test = read_arff_files.main(f'vowel/vowel.fold.00000{i}.test.arff')
        df_train, meta_train = read_arff_files.main(f'vowel/vowel.fold.00000{i}.train.arff')
        x_test, y_test = vowel_pre_processing.main(df_test, meta_test, norm_type='min_max')
        x_train, y_train = vowel_pre_processing.main(df_train, meta_train, norm_type='min_max')

        part_len = len(x_test)
        part_predictions = []
        start = timeit.default_timer()
        kNN_vowel = kNN(k=5)
        kNN_vowel.fit(x_train, y_train)

        for j in range(part_len):
            prediction = kNN_vowel.predict(x_test[j,:])
            part_predictions.append(prediction)

        stop = timeit.default_timer()
        time = stop - start
        vowel_times.append(time)
        c, i, p = accuracy(y_test, part_predictions)
        print(f'Correct: {c}, Incorrect: {i}, Accuracy: {round(p * 100, 2)}%, Time: {round(time, 2)}')
    """


