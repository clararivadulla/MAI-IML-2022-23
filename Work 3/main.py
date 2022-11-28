from pre_processing import read_arff_files, vowel_pre_processing, satimage_pre_processing
from kNN.kNN import kNN
from metrics.accuracies import accuracy
import timeit

if __name__ == '__main__':

    """
    print(
        f'\n\n··················································\nPEN-BASED DATASET\n··················································')

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
        print(f'Correct: {c}, Incorrect: {i}, Accuracy: {p}, Time: {time}')
"""
    """
    print(
        f'\n\n··················································\nSATIMAGE DATASET\n··················································')

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
        print(f'Correct: {c}, Incorrect: {i}, Accuracy: {p}, Time: {time}')
"""
    print(
        f'\n\n··················································\nVOWEL DATASET\n··················································')

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
        kNN_vowel = kNN(k=5, dist_metric='minkowski', voting='inverse_distance', weights='mutual_info_score')
        kNN_vowel.fit(x_train, y_train)

        for j in range(part_len):
            prediction = kNN_vowel.predict(x_test[j,:])
            part_predictions.append(prediction)

        stop = timeit.default_timer()
        time = stop - start
        vowel_times.append(time)
        c, i, p = accuracy(y_test, part_predictions)
        print(f'Correct: {c}, Incorrect: {i}, Accuracy: {p}, Time: {time}')


