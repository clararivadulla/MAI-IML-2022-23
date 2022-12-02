from performance.performance_rt import test_performance_rt
from pre_processing import data
from reduction_techniques.EENTh import EENTh

if __name__ == '__main__':
    """
    vowel_data = data.get_data(dataset_name='vowel')
    for d in vowel_data:
        x_train, y_train, x_test, y_test = d
        EENTh_config = EENTh(k=5)
        reduced_x, reduced_y = EENTh_config.reduce(x_train, y_train)
        print('Len x_train: ' + str(len(x_train)) + ' Len reduced_x: ' + str(len(reduced_x)))
        print('Len y_train: ' + str(len(y_train)) + ' Len reduced_y: ' + str(len(reduced_y)))
    """

    # In this file, we run kNN for every dataset with the best parameters found for that dataset
    print(
        f'\n··················································\nPERFORMANCE TESTS FOR VOWEL DATASET WITH REDUCTION TECHNIQUES\n··················································')
    dataset_name = 'vowel'
    vowel_data = data.get_data(dataset_name=dataset_name)
    vowel_results_sorted_by_accuracy = test_performance_rt(data=vowel_data, dataset_name=dataset_name)

    print(
        f'\n··················································\nPERFORMANCE TESTS FOR PEN-BASED DATASET WITH REDUCTION TECHNIQUES\n··················································')
    dataset_name = 'pen-based'
    pen_based_data = data.get_data(dataset_name=dataset_name)
    pen_based_results_sorted_by_accuracy = test_performance_rt(data=pen_based_data, dataset_name=dataset_name)

    print(
        f'\n··················································\nPERFORMANCE TESTS FOR SATIMAGE DATASET WITH REDUCTION TECHNIQUES\n··················································')
    dataset_name = 'satimage'
    sat_image_data = data.get_data(dataset_name=dataset_name)
    satimage_results_sorted_by_accuracy = test_performance_rt(data=sat_image_data, dataset_name=dataset_name)
