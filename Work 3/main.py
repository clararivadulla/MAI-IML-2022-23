from performance.performance import test_performance
from pre_processing import data

if __name__ == '__main__':

    # In this file, we run kNN for every dataset with the best parameters found for that dataset
    print(f'\n··················································\nPERFORMANCE TESTS FOR VOWEL DATASET\n··················································')
    dataset_name = 'vowel'
    vowel_data, numeric_cols, nominal_cols = data.get_data(dataset_name=dataset_name)
    vowel_results_sorted_by_accuracy = test_performance(data=vowel_data, dataset_name=dataset_name, numeric_cols=numeric_cols, nominal_cols=nominal_cols)

    print(f'\n··················································\nPERFORMANCE TESTS FOR PEN-BASED DATASET\n··················································')
    dataset_name = 'pen-based'
    pen_based_data, numeric_cols, nominal_cols = data.get_data(dataset_name=dataset_name)
    pen_based_results_sorted_by_accuracy = test_performance(data=pen_based_data, dataset_name=dataset_name, numeric_cols=numeric_cols, nominal_cols=nominal_cols)

    print(f'\n··················································\nPERFORMANCE TESTS FOR SATIMAGE DATASET\n··················································')
    dataset_name = 'satimage'
    sat_image_data, numeric_cols, nominal_cols = data.get_data(dataset_name=dataset_name)
    satimage_results_sorted_by_accuracy = test_performance(data=sat_image_data, dataset_name=dataset_name, numeric_cols=numeric_cols, nominal_cols=nominal_cols)

    print()
