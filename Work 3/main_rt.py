from performance.performance_rt import reduction_techniques
from pre_processing import data
from reduction_techniques.EENTh import EENTh

if __name__ == '__main__':

    # In this file, we run kNN for every dataset with the best parameters found for that dataset
    print(
        f'\n··················································\nVOWEL DATASET KNN WITH REDUCTION TECHNIQUES\n··················································')
    dataset_name = 'vowel'
    vowel_data, numeric_cols, nominal_cols = data.get_data(dataset_name=dataset_name)
    vowel_results_sorted_by_accuracy = reduction_techniques(data=vowel_data, dataset_name=dataset_name, k=1, distance_metric='cosine', voting_scheme='sheppard', weighting_scheme='mutual_info_score', numeric_cols=numeric_cols, nominal_cols=nominal_cols)

    print(
        f'\n··················································\nPEN-BASED DATASET KNN WITH REDUCTION TECHNIQUES\n··················································')
    dataset_name = 'pen-based'
    pen_based_data, numeric_cols, nominal_cols = data.get_data(dataset_name=dataset_name)
    #pen_based_results_sorted_by_accuracy = reduction_techniques(data=pen_based_data, dataset_name=dataset_name, k=3, distance_metric='minkowski', voting_scheme='majority', weighting_scheme='uniform', numeric_cols=numeric_cols, nominal_cols=nominal_cols)

    print(
        f'\n··················································\nSATIMAGE DATASET KNN WITH REDUCTION TECHNIQUES\n··················································')
    dataset_name = 'satimage'
    sat_image_data, numeric_cols, nominal_cols = data.get_data(dataset_name=dataset_name)
    #satimage_results_sorted_by_accuracy = reduction_techniques(data=sat_image_data, dataset_name=dataset_name, k=5, distance_metric='minkowski', voting_scheme='majority', weighting_scheme='uniform', numeric_cols=numeric_cols, nominal_cols=nominal_cols)
