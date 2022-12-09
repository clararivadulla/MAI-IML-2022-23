from scipy.io import arff
import pandas as pd
from pre_processing import pre_process


def read_arff_file(filename):
    data, meta = arff.loadarff(filename)  # Load the arff file with the 'loadarff' function
    df = pd.DataFrame(data)  # Create a dataframe 'df' containing the data of the arff file read
    return df, meta

def get_data(dataset_name):
    data = []
    for i in range(10):
        #print(f'{dataset_name}/{dataset_name}.fold.00000{i}', end=' ')
        df_test, meta_test = read_arff_file(f'datasets/{dataset_name}/{dataset_name}.fold.00000{i}.test.arff')
        df_train, meta_train = read_arff_file(f'datasets/{dataset_name}/{dataset_name}.fold.00000{i}.train.arff')

        x_train, y_train, numeric_cols, nominal_cols = pre_process.pre_process_dataset(df_train, meta_train, dataset_name=dataset_name)
        x_test, y_test, numeric_cols, nominal_cols = pre_process.pre_process_dataset(df_test, meta_test, dataset_name=dataset_name)
        data.append((x_train, y_train, x_test, y_test))

    return data, numeric_cols, nominal_cols