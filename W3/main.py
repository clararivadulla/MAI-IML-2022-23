from pre_processing import read_arff_files, vowel_pre_processing, iris_pre_processing, pima_diabetes_pre_processing
from kNN import KNN
if __name__ == '__main__':

    vowel_dataframes = []

    for i in range(10):

        df_test, meta_test = read_arff_files.main(f'vowel/vowel.fold.00000{i}.test.arff')
        df_train, meta_train = read_arff_files.main(f'vowel/vowel.fold.00000{i}.train.arff')
        data_test, labels_test = vowel_pre_processing.main(df_test, meta_test, norm_type='min_max')
        data_train, labels_train = vowel_pre_processing.main(df_train, meta_train, norm_type='min_max')
        vowel_dataframes.append(((data_test, labels_test), (data_train, labels_train)))

