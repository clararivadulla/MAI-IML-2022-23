from pre_processing import read_arff_files, vowel_pre_processing
from kNN.kNN import kNN
if __name__ == '__main__':

    vowel = []

    for i in range(10):

        df_test, meta_test = read_arff_files.main(f'vowel/vowel.fold.00000{i}.test.arff')
        df_train, meta_train = read_arff_files.main(f'vowel/vowel.fold.00000{i}.train.arff')
        data_test, labels_test = vowel_pre_processing.main(df_test, meta_test, norm_type='min_max', train=False)
        data_train = vowel_pre_processing.main(df_train, meta_train, norm_type='min_max', train=True)
        vowel.append([data_train, [data_test, labels_test]])

    kNN = kNN(k=5)
    vowel_predictions = []
    for part in range(len(vowel)):
        part_len = len(vowel[part][1][0])
        part_predictions = []
        for i in range(part_len):
            prediction = kNN.predict(vowel[part][0], vowel[part][1][0][i])
            part_predictions.append(prediction)
        vowel_predictions.append(part_predictions)

    print(vowel_predictions)