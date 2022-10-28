from pre_processing import read_arff_files, pima_diabetes_pre_processing

def main():
    print(
        '\n\n··················································\nPIMA DIABETES DATASET\n··················································')

    df, meta = read_arff_files.main('pima_diabetes.arff')
    data, labels = pima_diabetes_pre_processing.main(df)

    # Our own Principal Component Analysis (PCA)
    print(
        '**************************************************\nOur own PCA\n**************************************************')
    print('K-Means ------------------------------------------')
    print('Agglomerative Clustering -------------------------')

    # sklearn PCA
    print(
        '**************************************************\nsklearn PCA\n**************************************************')
    print('K-Means ------------------------------------------')
    print('Agglomerative Clustering -------------------------')

    # sklearn IncrementalPCA
    print(
        '**************************************************\nsklearn IncrementalPCA\n**************************************************')
    print('K-Means ------------------------------------------')
    print('Agglomerative Clustering -------------------------')

    # Feature Agglomeration
    print(
        '**************************************************\nFeature Agglomeration\n**************************************************')
    print('K-Means ------------------------------------------')
    print('Agglomerative Clustering -------------------------')

if __name__ == '__main__':
    main()
