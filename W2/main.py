from pre_processing import read_arff_files, vowel_pre_processing, iris_pre_processing, pima_diabetes_pre_processing
import run_clustering

if __name__ == '__main__':
    df, meta = read_arff_files.main('vowel.arff')
    data, labels = vowel_pre_processing.main(df, meta, norm_type='min_max')
    run_clustering.run(data, labels, dataset_name="Vowel", k=4, num_features=3, plot_3D=True)

    df, meta = read_arff_files.main('iris.arff')
    data, labels = iris_pre_processing.main(df)
    run_clustering.run(data, labels, dataset_name="Iris", k=3, num_features=3, plot_3D=False)

    df, meta = read_arff_files.main('pima_diabetes.arff')
    data, labels = pima_diabetes_pre_processing.main(df)
    run_clustering.run(data, labels, dataset_name="Pima Diabetes", k=2, num_features=2, plot_3D=False)

    #main_cmc.main()
    #main_iris.main()
    #main_pima_diabetes.main()