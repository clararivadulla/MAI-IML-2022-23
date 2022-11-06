from pre_processing import read_arff_files, iris_pre_processing, cmc_pre_processing, pima_diabetes_pre_processing
import run_clustering

if __name__ == '__main__':
    df, meta = read_arff_files.main('iris.arff')
    data, labels = iris_pre_processing.main(df)
    run_clustering.run(data, labels, dataset_name="Iris")

    df, meta = read_arff_files.main('cmc.arff')
    data, labels = cmc_pre_processing.main(df)
    run_clustering.run(data, labels, dataset_name="CMC")

    df, meta = read_arff_files.main('pima_diabetes.arff')
    data, labels = pima_diabetes_pre_processing.main(df)
    run_clustering.run(data, labels, dataset_name="Pima Diabetes")

    #main_cmc.main()
    #main_iris.main()
    #main_pima_diabetes.main()