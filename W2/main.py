import main_cmc
import main_iris
import main_pima_diabetes
from pre_processing import read_arff_files, iris_pre_processing
import run_clustering

if __name__ == '__main__':
    df, meta = read_arff_files.main('iris.arff')
    data, labels = iris_pre_processing.main(df)
    run_clustering.run(data, labels, dataset_name="Iris")

    #main_cmc.main()
    #main_iris.main()
    #main_pima_diabetes.main()