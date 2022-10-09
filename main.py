from pre_processing import credit_a_pre_processing, cmc_pre_processing, read_arff_files, pima_diabetes_pre_processing

if __name__=='__main__':

    # cmc dataset reading and pre-processing
    df, meta = read_arff_files.main('cmc.arff')
    df = cmc_pre_processing.main(df)
    print(df)

    # credit-a dataset reading and pre-processing
    df2, meta2 = read_arff_files.main('credit-a.arff')
    df2 = credit_a_pre_processing.main(df2, meta2)
    print(df2)

    # pima-diabetes dataset reading and pre-processing
    df3, meta3 = read_arff_files.main('pima_diabetes.arff')
    df3 = pima_diabetes_pre_processing.main(df3)
    print(df3)

    """
    #read_arff_files.main()
    dataframe, meta = read_arff_file('datasets/credit-a.arff')

    print(dataframe.head(10))
    print(dataframe.describe())

    (nominal, numeric) = get_columns_by_type(meta)

    byte_strings_to_strings(dataframe)

    #drop_rows_with_missing_values(dataframe)

    substitute_missing_values_by_mean(dataframe, numeric)
    substitute_missing_values_by_most_common(dataframe, nominal)

    label_encode_columns(dataframe, nominal)
    #dataframe = one_hot_encode_columns(dataframe, nominal)

    min_max_scale_columns(dataframe, numeric)

    train_data = dataframe.iloc[:0.9*dataframe.shapep[0],:].copy()
    test_data = dataframe.iloc[0.9 * dataframe.shapep[0]:, :].copy()

    #print(dataframe.head(10))
    #print(dataframe.describe())

    clustering = AgglomerativeClustering().fit(dataframe)

    kmeans = Kmeans(k = 4, max_iter = 100, seed = None).train(train_data)
    classify = kmeans.classify(test_data)
    print (classify[0], classify[1])

    #print(clustering.labels_)
    """
