from pre_processing import pre_processing_functions


def main(df):
    class_labels = pre_processing_functions.remove_and_return_class_column(df).to_numpy()
    
    # L2 norm
    pre_processing_functions.normalize_columns(df, df.columns)
    
    data = pre_processing_functions.df_to_numeric_array(df)
    
    return data, class_labels
