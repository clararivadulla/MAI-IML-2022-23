from pre_processing import pre_processing_functions


def main(df, numerical_only = False, norm_type='gaussian'):
    
    class_labels = pre_processing_functions.remove_and_return_class_column(df).to_numpy()

    pre_processing_functions.byte_strings_to_strings(df)
    
    df = pre_processing_functions.one_hot_encode_columns(df, ['weducation', 'heducation', 'hoccupation', 'living_index'])
    
    if norm_type=='gaussian':
        pre_processing_functions.standardize_columns(df, ['wage', 'children'])
    elif norm_type == "l2":
        pre_processing_functions.normalize_columns(df, ['wage', 'children'])
    elif norm_type=="min_max":
        pre_processing_functions.min_max_scale_columns(df, ['wage', 'children'])

    if numerical_only:
        df = df[['wage', 'children']]

    data = pre_processing_functions.df_to_numeric_array(df)

    return data, class_labels
