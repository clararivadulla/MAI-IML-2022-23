from pre_processing import pre_processing_functions


def main(df, meta, norm_type='gaussian'):

    (nominal_cols, numeric_cols) = pre_processing_functions.get_columns_by_type(meta)

    pre_processing_functions.drop_rows_with_na_values(df)
    pre_processing_functions.substitute_missing_values_by_mean(df, numeric_cols)

    class_labels = pre_processing_functions.remove_and_return_class_column(df).to_numpy()

    pre_processing_functions.byte_strings_to_strings(df)
    df = pre_processing_functions.one_hot_encode_columns(df, nominal_cols)

    if norm_type == 'gaussian':
        pre_processing_functions.standardize_columns(df, numeric_cols)
    elif norm_type == "l2":
        pre_processing_functions.normalize_columns(df, numeric_cols)
    elif norm_type == "min_max":
        pre_processing_functions.min_max_scale_columns(df, numeric_cols)

    data = pre_processing_functions.df_to_numeric_array(df)

    return data, class_labels
