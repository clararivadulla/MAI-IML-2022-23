from pre_processing import pre_processing_functions


def main(df, meta):
    df = pre_processing_functions.drop_class_column(df)

    pre_processing_functions.byte_strings_to_strings(df)

    #nominal columns
    cols = pre_processing_functions.find_cols_with_missing_values(df)
    pre_processing_functions.substitute_missing_values_by_most_common(df, cols)

    #numeric columns
    (nominal_cols, numeric_cols) = pre_processing_functions.get_columns_by_type(meta)
    pre_processing_functions.fill_na_values_with_mean(df, numeric_cols)

    df = pre_processing_functions.one_hot_encode_columns(df, nominal_cols)
    pre_processing_functions.min_max_scale_columns(df, numeric_cols)
    return df