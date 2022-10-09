from pre_processing import pre_processing_functions


def main(df):
    pre_processing_functions.byte_strings_to_strings(df)
    cols = pre_processing_functions.find_cols_with_missing_values(df)
    pre_processing_functions.substitute_missing_values_by_most_common(df, cols)
    df = pre_processing_functions.one_hot_encode_columns(df, ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'])
    pre_processing_functions.min_max_scale_columns(df, ['A2', 'A3', 'A8', 'A11', 'A14', 'A15'])
    return df