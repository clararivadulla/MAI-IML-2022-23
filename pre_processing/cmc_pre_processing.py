from pre_processing import pre_processing_functions


def main(df):
    pre_processing_functions.byte_strings_to_strings(df)
    df = pre_processing_functions.one_hot_encode_columns(df, ['weducation', 'heducation', 'hoccupation', 'living_index'])
    pre_processing_functions.min_max_scale_columns(df, ['wage', 'children'])
    return df