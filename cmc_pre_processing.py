import pre_processing


def main(df):
    pre_processing.byte_strings_to_strings(df)
    df = pre_processing.one_hot_encode_columns(df, ['weducation', 'heducation', 'hoccupation', 'living_index'])
    pre_processing.min_max_scale_columns(df, ['wage', 'children'])
    return df