from pre_processing import pre_processing_functions


def main(df):
    pre_processing_functions.min_max_scale_columns(df, df.columns)
    return df