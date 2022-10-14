from pre_processing import pre_processing_functions


def main(df):
    df = pre_processing_functions.drop_class_column(df)

    pre_processing_functions.min_max_scale_columns(df, df.columns)
    return df