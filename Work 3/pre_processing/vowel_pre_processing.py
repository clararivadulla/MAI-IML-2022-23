from pre_processing import pre_processing_functions


def main(df, meta, norm_type='gaussian'):

    (nominal_cols, numeric_cols) = pre_processing_functions.get_columns_by_type(meta)
    pre_processing_functions.byte_strings_to_strings(df)
    class_labels = pre_processing_functions.remove_and_return_class_column(df).to_numpy()
    pre_processing_functions.label_encode_columns(df, nominal_cols)

    if norm_type == 'gaussian':
        pre_processing_functions.standardize_columns(df, numeric_cols)
    elif norm_type == "l2":
        pre_processing_functions.normalize_columns(df, numeric_cols)
    elif norm_type == "min_max":
        pre_processing_functions.min_max_scale_columns(df, numeric_cols)

    data = pre_processing_functions.df_to_numeric_array(df)

    if len(numeric_cols) == 0:
        num_idx = None
    else:
        num_idx = []
        for i in range(len(numeric_cols)):
            id = list(df.columns).index(numeric_cols[i])
            num_idx.append(id)

    if len(nominal_cols) == 0:
        nom_idx = None
    else:
        nom_idx = []
        for i in range(len(nominal_cols)):
            id = list(df.columns).index(nominal_cols[i])
            nom_idx.append(id)

    return data, class_labels, num_idx, nom_idx
