from pre_processing import vowel_pre_processing, penbased_pre_processing, satimage_pre_processing


def pre_process(df, meta, dataset_name=''):
    if dataset_name == 'vowel':
        return vowel_pre_processing.main(df, meta, norm_type='min_max')
    elif dataset_name == 'pen-based':
        return penbased_pre_processing.main(df, meta, norm_type='min_max')
    elif dataset_name == 'satimage':
        return satimage_pre_processing.main(df, meta, norm_type='min_max')