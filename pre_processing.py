import read_arff_files

def pre_processing(dataframes):
    for df in dataframes:
        df = different_ranges(df)
        df = different_types(df)
        df = missing_values(df)

def different_ranges(dataframe):
    ''' TO DO '''
    return dataframe

def different_types(dataframe):
    ''' TO DO '''
    return dataframe

def missing_values(dataframe):
    ''' TO DO '''
    return dataframe

def main():
    dataframes = read_arff_files.main()
    return pre_processing(dataframes)