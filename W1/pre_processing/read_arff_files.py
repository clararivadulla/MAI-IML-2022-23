from scipy.io import arff
import pandas as pd


def read_arff_file(filename):
    data, meta = arff.loadarff(filename)  # Load the arff file with the 'loadarff' function
    df = pd.DataFrame(data)  # Create a dataframe 'df' containing the data of the arff file read
    return df, meta


def main(filename):
    # Choose 3 files to read their data and keep every data set inside a dataframe
    df, meta = read_arff_file('datasets/' + filename)
    return df, meta
