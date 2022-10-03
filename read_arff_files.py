from scipy.io import arff
import pandas as pd

def read_arff_file(filename):
    data, meta = arff.loadarff(filename) # Load the arff file with the 'loadarff' function
    df = pd.DataFrame(data) # Create a dataframe 'df' containing the data of the arff file read
    # Delete the column containing the class of the data set, since it is not necessary for this exercise
    if 'class' in df.columns:
        df = df.drop('class', axis=1)
    if 'Class' in df.columns:
        df = df.drop('Class', axis=1)
    return df, meta

def main():
    # Choose 3 files to read their data and keep every data set inside a dataframe
    df1 = read_arff_file('datasets/adult.arff')
    df2 = read_arff_file('datasets/hepatitis.arff')
    df3 = read_arff_file('datasets/pima_diabetes.arff')
    return df1, df2, df3
