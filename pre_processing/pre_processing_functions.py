from sklearn import preprocessing
import pandas as pd


def byte_strings_to_strings(df):
    s_df = df.select_dtypes([object])
    s_df = s_df.stack().str.decode('utf-8').unstack()
    for column in s_df:
        df[column] = s_df[column]


def min_max_scale_columns(df, cols):
    min_max_scaler = preprocessing.MinMaxScaler()
    df[cols] = min_max_scaler.fit_transform(df[cols])


def label_encode_columns(df, cols):
    label_encoder = preprocessing.LabelEncoder()
    for col in cols:
        df[col] = label_encoder.fit_transform(df[col])


def one_hot_encode_columns(df, cols):
    return pd.get_dummies(data=df, columns=cols)


def get_columns_by_type(meta):
    nominal_cols = []
    numeric_cols = []
    for i in range(0, len(meta.names())):
        if (meta.names()[i].lower() == 'class'):
            continue

        if (meta.types()[i] == 'nominal'):
            nominal_cols.append(meta.names()[i])

        if (meta.types()[i] == 'numeric'):
            numeric_cols.append(meta.names()[i])

    return (nominal_cols, numeric_cols)


def substitute_missing_values_by_mean(df, cols):
    for col in cols:
        df[col] = df[col].replace('?', df[col].mean())


def substitute_missing_values_by_median(df, cols):
    for col in cols:
        df[col] = df[col].replace('?', df[col].median())


def substitute_missing_values_by_most_common(df, cols):
    for col in cols:
        df[col] = df[col].replace('?', df[col].mode().iloc[0])


def find_cols_with_missing_values(df):
    mv = df.isin(['?']).any()
    cols = []
    for i in range(0, 15):
        if mv[i]:
            cols.append(df.columns[i])
    return cols


def drop_rows_with_na_values(df):
    df.dropna(axis=0, inplace=True)


def fill_na_values_with_mean(df, cols):
    for col in cols:
        df[col] = df[col].fillna(df[col].mean())


def fill_na_values_with_median(df, cols):
    for col in cols:
        df[col] = df[col].fillna(df[col].median())


def drop_class_column(df):
    if 'class' in df.columns:
        df = df.drop('class', axis=1)
    if 'Class' in df.columns:
        df = df.drop('Class', axis=1)
    return df


def df_to_numeric_array(df):
    df[df.columns] = df[df.columns].astype(float)
    data = df.to_numpy()
    return data