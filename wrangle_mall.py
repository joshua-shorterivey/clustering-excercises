#basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import env

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_mallcustomer_data():
    '''
    Reads in all fields from the customers table in the mall_customers schema from data.codeup.com
    
    parameters: None
    
    returns: a single Pandas DataFrame with the index set to the primary customer_id field
    '''
    df = pd.read_sql('SELECT * FROM customers;', get_connection('mall_customers'))
    df.to_csv('mall.csv')
    return df.set_index('customer_id')

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prnt_miss})
    rows_missing = rows_missing.reset_index().groupby(['num_cols_missing', 'percent_cols_missing']).count().reset_index().\
        rename(columns={'customer_id':'count'})

    return rows_missing

def nulls_by_col(df):
    num_missing= df.isnull().sum()
    percnt_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame(
        {
            'num_rows_missing': num_missing,
            'percent_rows_missing': percnt_miss
        }
    )
    return cols_missing

def summarize(df):
    print('DaataFrame head:\n')
    print(df.head().to_markdown())
    print('-----')
    print('DataFrame info:\n')
    print (df.info())
    print('---')
    print('DataFrame describe:\n')
    print (df.describe())
    print('---')
    print('DataFrame null value asssessment:\n')
    print('Nulls By Column:', nulls_by_col(df))
    print('----')
    print('Nulls By Row:', nulls_by_row(df))
    numerical_cols = [col for col in df.columns if df[col].dtype !='O']
    categorical_cols = [col for col in df.columns if col not in numerical_cols]
    print('value_counts: \n')
    for col in df.columns:
        print(f'Column Names: {col}')
        if col in categorical_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
            print('---')
    print('Report Finished')
    return

def get_upper_outliers(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k*iqr

    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df[col + '_upper_outliers'] = get_upper_outliers(df[col], k)
    return df

def split_train_test_validate(df):
    train_validate, test = train_test_split(df, test_size= .2, random_state=514)
    train, validate = train_test_split(train_validate, test_size= .3, random_state=514)
    print(train.shape, validate.shape, test.shape)
    return train, validate, test

def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.index), 0))
    df = df.dropna(axis=0, thresh=threshold)

    return df


def scale_split_data (train, validate, test):
    train_scaled = MinMaxScaler(train)
    validate_scaled = MinMaxScaler(validate)
    test_scaled = MinMaxScaler(test)

    return train_scaled, validate_scaled, test_scaled



def wrangle_mall():


    outlier_cols = [col for col in df.columns if col.endswith('_outliers')]
    for col in outlier_cols:
        print(col, ':\n')
        subset = df[col][df[col] > 0] 
        print(subset.describe())




