import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import exists
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import env

# !!!!!!!! WRITE UP A MODULE DESCRIPTION

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_zillow_data():
    '''
    Reads in all fields from the customers table in the mall_customers schema from data.codeup.com
    
    parameters: None
    
    returns: a single Pandas DataFrame with the index set to the primary customer_id field
    '''

    sql = """
    SELECT 
        prop.*,
        predictions_2017.logerror as logerror,
        aircon.airconditioningdesc as aircon,
        arch.architecturalstyledesc as architecture,
        buildclass.buildingclassdesc as building_class, 
        heating.heatingorsystemdesc as heating,
        landuse.propertylandusedesc as landuse, 
        story.storydesc as story,
        construct_type.typeconstructiondesc as construct_type
    FROM properties_2017 prop
    JOIN (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid
        ) pred USING (parcelid)
    JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                        AND pred.max_transactiondate = predictions_2017.transactiondate
    LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
    LEFT JOIN airconditioningtype aircon USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype buildclass USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype heating USING (heatingorsystemtypeid)
    LEFT JOIN storytype story USING (storytypeid)
    LEFT JOIN typeconstructiontype construct_type USING (typeconstructiontypeid)
    WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential") 
        AND transactiondate like '%%2017%%';
    """

    if exists('zillow_data.csv'):
        df = pd.read_csv('zillow_data.csv')
    else:
        df = pd.read_sql(sql, get_connection('zillow'))
        df.to_csv('zillow_data.csv', index=False)
    return df

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    percnt_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame(
        {
            'num_rows_missing': num_missing,
            'percent_rows_missing': percnt_miss
        }
    )
    return cols_missing

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prnt_miss})
    rows_missing = rows_missing.reset_index().groupby(['num_cols_missing', 'percent_cols_missing']).count().reset_index()

    return rows_missing

def summarize(df):
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
    numerical_cols = df.select_dtypes(include='number').columns.to_list()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()
    print('value_counts: \n')
    for col in df.columns:
        print(f'Column Names: {col}')
        if col in categorical_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False, dropna=False))
            print('---')
    print('Report Finished')
    return

def handle_missing_values(df, prop_required_columns=0.60, prop_required_row=0.75):
    
    #fill na values
    df.heating.fillna('None', inplace=True)
    df.aircon.fillna('None', inplace=True)
    df.basementsqft.fillna(0,inplace=True)
    df.garagecarcnt.fillna(0,inplace=True)
    df.garagetotalsqft.fillna(0,inplace=True)
    df.unitcnt.fillna(1,inplace=True)
    df.poolcnt.fillna(0, inplace=True)

    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)

    return df

def split_data(df):

    df = df.dropna()
    train_validate, test = train_test_split(df, test_size= .2, random_state=514)
    train, validate = train_test_split(train_validate, test_size= .3, random_state=514)
    print(train.shape, validate.shape, test.shape)
    return train, validate, test

def scale_split_data (train, validate, test):
    #create scaler object
    scaler = MinMaxScaler()

    # create copies to hold scaled data
    train_scaled = train.copy(deep=True)
    validate_scaled = validate.copy(deep=True)
    test_scaled =  test.copy(deep=True)

    #create list of numeric columns for scaling
    num_cols = train.drop(columns=['logerror']).select_dtypes(include='number')

    #fit to data
    scaler.fit(num_cols)

    # apply
    train_scaled[num_cols.columns] = scaler.transform(train[num_cols.columns])
    validate_scaled[num_cols.columns] =  scaler.transform(validate[num_cols.columns])
    test_scaled[num_cols.columns] =  scaler.transform(test[num_cols.columns])

    return train_scaled, validate_scaled, test_scaled

def engineer_features(df):
    """
    takes in dataframe (tailored use case for zillow)
    makes an age feature
    renames features
    narrows down values of aircon
    makes openness feature
    makes tax/sqft feature
    uses quantiles to categorize home size
    makes est tax rate feature
    renames county values
    returns df
    """

    #remove unwanted columns, and reset index to id --> for the exercises
    #age
    df['age'] = 2022 - df['yearbuilt']

    #rename 
    df = df.rename(columns={'fips': 'county',
                            'bedroomcnt': 'bedrooms', 
                            'bathroomcnt':'bathrooms', 
                            'calculatedfinishedsquarefeet': 'area', 
                            'taxvaluedollarcnt': 'home_value',
                            'yearbuilt': 'year_built', 
                            'taxamount': 'tax_amount', 
                            })

    df = df[ (df["bedrooms"] > 0) | (df["bathrooms"] > 0) ]

    #not best way due to adapting from encoding
    df["aircon"] =  np.where(df.aircon == "None", "None",
                                np.where(df.aircon == "Central", "Central", "Other"))

    #not best way due to adapting from encoding
    df["heating"] =  np.where(df.heating == "Central", "Central",
                                np.where(df.heating == "None", "None",
                                np.where(df.heating == "Floor/Wall", "Floor/Wall", "Other")))

    ## feature to determine how open a house is likely to be
    df["openness"] = df.area / (df.bathrooms + df.bedrooms)

    ## feature to determine the relative tax per sqft
    df["tax_per_sqft"] = df.structuretaxvaluedollarcnt / df.area
    
    #use quantiles to calculate subgroups and assign to new column
    q1, q3 = df.area.quantile([.25, .75])
    df['home_size'] = pd.cut(df.area, [0,q1,q3, df.area.max()], labels=['small', 'medium', 'large'], right=True)

    #### Estimated Tax Rate
    df['est_tax_rate'] = df.tax_amount / df.home_value
    # value of home per square foot
    df['val_per_sqft'] = df.home_value / df.area

    df.county = df.county.map({6037: 'LA County', 6059: 'Orange County', 6111: 'Ventura County'})

    return df

def prep_zillow (df):
    """ 
    Purpose
        Perform preparation functions on the zillow dataset
    Parameters
        df: data acquired from zillow dataset
    Output
        df: the unsplit and unscaled data with removed columns
        X_train:
        X_train_scaled:
        X_validate:
        X_validate_scaled:
        X_test:
        X_test_scaled:
    """

    # #remove unwanted columns, and reset index to id --> for the exercises
    df = engineer_features(df)

    df = df.drop(columns=['parcelid', 'buildingqualitytypeid','censustractandblock',
                        'heatingorsystemtypeid', 'propertylandusetypeid', 'year_built', 
                        'rawcensustractandblock', 'landuse', 'fullbathcnt', 'basementsqft',
                        'assessmentyear', 'regionidcounty','regionidzip', 'regionidcity',
                        'propertycountylandusecode', 'propertyzoningdesc', 'calculatedbathnbr',
                        'finishedsquarefeet12', 'garagecarcnt', 'landtaxvaluedollarcnt',
                        'structuretaxvaluedollarcnt', 'home_value', 'unitcnt', 'roomcnt'])
    df = df.set_index('id')

    # take care of any duplicates:
    df = df.drop_duplicates()
    
    #split the data
    train, validate, test = split_data(df)

    #scale the data
    train_scaled, validate_scaled, test_scaled = scale_split_data(train, validate, test)

    return df, train, validate, test, train_scaled, validate_scaled, test_scaled     

def wrangle_zillow():
    """ 
    Purpose
        Perform acuire and preparation functions on the zillow dataset
    Parameters
        None
    Output
        df: the unsplit and unscaled data
        X_train:
        X_train_scaled:
        X_validate:
        X_validate_scaled:
        X_test:
        X_test_scaled:
    """
    #initial data acquisition
    df = get_zillow_data()

    # handle the missing data --> decisions made in advance
    df = handle_missing_values(df, prop_required_columns=0.64)
    
    #drop columns that are unneeded, split data
    df, train, validate, test, train_scaled, validate_scaled, test_scaled = prep_zillow(df)

    print(df.home_size)

    #summarize the data
    summarize(df)

    return df, train, validate, test, train_scaled, validate_scaled, test_scaled  
