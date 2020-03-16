import sys

import pandas as pd
import numpy as np

import yfinance as yf
import pycountry

import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

def load_data(path_input_index, path_input_currencies):
    '''
    INPUT:
    df_input_index - filepath to the csv file with the indices to download from yfinance 
    path_input_currencies - filepath to the csv file with the currencies to download from yfinance 

    OUTPUT:
    df_index - dataframe with the downloaded indices
    df_currency - dataframe with the downloaded currencies
    
    This function reads in two csv files as input and creates a dataframe for the indices and the currencies based on that input
    '''
    
    # load index input file
    df_input_index = pd.read_csv(path_input_index, sep=';', index_col='Index')
    # load currency dataset
    df_input_currencies = pd.read_csv(path_input_currencies, sep=';')
    
    #create the dataframe for the indices
    df_index = pd.DataFrame()
    #loop through the input dataframe and append one column to the df_index dataframe
    for index, row in df_input_index.iterrows():
        # create yfinance object and get historical market data
        yf_temp = yf.Ticker(index)
        df_temp = yf_temp.history(period="max")
        #drop not needed columns of returned dataframe
        df_temp =df_temp.drop(df_temp.columns.difference(['Open', 'High', 'Low', 'Close']), 1)
        #rename left column
        df_temp = df_temp.rename(columns={'Open': row['Country'] + '_' + index + '_Open', 
                                          'High': row['Country'] + '_' + index + '_High',
                                          'Low': row['Country'] + '_' + index + '_Low', 
                                          'Close': row['Country'] + '_' + index + '_Close'})
        df_index = df_index.join(df_temp, how='outer')
    
    #Loop over the currencies and append one column (named <Currency>) which contains the close value for each currency
    df_currency = pd.DataFrame()
    list_currencies = df_input_currencies.Currency.tolist()
    list_currencies = list(dict.fromkeys(list_currencies))

    for currency in list_currencies:
        # create yfinance object and get historical market data
        yf_temp = yf.Ticker(currency)
        df_temp = yf_temp.history(period="max")
        #drop not needed columns of returned dataframe
        df_temp =df_temp.drop(df_temp.columns.difference(['Open', 'High', 'Low', 'Close']), 1)
        #rename left column
        df_temp = df_temp.rename(columns={'Open': currency + '_Open',
                                          'High': currency + '_High',
                                          'Low': currency + '_Low',
                                          'Close': currency + '_Close'})
        df_currency = df_currency.join(df_temp, how='outer')
    
    return df_index, df_currency


def clean_df(df, block_size=100, values_from_start=3500, nan_allowed=5/7):
    '''
    INPUT:
    df - dataframe to be checked 
    block_size - size if the checked blocks
    values_from_start - minimum number of values a column must have without any gaps (from the bottom)
    nan_allowed - float which states how many Nan values are allowed

    OUTPUT:
    df_clean - dataframe with only the columns left which fullfill the required input 
    
    This function cleans a dataframe by dropping all columns which have to less values 
    without a gap and afterwards it fills the NaN values in the remaining columns     
    '''
    #Cleaning the loaded data
    df_clean = df.copy()
    #Eliminate the columns with big gaps in the last values_from_start entries (minimum datapoints for traing the model)
    length = df_clean.shape[0]
    for i in range(int(values_from_start/block_size)):
        df_clean = df_clean.loc[:, df_clean.iloc[(length-(i+1)*block_size):(length-i*block_size)].isnull().mean() < nan_allowed]

    #Use Front fill only for columns with Close in name - nan values in the middle occure only because of bank holiday's
    columns_list = df_clean.columns.tolist()
    close_columns_list = []
    for column in columns_list:
        if '_Close' in column:
            close_columns_list.append(column)
            df_clean[[column]] = df_clean[[column]].fillna(method='ffill')
    
    #drop all rows which still have nan values in a close Column
    df_clean = df_clean.dropna(subset=close_columns_list)    
    
    #for left NaN values use a backfill over the rows therefore on bank holidays the values will be the entire day constant
    df_clean = df_clean.fillna(axis=1, method='bfill')

    #drop all rows which still have nan values
    df_clean = df_clean.dropna(how='any')
    
    return df_clean

def print_results(df, df_cleaned, subject):
    '''
    INPUT:
    df - original dataframe 
    df_cleaned - cleaned dataframe
    subject - indices or currencies
    
    This function prints the results of the cleaning to the console.
    ''' 
    #compare how many datapoints are left
    number_values_originally = df.shape[0]
    number_values_left = df_cleaned.shape[0]

    print('For the ' + subject + '  data ' + str(number_values_left) + ' datapoints are left from ' + str(number_values_originally) + '.')
    
    #get the number of droped and kept indices
    number_of_originally_columns = df.shape[1]/4
    number_of_kept_columns = df_cleaned.shape[1]/4
    
    print('For the ' + subject + ' ' + str(number_of_kept_columns) + ' from ' + str(number_of_originally_columns) + ' provided ' + subject + ' could be kept.') 
    
def save_data(df_index_clean, df_currency_clean, database_filename):
    '''
    INPUT:
    df_index_clean - cleaned dataframe with the remaining index data 
    df_currency_clean - cleaned dataframe with the remaining currency data 
    database_filename - Name of the database the cleaned data should be stored in

    This function saves the clean dataset into an sqlite database by using pandas to_sql method combined with the SQLAlchemy library.
    '''  
    
    engine = create_engine('sqlite:///' + database_filename)
    Base = declarative_base()
    Base.metadata.drop_all(engine)
    df_index_clean.to_sql('index_data', engine, index=True, if_exists='replace')
    df_currency_clean.to_sql('currency_data', engine, index=True, if_exists='replace') 

def main():
    '''
    This is the main function which is executed when calling the CP_ETL_pipeline.py file over the console. It reads in the arguments 
    
        - path_input_index
        - path_input_currencies
        - block_size
        - values_from_start
        - nan_allowed
        - database_filename
    
    and executes the functions above. If one argument is missing an error is raised.
    '''
    
    if len(sys.argv) == 7:

        path_input_index = str(sys.argv[1])
        path_input_currencies = str(sys.argv[2]) 
        block_size = int(sys.argv[3]) 
        values_from_start = int(sys.argv[4]) 
        nan_allowed = float(sys.argv[5]) 
        database_filename = str(sys.argv[6])
        
        print('Loading data...\n    INDICES: {}\n    CURRENCIES: {}'
              .format(path_input_index, path_input_currencies))
        df_index, df_currency = load_data(path_input_index, path_input_currencies)

        print('Cleaning currency data...')
        df_currency_clean = clean_df(df_currency, block_size, values_from_start, nan_allowed)
        print_results(df_currency, df_currency_clean, 'currencies')
        
        print('Cleaning index data...')
        df_index_clean = clean_df(df_index, block_size, values_from_start, nan_allowed)
        print_results(df_index, df_index_clean, 'indices')
        
        print('Saving data...\n    DATABASE: {}'.format(database_filename))
        save_data(df_index_clean, df_currency_clean, database_filename)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the input for the currencies and '\
              'indices as the first and second argument respectively, as '\
              'well as the blocksize, values from start and allowed nan values '\
              'as the third, fourth and fifth argument and the filepath of the '\
              'database to save the cleaned data to as the sixth argument. '\
              '\n\nExample: python CP_ETL_pipeline.py '\
              'Input_index.csv Input_currencies.csv 100 4100 0.7143 '\
              'db_currency_predictor.db')


if __name__ == '__main__':
    main()