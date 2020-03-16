import sys

import sqlite3
from sqlalchemy import create_engine

import pickle

import numpy as np
import pandas as pd
import datetime

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error


def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - filepath to the sqllite database where the data from the etl pipeline is stored file with disaster response messages 
    
    OUTPUT:
    df_data - dataframe with the features
   
    This function reads in the train and test data from the sqlite database and splits them in features and labels
    '''
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df_index = pd.read_sql_table('index_data', engine, index_col='Date')
    df_currency = pd.read_sql_table('currency_data', engine, index_col='Date')
    
    #merge the two dataframes and front fill nan
    df_data = df_currency.join(df_index, how='outer')
    df_data = df_data.fillna(method='ffill')
    df_data = df_data.dropna(how='any')
    
    return df_data

def create_label_df(df, label, number_features=10, labels_future=1, dropnan=True):
    '''
    INPUT:
    df - input dataframe as the foundation for the model 
    label - column of the dataframe we want to build the model for
    number_features - number of additional features for each Open and Close value should be created 
    labels_future - number of labels for the label dataframe (in case not only the next day but the next view days should be predicted)
    dropnan - boolean to determinate if the resulting rows with NaN values due to the shifts should be dropped

    OUTPUT:
    X - dataframe with all features
    Y - dataframe with all labels 
    
    This function takes a dataframe creates the additional features and the labels and returns two corresponding dataframes.
    '''
    df_out = df.copy()    
    original_columns = df_out.columns.tolist()
    columns_ordered = []
    new_columns = []
    
    #Go through the columns and add features with the values on the previous days
    for column in original_columns:
        columns_ordered.append(column)
        #for high or low values add maximum 3 values
        if ('_High' in column) or ('_Low' in column):
            for i in range(min(number_features, 3)):
                df_out[column + '_t-' + str(i+1)]=df_out[column].shift(periods=(i+1))
                columns_ordered.append(column + '_t-' + str(i+1))
        #for other values add as many values as requested
        else:
            for i in range(number_features):
                df_out[column + '_t-' + str(i+1)]=df_out[column].shift(periods=(i+1))
                columns_ordered.append(column + '_t-' + str(i+1))
    
    #change the order of the columns
    df_out = df_out[columns_ordered]
    
    #add label column to the dataframe
    for i in range(labels_future):
        df_out[label + '_t+' + str(i+1)]=df_out[label].shift(periods=-(i+1))
        new_columns.append(label + '_t+' + str(i+1))
    
    #drop NaN
    if dropnan:
        df_out = df_out.drop(df_out.head(number_features).index)
        df_out = df_out.drop(df_out.tail(labels_future).index)
        
    #split into X and Y
    X = df_out[columns_ordered]
    Y = df_out[new_columns]
    
    return X, Y    
    
def train_test_split(X, Y, test_split = 0.7):
    '''
    INPUT:
    X - dataframe with all features
    Y - dataframe with all labels 
    test_split - float which determinates the ratio of the split
    
    OUTPUT:
    X_train - dataframe with all features in the train set
    X_test - dataframe with all labels in the test set
    Y_train - dataframe with all features in the train set
    Y_test - dataframe with all labels in the test set
    
    This function performs a train test split.
    '''
    
    # the percent of data to be used for testing
    n = int(X.shape[0] * test_split)

    # splitting the dataset up into train and test sets

    X_train = X[:n]
    Y_train = Y[:n]

    X_test = X[n:]
    Y_test = Y[n:]
    
    return X_train, X_test, Y_train, Y_test

    
def build_model():
    '''
    INPUT:
    clf - model for which the pipeline should be build for
    clf_params - parameters for gridsearch 
    
    OUTPUT:
    cv - gridsearch object
    
    This function creates the gridsearch object to be fitted.
    '''
    
    clf = GradientBoostingRegressor(random_state=42)
    clf_params = {
                "classifier__loss": ["ls"],
                "classifier__learning_rate": [0.1],
                "classifier__criterion": ["friedman_mse"],
                "classifier__subsample": [0.9],
                "classifier__n_estimators": [50], 
                "classifier__max_features": [None]
                }
    
    # set up transformation pipeline
    preprocess = Pipeline([
                    ('normalizer', MinMaxScaler())
                ])
    
    # set up total pipeline
    pipeline = Pipeline([
                    ('preprocess', preprocess),
                    ('classifier', clf)
                ])
    
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=clf_params)
    
    return cv
    
def evaluate_results(model, X_train, X_test, Y_train, Y_test):
    '''
    INPUT:
    model - model to be evaluated
    label - label the model tries to predict 
    X_train - dataframe with all features in the train set
    X_test - dataframe with all labels in the test set
    Y_train - dataframe with all features in the train set
    Y_test - dataframe with all labels in the test set
    
    This function calculates the mean squared error and r2 score for the test and the train set and also visualizes the results
    '''   
    
    #make predictions for train and test data
    Y_test_preds = model.predict(X_test)
    Y_train_preds = model.predict(X_train)
    
    #print the best parameters from gridsearch
    print("best parameters are: {}".format(model.best_params_))
    
    #print the metric values for train and test data 
    print('mean squared error on test set: ' + str(mean_squared_error(Y_test, Y_test_preds)))
    print('r2 score on test set: ' + str(r2_score(Y_test, Y_test_preds)))
    print('mean squared error on train set: ' + str(mean_squared_error(Y_train, Y_train_preds)))
    print('r2 score on train set: ' + str(r2_score(Y_train, Y_train_preds)))
    

def save_model(model, label):
    '''
    INPUT:
    model - the model which should be saved 
    label - label the model is trained to predict
    
    This function saves the model to the given path
    '''
    
    pickle.dump(model, open('CP_model_' + label + '_trained.pkl', 'wb'))

#Adding timedata features
def add_dateinfo(df):
    '''
    INPUT:
    df - dataframe with datetime as index

    OUTPUT:
    df_out - adjusted dataframe
    
    This function takes a dataframe and deletes all rows which don't contain a certain string
    '''
    df_out = df.copy()
    df_out.index = pd.to_datetime(df_out.index)
    #iterate through the rows of the dataframe
    weekday_list = []
    month_list = []
    week_list = []
    day_list = []
    for index, row in df_out.iterrows():
        weekday_list.append(index.weekday())
        month_list.append(index.month)
        week_list.append(index.week)
        day_list.append(index.day)
    
    df_out['weekday'] = weekday_list
    df_out['month'] = month_list
    df_out['week'] = week_list
    df_out['day'] = day_list
    
    return df_out
              
              
def main():
    '''
    This is the main function which is executed when calling the train_classifier.py file over the console. It reads in the arguments 
    
        - database_filepath
        - label
    
    and executes the functions above. If one argument is missing an error is raised.
    '''
    
    if len(sys.argv) == 3:
        
        database_filepath, label = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df_data = load_data(database_filepath)
    
        #add features from date
        df = add_dateinfo(df_data)
        
        #create features and labels dataframe and make a train test split
        X, Y = create_label_df(df, label)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 0.8)
        
        print('Building model...')
        #fit the model
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train.values.ravel())
        
        #evaluate the model
        print('Evaluating model...')
        evaluate_results(model, X_train, X_test, Y_train, Y_test)
        
        print('Saving model...\n    MODEL: {}'.format('CP_model_' + label + '_trained.pkl'))
        save_model(model, label)
        
        print('Trained model saved!')
        
    else:
        print('Please provide the filepath of the currency and index database '\
              'as the first argument and the name of the currency the model '\
              'should be buidl for asas the second argument. \n\nExample: python '\
              'CP_ML_pipeline.py ../data/db_currency_predictor.db EURUSD=X_Open')             
              
if __name__ == '__main__':
    main()