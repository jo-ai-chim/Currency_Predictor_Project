# Currency Predictor

### Table of Contents

1. [Installation](#installation)
2. [Instructions](#instructions)
3. [Project Motivation](#motivation)
4. [Files](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

To run the code in this project beside the standard libraries already included in the standard Anaconda installation (sys, numpy, pandas, pickle, datetime, sklearn, sqlite3 and sqlalchemy) you need to install 

- plotly library - the according documentation can be found [here](https://plot.ly/).
- yfinance library - the according documentation can be found [here](https://pypi.org/project/yfinance/).
- pycountry library - the according documentation can be found [here](https://pypi.org/project/pycountry/).

The code should run with no issues using Python versions 3.*.

## Instructions <a name="instructions"></a>

Beside the jupyter notebooks you can run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that gets the data from yahoo finance, cleans it and stores it the database
        `python data/CP_ETL_pipeline.py data/Input_index.csv data/Input_currencies.csv 100 4100 0.7143 data/db_currency_predictor.db
    - To run ML pipeline that trains the classifier and saves it
        `python models/CP_ML_pipeline.py data/db_currency_predictor.db EURUSD=X_Open`

## Project Motivation <a name="motivation"></a>

If you search for models for time series predictions you find a lot of possibilities. From deep learning neuronal networks like LSTM (long short term memory) with tensorflow or pytorch over the facebook Prophet package to different models like Vector Auto Regression (VAR) or ARIMA from the statsmodels module.

Unfortunately you find very little input on how to make time series predictions using scikit learn. Especially when you have multivariate input. So I wondered if it is even possible to make good time series predictions with the supervised learning models in scikit learn. 

Since most of the time series projects you find are concerning financial data, I also decided to chose a use case from this area. So the goal of this project is to make a prediction for the currency exchange rates. Here the question came up were to get the data from. 

So to sum it up - with this project I want to answer the following questions:

1. Is there open financial time series data which can be used for making a prediction on currency exchange rates?
2. Is it possible to make good predictions for multivariate time series data using the supervised learning models within scikit learn?

## Files <a name="files"></a>

The project contains (at least) 8 files beside this readme.

data files:

- Input_currencies.csv in data folder - Input Data with currencies that should be downloaded
- Input_index.csv in data folder - Input Data with the indices that should be downloaded
- db_currency_predictor.db in data folder - Database with Input Data for training and testing the model
- CP_model_[currency]_trained.pkl in models folder - saved the model in new file for every currency the model is trained for

code files:

- Currency_Predictor_ETL.ipynb in data folder - explanation how the ETL pipeline was developed
- CP_ETL_pipeline.py in data folder - contains the etl pipeline to download the data, transform it and save it in the database
- Currency_Predictor_ML.ipynb in models folder - explanation how the ML pipeline was developed
- CP_ML_pipeline.py in models folder - sets up and evaluates the model and stores it to a pickle file 

## Results<a name="results"></a>

The analyses of the retrieved data from yahoo finance (see Currency_Predictor_ETL.ipynb) showed that - even there are some gaps - the data is quite good and especially good enough for setting up a supervised learning model.

The following analyses for setting up a model (see Currency_Predictor_ML.ipynb) with scikit learn showed that there are quite some supervised learning models which can be used for making a prediction for multivariate time series data. We also managed to optimize one of those models to a mean squared error of 0.000045 and a r2 score of 0.98 what is pretty good.

Since this is only a first analysis to check if it is possible to make mulitvariate time series predictions based on open data with scikit learn, I want to note that there are a lot of possibilities to improve the predictions even more.

- Based on the analyses we can expect an improvement by simply adding more indices or whole different features like the gold price to the input. 
- The gridsearch was almost at it's limit. Therefore we needed to make some restrictions. Here there are quite a view options to optimize the model further. First of all the scaler could be checked (maybe for some models a RobustScaler, QuantileTransformer, or Normalizer would lead to a better performance). Also the functions (like the train test split and the creation of the timedata features) could be implemented as transformers and then be considered during the gridsearch.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

I want to give credit for providing the data to the following sources:

- [Yahoo-Finance](https://de.finance.yahoo.com/) - You can find more information regarding the use of yahoo finance data [here](https://de.hilfe.yahoo.com/kb/finance-for-web/SLN2310.html?impressions=true).
- [Wikipedia Currency list](https://en.wikipedia.org/wiki/List_of_circulating_currencies) - You can find more information regarding the use of Wikipedia data [here](https://en.wikipedia.org/wiki/Wikipedia:About).
- [Wikipedia Index list](https://en.wikipedia.org/wiki/List_of_stock_market_indices) - You can find more information regarding the use of Wikipedia data [here](https://en.wikipedia.org/wiki/Wikipedia:About).

I also want to give credit to the developers of the [plotly](https://plot.ly/), [yfinance](https://pypi.org/project/yfinance/) and [pycountry](https://pypi.org/project/pycountry/) libraries. And last but not least to the contributors to the [stackoverflow platform](https://stackoverflow.com/). 

If you decide to improve the model or if you have other feedback, I would be happy if you contact me over github. 
