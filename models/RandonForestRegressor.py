from asyncio.log import logger
import pandas as pd
import numpy as np
import dvc.api
import mlflow
import mlflow.sklearn
import logging
import pickle
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import *

import os
import sys
cwd = os.getcwd()
sys.path.append(f"{cwd}/scripts")

from data_preprocess import preprocess
from logger_creator import log
from loss_functions import rmse
data_version = "version1"
data_url = dvc.api.get_url(
    path = 'data/tarin_store.csv',
    repo = '../',
    rev = data_version
)

logger = log(path = '../logs/', file = 'randomforestregressor_log.log')
logger.info("Random Forest is Rolling....")

mlflow.set_experiment("Pharmaceutical sales prediction accros multiple stores in case of Rosemann Pharmaceuticals")

if __name__ == "main":
    warnings.filterwarnings("ignore")
    train_store = pd.read_csv('../data/train_store.csv, parse_dates=True, index_col=0')
    mlflow.log_param('data_version', data_version)
    mlflow.log_param('model_type', 'Random Forest Reressor')
    mlflow.log_param('data_url', data_url)

    test = pd.read_csv('../data/test.csv', parse_date = True, index_col = "Date")

    X_train, X_test, y_train, y_test = train_test_split(train_store, test, test_size=0.2, random_state=20)
    logger.info("Training and testing split was successful.")
    mlflow.log_param("Input columns:", X_train.shape[0])
    mlflow.log_param("Input rows:", X_train.shape[1])

    randomforestregressor = RandomForestRegressor(
        n_estimators = 60,
        criterion = 'mse',
        max_depth = 15,
        min_samples_leaf = 1,
        min_samples_split = 2,
        min_weight_fraction_leaf = 0.0,
        max_features = 'auto',
        max_leaf_nodes = None,
        min_impurity_decrease = 0.0,
        min_impurity_split = None,
        bootstrap = True,
        oob_score = False,
        n_jobs = 4,
        random_state = 18,
        verbose = 0,
        warm_start = False)
    
    randomforestregressor.fit(X_train, y_train)
    logger.info("Model fitting completed successfully")
    mlflow.sklearn.log_model(randomforestregressor, "Random Forest Regressor Model")

    #Prediction and Evaluation of the model
    ybar = randomforestregressor(X_test)
    prediction_error = rmse(y_test, ybar)

    logger.info(f"Model Prediction Error{prediction_error}")
    mlflow.log_param("Model Prediction Error", prediction_error)
    with open("Random_forest_regressor.txt", "w") as outfile:
        outfile.write("Model Prediction Error in:" + str(prediction_error))