from asyncio.log import logger
import pandas as pd
import numpy as np
import dvc.api
import mlflow
import mlflow.sklearn
import logging
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import *

import os
import sys
cwd = os.getcwd()
sys.path.append(f"{cwd}/scripts")

from data_preprocess import preprocess
from logger_creator import log

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

    train_store = pd.read_csv('../data/train_store.csv, parse_dates=True, index_col=0')
    mlflow.log_param('data_version', data_version)
    mlflow.log_param('model_type', 'Random Forest Reressor')
    mlflow.log_param('data_url', data_url)

    test = pd.read_csv('../data/test.csv', parse_date = True, index_col = "Date")

    X_train, X_test, y_train, y_test = train_test_split(train_store, test, test_size=0.2, random_state=0)