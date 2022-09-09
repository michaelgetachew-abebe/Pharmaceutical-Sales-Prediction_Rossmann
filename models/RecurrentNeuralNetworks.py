import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

time_series_data = pd.read_csv('../data/train_store .csv',
                               index_col="Date", parse_dates=True)
time_series_data = time_series_data['Sales']
time_series_data = pd.DataFrame({
    "Date": time_series_data.index,
    "Sales": time_series_data
})
time_series_data = time_series_data.set_index("Date")
print(time_series_data.head())