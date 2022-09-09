import pandas as pd
import numpy as np
import seaborn as sns
import sys

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder

def preprocess(train_store, test):
    # since competition open since have similar meanings we can merge into once
    train_store['CompetitionOpenSince'] = np.where((train_store['CompetitionOpenSinceMonth'] == 0) & (train_store['CompetitionOpenSinceYear'] == 0), 0, (train_store.Month - train_store.CompetitionOpenSinceMonth) +
                                                   (12 * (train_store.Year - train_store.CompetitionOpenSinceYear)))

    # we can get rid of `CompetitionOpenSinceYear` and `CompeitionOpenSinceMonth`
    del train_store['CompetitionOpenSinceYear']
    del train_store['CompetitionOpenSinceMonth']

    # data extraction
    # TODO: extract to sklearn pipelines
    test['Year'] = test.index.year
    test['Month'] = test.index.month
    test['Day'] = test.index.day
    test['WeekOfYear'] = test.index.weekofyear

    print(train_store.dtypes)
    # print(train_store[train_store['StateHoliday'].na()])
    # transform stateholiday
    train_store["StateHoliday"] = train_store['StateHoliday'].map(
        {"0": 0, "a": 1, "b": 1, "c": 1})
    features = test.columns.tolist()
    features.pop(0)
    features_df = train_store[features]
    print(features_df.head())
    targets = np.log(train_store.Sales)
    print(targets)
    # targets = float(targets)
    return features_df, targets

class data_preprocess:
    def __init__(self) -> None:
        """Initilize class."""
        try:
            pass
            #section left for logging information if class is initialized successfully
        except Exception:
            #section for logging class information in exception
            sys.exit(1)

    def get_numerical_columns(self, df):
        """Get numerical columns from dataframe."""
        
        num_col = df.select_dtypes(
            exclude="object").columns.tolist()
        return num_col
    
    def get_categorical_columns(self, df):
        """Get categorical columns from dataframe."""
        return df.select_dtypes(
                include="object").columns.tolist()
    
    def drop_duplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.drop_duplicates(subset='Date')

        return df
    
    def get_missing_values(self, df):
        
        return df.isnull().sum()

    def convert_to_datetime(self, df, column):
        df[column] = pd.to_datetime(df[column])
        return df

    def join_dataframes(self, df1, df2, on, how="inner"):
    
        return pd.merge(df1, df2, on=on)
        