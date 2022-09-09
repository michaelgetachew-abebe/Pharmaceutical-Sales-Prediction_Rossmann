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
        
    def is_weekend(self, date):

        return 1 if (date.weekday() > 4 or date.weekday() < 1) else 0
        
    def extract_fields_date(self, df, date_column):
        
        df['Year'] = df[date_column].dt.year
        df['Month'] = df[date_column].dt.month
        df['Day'] = df[date_column].dt.day
        df['DayOfWeek'] = df[date_column].dt.dayofweek
        df['weekday'] = df[date_column].dt.weekday
        df['weekofyear'] = df[date_column].dt.weekofyear
        df['weekend'] = df[date_column].apply(self.is_weekend)
        return df

    def label_encode(self, df, columns):

        label_encoded_columns = []
        # For loop for each columns
        for col in columns:
            # We define new label encoder to each new column
            le = LabelEncoder()
            # Encode our data and create new Dataframe of it,
            # notice that we gave column name in "columns" arguments
            column_dataframe = pd.DataFrame(
                le.fit_transform(df[col]), columns=[col])
            # and add new DataFrame to "label_encoded_columns" list
            label_encoded_columns.append(column_dataframe)

        # Merge all data frames
        label_encoded_columns = pd.concat(label_encoded_columns, axis=1)
        return label_encoded_columns

    def fill_missing_median(self, df, columns):
        
        for col in columns:
            df[col] = df[col].fillna(df[col].median())
        return df

    def get_missing_data_percentage(self, df):
        
        total = df.isnull().sum().sort_values(ascending=False)
        percent_1 = total/df.isnull().count()*100
        percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
        missing_data = pd.concat(
            [total, percent_2], axis=1, keys=['Total', '%'])
        return missing_data
    
    def fill_missing_with_zero(self, df, columns):
        
        for col in columns:
            df[col] = df[col].fillna(0)
        return df

    def replace_outliers_iqr(self, df, columns):
        
        for col in columns:
            Q1, Q3 = df[col].quantile(
                0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            cut_off = IQR * 1.5
            lower, upper = Q1 - cut_off, Q3 + cut_off

            df[col] = np.where(
                df[col] > upper, upper, df[col])
            df[col] = np.where(
                df[col] < lower, lower, df[col])
        return df
    