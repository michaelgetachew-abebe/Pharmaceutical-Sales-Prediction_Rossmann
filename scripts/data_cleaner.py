import pandas as pd
import numpy as np
from logger_creator import logwritter

class data_cleaner:

    def __init__(self, df: pd.DataFrame, deep=False) -> None:
        
        self.logger = logwritter(
            "../logs/data_cleaner_log.log").get_logwritter()
        if(deep):
            self.df = df.copy(deep=True)
        else:
            self.df = df

    def remove_unwanted_columns(self, columns: list) -> pd.DataFrame:
        self.df.drop(columns, axis=1, inplace=True)
        return self.df

    def separate_date_time_column(self, column: str, col_prefix_name: str) -> pd.DataFrame:
        
        try:

            self.df[f'{col_prefix_name}Date'] = pd.to_datetime(
                self.df[column]).dt.date
            self.df[f'{col_prefix_name}Time'] = pd.to_datetime(
                self.df[column]).dt.time

            return self.df

        except:
            print("Failed to separate the date-time column")

    def separate_date_column(self, date_column: str, drop_date=True) -> pd.DataFrame:
        try:
            date_index = self.df.columns.get_loc(date_column)
            self.df.insert(date_index + 1, 'Year', self.df[date_column].apply(
                lambda x: x.date().year))
            self.df.insert(date_index + 2, 'Month', self.df[date_column].apply(
                lambda x: x.date().month))
            self.df.insert(date_index + 3, 'Day',
                           self.df[date_column].apply(lambda x: x.date().day))

            if(drop_date):
                self.df = self.df.drop(date_column, axis=1)
        except:
            print("Failed to separate the date to its components")