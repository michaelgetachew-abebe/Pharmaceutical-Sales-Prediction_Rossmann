from cmath import log
import os,sys

import pandas as pd
import numpy as np

from logger_creator import logwritter
# Create an instance of the logwriter class

#log_writter = logwritter("../logs/data_preprocessing_logs.log").get_logwritter()

sys.path.append(os.path.abspath(os.path.join('../logs')))


class data_preprocessor:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.logger = logwritter("logs/data_preprocessing_log.log").get_logwritter()

    def drop_duplicates(self) -> pd.DataFrame:
        dropped = self.df[self.df.duplicated()].index
        self.logger.info(f"Dropping duplicates: {dropped}")
        return self.df.drop(index=dropped, inplace=True)

    def see_datatypes(self) -> pd.DataFrame:
        data_types = self.df.dtypes
        self.logger.info(f"Data Types: {data_types}")
        return self.df.dtypes

    def show_data_description(self) -> pd.DataFrame:
        description = self.df.describe()
        self.logger.info(f"Description : {description}")
        return self.df.describe()

    def show_data_info(self) -> pd.DataFrame:
        self.logger.info(f"Displaying data Information")
        return self.df.info()

    def show_statistical_info(self) -> pd.DataFrame:
        self.logger.info(f"Showing statistical info")
        return self.df.agg(['mean'])

    def show_correlation(self) -> pd.DataFrame:
        self.logger.info(f"Showing correlation")
        return self.df.corr()

    def collective_grouped_mean(self, colomnName: str) -> pd.DataFrame:
        groupby_colomnName = self.df.groupby(colomnName)
        self.logger.info(f"Collective grouped mean")
        return groupby_colomnName.mean()

    def list_coloumn_names(self) -> pd.DataFrame:
        self.logger.info(f"Showing coloumn names")
        return self.df.columns

    def colums_WithMissingValue(self):
        miss = []
        dff = self.df.isnull().any()
        summ = 0
        for col in dff:
            if col == True:
                miss.append(dff.index[summ])
            summ += 1
        self.logger.info(f"Colums with missing values: {miss}")
        return miss

    def get_column_based_missing_percentage(self):
        col_null = self.df.isnull().sum()
        total_entries = self.df.shape[0]
        missing_percentage = []
        for col_missing_entries in col_null:
            value = str(
                round(((col_missing_entries/total_entries) * 100), 2)) + " %"
            missing_percentage.append(value)

        missing_df = pd.DataFrame(col_null, columns=['total_missing_values'])
        missing_df['missing_percentage'] = missing_percentage
        self.logger.info(f"Showing missing percentage")
        return missing_df
