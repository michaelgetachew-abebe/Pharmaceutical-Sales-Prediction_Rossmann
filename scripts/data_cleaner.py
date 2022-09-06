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

    