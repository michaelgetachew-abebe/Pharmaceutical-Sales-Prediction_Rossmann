from cmath import log
import os,sys

import pandas as pd
import numpy as np

from logger_creator import logwritter
# Create an instance of the logwriter class

#log_writter = logwritter("logs/data_preprocessing_logs.log").get_logwritter()

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
        self.logger.info("S")