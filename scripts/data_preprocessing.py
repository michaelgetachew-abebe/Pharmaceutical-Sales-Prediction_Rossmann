import os,sys

import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join('../logs')))

class data_preprocessor:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        