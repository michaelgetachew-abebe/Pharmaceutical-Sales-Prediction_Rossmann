import sys
import os
import pandas as pd
cwd = os.getcwd()
sys.path.append("../scripts")
# sys.path.append("./dashboard")

import streamlit as st


st.set_page_config(page_title="TelCo Telecom Analytics", layout="wide")

# st.sidebar.markdown("Please select the desired page")

st.markdown("""
    ## Pharmaceutical Sales Prediction across multiple stores
    ### Week III: 10 Academy Challenge/Training 
    >This project is a part of [10Academy's](https://www.10academy.org/) challenge and is aimed to provide an end-to-end product that delivers this prediction to analysts in the finance team.
    ## Data
    - The data extracted from Kaggle's [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data) 
    
""")


st.markdown("""
        ### Files
        - train.csv - historical data including Sales
        - test.csv - historical data excluding Sales
        - sample_submission.csv - a sample submission file in the correct format
        - store.csv - supplemental information about the stores
        
        ### Data fields
        Most of the fields are self-explanatory. The following are descriptions for those that aren't.
        - Id - an Id that represents a (Store, Date) duple within the test set
        - Store - a unique Id for each store
        - Sales - the turnover for any given day (this is what you are predicting)
        - Customers - the number of customers on a given day
        - Open - an indicator for whether the store was open: 0 = closed, 1 = open
        - StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
        - SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
        - StoreType - differentiates between 4 different store models: a, b, c, d
        - Assortment - describes an assortment level: a = basic, b = extra, c = extended
        - CompetitionDistance - distance in meters to the nearest competitor store
        - CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
        - Promo - indicates whether a store is running a promo on that day
        - Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
        - Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
        - PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
""")



st.markdown("""
    - Data Cleaning
    - Data Preprocessing
    - Exploratory Data analysis
    - Trained a RandomForestRegressor Model
    - LSTM prediction
""")