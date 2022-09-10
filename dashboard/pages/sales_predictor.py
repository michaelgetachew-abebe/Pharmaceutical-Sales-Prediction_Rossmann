from operator import mod
import sys
import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')

cwd = os.getcwd()
sys.path.append(f'{cwd}/scripts/')

from preprocessor import Preprocessor
from data_cleaner import DataCleaner
from util import Util
from model import Model

preprocess = Preprocessor()
util = Util()
cleaner= DataCleaner()

cols = ['Store', 'Customers', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment','CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promos2SinceWeek', 'Promo2SinceYear','PromoInterval', 'Open','Date', 'DayOfWeek'  ]
    


def plot_predictions(date, sales):
    fig = plt.figure(figsize=(20, 7))
    ax = sns.lineplot(x=date, y=sales)
    ax.set_title("Predicted Sales", fontsize=24)
    ax.set_xlabel("Row index", fontsize=18)
    ax.set_ylabel("Sales", fontsize=18)
    
    # fig = plt.figure(figsize=(18, 5))
    

    return fig 




st.title("Rossmann Pharmaceuticals Sales Forecaster")

# @st.cache


input_data = pd.DataFrame(columns=cols)
submitted = False

df = None
file=True
data_added = False
uploaded_file = None
model = None

option = st.selectbox('Select Input Preference',("","File Upload (CSV)","Manual Input"))

if option == "File Upload (CSV)":
    file=True
    uploaded_file = st.file_uploader("Upload CSV to predict sales", type=".csv")
elif option =="Manual Input":
    file=False
    values = []
    with st.form(key='inference', clear_on_submit=True):
        
        for feature in cols:
            
            values.append(st.text_input(feature,key=feature))

        submitted = st.form_submit_button("Show Prediction")

else:
    st.warning("Please Choose an option to insert your data")   
    

if not file and submitted:

    
    data = pd.DataFrame(values,index=cols).T
    st.table(data.T)
    data['Date'] = pd.to_datetime(data['Date'])
    
    data_added = True

elif file and (uploaded_file is not None):
    print("File uploaded")
    st.success("File uploaded, Below is your file preview")
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    st.dataframe(data)
    data_added = True
    

@st.cache
def load_model():
    model=util.read_model_dvc("models/09-09-2022-11-31-48.pkl",repo,"random-forest-v0",low_memory=False)
    return model


if data_added:

    
    repo="https://github.com/Nathnael12/pharmaceutical-sales-pridiction.git"
    with st.spinner("Please wait a moment"):
        # df=preprocess.prepare(data)
        df=preprocess.format_dtype(data)
        # st.table(df)
        df=cleaner.feature_encodder(df)
        # st.table(df)
        df = preprocess.feature_engineering(df)
        # st.table(df)
        
        # model = load_model()
        model = pickle.load(open(f'{cwd}/models/09-09-2022-11-31-48.pkl', 'rb'))
        
        prediction = model.predict(df)
        df["Date"]=pd.DatetimeIndex(df["Day"].astype(str)+'-' + df["Month"].astype(str)+'-' + df["Year"].astype(str))
        

   

    fig = plot_predictions(df['Date'], prediction)
    download=pd.DataFrame(index=df['Date'],columns=["Sales Prediction"],data=prediction).to_csv()
    st.pyplot(fig)

    st.download_button("Download CSV",data=download,file_name="Prediction.csv")