import streamlit as st
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from helper_demand_functions import *


class Upload:
    @staticmethod
    def upload_store_data():
        st.text("Please upload product dataset")
        uploaded_file_1 = st.file_uploader("Choose a csv files", type = 'csv')
        
        if uploaded_file_1 is not None:
            return uploaded_file_1
        else:
            st.error("No file uploaded. Please upload a CSV file")
    @staticmethod
    def upload_time_data():
        st.text("Please upload time dataset")
        uploaded_file_2 = st.file_uploader("Choose a time csv file", type = 'csv')
        
        if uploaded_file_2 is not None:
            return uploaded_file_2
        
        else:
            st.error("No file uploaded. Please upload a CSV file")
        

store_data = Upload.upload_store_data()
time_data = Upload.upload_time_data()

if store_data:
    data = pd.read_csv(store_data)
    st.write("Product datatset:")
    st.write(data)
    
    
if time_data:
    calender = pd.read_csv(time_data)
    st.write('Time dataset:')
    st.write(calender)
