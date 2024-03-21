import streamlit as st
import pandas as pd
import numpy as np
from collections import OrderedDict
#from sklearn.linear_model import ElasticNetCV, ElasticNet
#from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
#from helper_demand_functions import *
from utils import utils

def local_css(file_name):
    with open(file_name) as f:
        css = f.read()
        
    return css

st.set_page_config(
    page_title="Promotion optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title('Promotion optimization for retail stores')
#st.set_page_config(page_title = "AI Planning") 
#st.sidebar.success("navigation bar")


st.subheader('How to use this application')
st.markdown("""
    1. Upload your store datasets
    2. Optimize 
    """)

utils.add_logo()
# with open('./style.css') as f:
#     css = f.read()
#     #print(css)
    
st.subheader("Sample dataset format for store")
sample_data = pd.read_csv('./combined_milk_final.csv')
st.write(sample_data)
csv = sample_data.to_csv(index=False)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='sample_data.csv',
    mime='text/csv',
)

css = local_css("./style.css")
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

