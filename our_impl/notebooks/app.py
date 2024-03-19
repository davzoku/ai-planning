import streamlit as st
import pandas as pd
import numpy as np
from collections import OrderedDict
#from sklearn.linear_model import ElasticNetCV, ElasticNet
#from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from helper_demand_functions import *

st.title('Promotion optimization for retail stores')
#st.set_page_config(page_title = "AI Planning") 
st.sidebar.success("navigation bar")
st.image('cover_img.png')

st.header('How to use this application')
st.markdown("""
    1. Upload Data
    2. Run Demand Function to generate demand predictions
    3. Run GA to calculate predicted costs and profits
    """)
