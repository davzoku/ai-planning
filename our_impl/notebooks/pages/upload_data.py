import streamlit as st
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from helper_demand_functions import *

st.text("please upload product dataset")
uploaded_file_1 = st.file_uploader("Choose a csv files", type = 'csv')

st.text("please upload time dataset")
uploaded_file_2 = st.file_uploader("Choose a time csv file", type = 'csv')