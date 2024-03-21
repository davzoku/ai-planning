import streamlit as st
import pandas as pd
import numpy as np
from collections import OrderedDict
#from sklearn.linear_model import ElasticNetCV, ElasticNet
#from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
#from helper_demand_functions import *
from utils import utils

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

utils.add_logo()

# CUSTOM_CSS = """
# <style>
# .custom-nav {
#     display: flex;
#     align-items: center;
#     padding: 10px;
#     background-color: #f0f2f6; /* Light grey background */
# }
# .custom-logo {
#     height: 50px; /* Adjust based on your logo's size */
#     margin-right: 20px;
# }
# .custom-title {
#     font-size: 24px;
#     font-weight: bold;
#     color: #333; /* Dark grey text */
# }
# </style>
# """

# # Path or URL to your logo
# logo_path = "path/to/your/logo.png"  # Update this to your logo's path or URL

# # Inject custom HTML with your logo at the top of the app
# # st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# # st.markdown(f"""
# # <div class="custom-nav">
# #     <img src="{logo_path}" class="custom-logo">
# #     <div class="custom-title">OptiPromo</div>
# # </div>
# # """, unsafe_allow_html=True)

# st.markdown(
#     """
#     <style>
#         [data-testid="stSidebarNav"] {
#             background-image: "/logo.png";
#             background-repeat: no-repeat;
#             padding-top: 120px;
#             background-position: 20px 20px;
#         }
#         [data-testid="stSidebarNav"]::before {
#             content: "OptiPromo";
#             margin-left: 20px;
#             margin-top: 20px;
#             font-size: 30px;
#             position: relative;
#             top: 100px;
#             padding:20px;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )