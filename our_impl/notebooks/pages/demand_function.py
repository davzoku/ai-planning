
import streamlit as st
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from helper_demand_functions import *
from upload_data import *


if uploaded_file_1 and uploaded_file_2 is not None:
    data = pd.read_csv(uploaded_file_1)
    st.write("data from uploaded file:")
    st.write(data)
    
    calender = pd.read_csv(uploaded_file_2)
    st.write('data from uploaded file 2:')
    st.write(calender)


    store_id_input = st.number_input("enter a store_ID", min_value=0, format="%d")
    sku_id = st.text_input('enter a SKU_ID')

    store_df = data[data['Store_ID'] == store_id_input]
    store_sku_df = store_df[store_df['SKU'] == sku_id]

    st.write("store_SKU_data:")
    st.write(store_sku_df)

    #competitor data
    st.write('getting competitor data...')
    store_compet_sku_df = store_df[store_df['SKU'] != sku_id]

    compet_sku = store_df[store_df['SKU'] != sku_id]['SKU'].tolist()
    compet_sku = list(OrderedDict.fromkeys(compet_sku))

    store_compet_sku_df = store_compet_sku_df.pivot_table(index = ['Time_ID', 'Year', 'Store_ID'], columns = 'SKU', 
                                                        values = ['Price', 'Sales', 'Display1', 'Display2', 'Feature1', 'Feature2', 'Feature3', 'Feature4'])

    store_compet_sku_df.columns = ['_'.join([col[1], col[0]]) for col in store_compet_sku_df.columns]
    store_compet_sku_df.reset_index(inplace=True)

        
    store_compet_final = store_compet_sku_df.copy()
    for sku in compet_sku:
        store_compet_final = price_discount(store_compet_final, sku)
        store_compet_final = sum_columns(store_compet_final, sku, 'Feature')    # Negative Features
        store_compet_final = sum_columns(store_compet_final, sku, 'Display')    # Negative Display
        store_compet_final = sales_lag(store_compet_final, sku)                 # Negative Lag Sales 

    try:
        drop_cols = ['Store_ID', 'median_price']
        store_compet_final.drop(columns = drop_cols, inplace = True)
    
        st.write(store_compet_final)
    except:
        ""
    
    st.write('storing datasets...')
    window_size = st.number_input("enter a window size:", min_value=0, format="%d")
    store_sku_df.reset_index(drop = True, inplace= True)
    
    store_sku_df_part = store_sku_df.copy()
    store_sku_df_part = price_discount(store_sku_df_part)                                                                           # Price Column
    store_sku_df_part = sum_columns(store_sku_df_part, promotype = 'Feature', neg = False)                                          # Feature Column
    store_sku_df_part = sum_columns(store_sku_df_part, promotype = 'Display', neg = False)                                          # Display Column

    store_sku_df_part['Pricelag'] = -store_sku_df_part['Price'].shift(1)                                                            # Negative Price Lag Column
    store_sku_df_part['Featurelag'] = -store_sku_df_part['Feature'].shift(1)                                                        # Negative Feature Lag Column
    store_sku_df_part['Displaylag'] = -store_sku_df_part['Display'].shift(1)                                                        # Negative Display Lag Column

    store_sku_df_part['Saleslag'] = -np.log(store_sku_df_part['Sales'].shift(1))                                                    # Negative Log Sales Lag Column 
    store_sku_df_part['Sales_mov_avg'] = np.log(store_sku_df_part['Sales'].rolling(window = window_size).mean()).shift(1)           # Log Sales Rolling Mean Lag Column
    store_sku_df_part['Sales'] = np.log(store_sku_df_part['Sales'])                                                                 # Process Sales for Target Variable

    event_col = ['Halloween', 'Thanksgiving', 'Christmas', 'NewYear', 'President', 'Easter', 'Memorial', '4thJuly', 'Labour']
    event_cols = [col if col == 'NewYear' else [col, f'{col}_1'] for col in event_col]
    event_cols = [item for sublist in event_cols for item in ([sublist] if isinstance(sublist, str) else sublist)]
    calendar_cols = calender[['IRI Week']+ event_cols]
    calendar_cols = calendar_cols.fillna(0).astype(int)

    # Join Special Events to SKU Store Data
    store_sku_df_part = pd.merge(store_sku_df_part, calendar_cols, left_on='Time_ID', right_on='IRI Week', how='left')

    drop_cols = ['Discount','Store_ID', 'median_price', 'SKU', 'IRI Week']
    store_sku_df_part.drop(columns = drop_cols, inplace = True)
    
    st.write(store_sku_df_part)