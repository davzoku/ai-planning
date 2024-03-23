import streamlit as st
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import ElasticNetCV, ElasticNet, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
#from helper_demand_functions import *
from tqdm import tqdm
from stqdm import stqdm
from utils import utils
from sklearn.linear_model import Lasso

utils.add_logo()


def local_css(file_name):
    with open(file_name) as f:
        css = f.read()
        
    return css

css = local_css("./style.css")
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# class Upload:
#     @staticmethod
#     def upload_store_data():
#         st.text("Please upload product dataset")
#         uploaded_file_1 = st.file_uploader("Choose a csv files", type = 'csv')
        
#         if uploaded_file_1 is not None:
#             return uploaded_file_1
#         else:
#             st.error("No file uploaded. Please upload a CSV file")
#     @staticmethod
#     def upload_time_data():
#         st.text("Please upload time dataset")
#         uploaded_file_2 = st.file_uploader("Choose a time csv file", type = 'csv')
        
#         if uploaded_file_2 is not None:
#             return uploaded_file_2
        
#         else:
#             st.error("No file uploaded. Please upload a CSV file")
        

# def upload_store():
#     data = Upload.upload_store_data()
#     df = pd.read_csv(data)
#     st.write("Product datatset loaded!")
#     return data

# def upload_time():
#     data = Upload.upload_time_data()
#     df = pd.read_csv(data)
#     st.write("time datatset uploaded")
#     return data

    
def price_discount(df, sku = None, year_col='Year'):
    out_df = df.copy()
    price_col = f'{sku}_Price' if sku is not None else 'Price'
    # Calculate the lower bound as 95% of the maximum price within each year
    lower_bound = out_df.groupby(year_col)[price_col].transform(lambda x: np.max(x) * 0.95)
    # Filter the DataFrame based on the condition
    filtered_df = out_df[out_df[price_col] >= lower_bound]
    # Calculate the median for each group of time_year in the filtered DataFrame
    median_by_time_year = filtered_df.groupby(year_col)[price_col].median()
    # Copy the median values back into the original DataFrame by year_col
    out_df.loc[:,'median_price'] = out_df[year_col].map(median_by_time_year)
    # Compute the relative discount
    pc_disc =  out_df['median_price'] / out_df[price_col]
    
    
    z_scores = (pc_disc - np.mean(pc_disc)) / np.std(pc_disc)
    # Identify indices where the absolute z-score is greater than or equal to 3
    outlier_indices = np.where(np.abs(z_scores) >= 3)[0]
    # Replace outliers with the maximum value from X_pc_d excluding those outliers
    if len(outlier_indices) > 0:
        pc_disc[outlier_indices] = np.max(pc_disc[~np.isin(np.arange(len(pc_disc)), outlier_indices)])


    # Update Price column
    out_df.loc[:, price_col] = pc_disc

    return out_df , np.mean(pc_disc), np.std(pc_disc)


def sales_lag(df, sku = None, neg = True):
    out_df = df.copy()
    sales_col = f'{sku}_Sales' if sku is not None else 'Sales'
    out_df[sales_col] = out_df[sales_col].shift(1) if neg is True else out_df[sales_col].shift(1)
    return out_df


def sum_columns(df, sku = None, promotype = 'Feature', neg = True):
    out_df = df.copy()
    columns_to_max = [col for col in out_df.columns if col.startswith(sku+'_'+promotype)] if sku is not None else [col for col in out_df.columns if col.startswith(promotype)]
    if not columns_to_max:
        # print(f"No columns found with prefix '{sku}_{type}'")
        return out_df
    # Calculate the maximum values using numpy
    sum_values = np.sum(out_df[columns_to_max].values, axis=1)
    # Create a new column with the maximum values
    max_column_name = f'{sku}_{promotype}' if neg is True else f'{promotype}'
    out_df[max_column_name] = sum_values if neg is True else sum_values
    # Drop the columns used in the max calculation
    out_df.drop(columns=columns_to_max, inplace=True)
    return out_df


def demand_coef(data, calendar, store_id_input, sku_id, window_size, train_year_from, train_year_to, alphas):

    store_df = data[data['Store_ID'] == store_id_input]
    store_sku_df = store_df[store_df['SKU'] == sku_id]


    # Get competitor df
    store_compet_sku_df = store_df[store_df['SKU'] != sku_id]

    # Competitor sku list
    compet_sku = store_df[store_df['SKU'] != sku_id]['SKU'].tolist()
    compet_sku = list(OrderedDict.fromkeys(compet_sku))

    store_compet_sku_df = store_compet_sku_df.pivot_table(index = ['Time_ID', 'Year', 'Store_ID'], columns = 'SKU', 
                                                        values = ['Price', 'Sales', 'Display1', 'Display2', 'Feature1', 'Feature2', 'Feature3', 'Feature4'])

    store_compet_sku_df.columns = ['_'.join([col[1], col[0]]) for col in store_compet_sku_df.columns]
    store_compet_sku_df.reset_index(inplace=True)


    store_compet_final = store_compet_sku_df.copy()
    for sku in compet_sku:
        store_compet_final, _, _ = price_discount(store_compet_final, sku)
        store_compet_final = sum_columns(store_compet_final, sku, 'Feature')    # Features
        store_compet_final = sum_columns(store_compet_final, sku, 'Display')    # Display
        store_compet_final = sales_lag(store_compet_final, sku)                 # Lag Sales 

    drop_cols = ['Store_ID', 'median_price']
    store_compet_final.drop(columns = drop_cols, inplace = True)

    store_sku_df.reset_index(drop = True, inplace= True)

    # SKU Features
    store_sku_df_part = store_sku_df.copy()
    store_sku_df_part, dis_mean, dis_std = price_discount(store_sku_df_part)                                                  # Price Column
    store_sku_df_part = sum_columns(store_sku_df_part, promotype = 'Feature', neg = False)                                    # Feature Column
    store_sku_df_part = sum_columns(store_sku_df_part, promotype = 'Display', neg = False)                                    # Display Column

    store_sku_df_part['Pricelag'] = store_sku_df_part['Price'].shift(1)                                                      # Price Lag Column
    store_sku_df_part['Featurelag'] = store_sku_df_part['Feature'].shift(1)                                                  #  Feature Lag Column
    store_sku_df_part['Displaylag'] = store_sku_df_part['Display'].shift(1)                                                  #  Display Lag Column

    store_sku_df_part['Saleslag'] = store_sku_df_part['Sales'].shift(1)                                                     # Log Sales Lag Column 
    store_sku_df_part['Sales_mov_avg'] = store_sku_df_part['Sales'].rolling(window = window_size).mean().shift(1)           # Log Sales Rolling Mean Lag Column
    store_sku_df_part['Sales'] = store_sku_df_part['Sales']                                                                 # Process Sales for Target Variable


    # Add Special Events
    event_col = ['Halloween', 'Thanksgiving', 'Christmas', 'NewYear', 'President', 'Easter', 'Memorial', '4thJuly', 'Labour']
    event_cols = [col if col == 'NewYear' else [col, f'{col}_1'] for col in event_col]
    event_cols = [item for sublist in event_cols for item in ([sublist] if isinstance(sublist, str) else sublist)]
    calendar_cols = calendar[['IRI Week']+ event_cols]
    calendar_cols = calendar_cols.fillna(0).astype(int)

    # Join Special Events to SKU Store Data
    store_sku_df_part = pd.merge(store_sku_df_part, calendar_cols, left_on='Time_ID', right_on='IRI Week', how='left')

    drop_cols = ['Discount','Store_ID', 'median_price', 'SKU', 'IRI Week']
    store_sku_df_part.drop(columns = drop_cols, inplace = True)

    model_1 = Lasso(max_iter = 10000)
    model_2 = Lasso(max_iter = 10000)

    store_sku_part_trg = store_sku_df_part[(store_sku_df_part["Year"] >= train_year_from) & (store_sku_df_part["Year"] <= train_year_to)]
    store_sku_part_trg = store_sku_part_trg.iloc[window_size:]
    # store_sku_part_test = store_sku_df_part[(store_sku_df_part["Year"] == year_test)]

    store_compet_trg = store_compet_final[(store_compet_final["Year"] >= train_year_from) & (store_sku_df_part["Year"] <= train_year_to)]
    store_compet_trg = store_compet_trg.iloc[window_size:]
    # store_compet_test = store_compet_final[(store_compet_final["Year"] == year_test)]

    sku_sales_train = store_sku_part_trg['Sales']
    # sku_sales_test = store_sku_part_test['Sales']

    # Feature Variables
    # feature_list = []
    sku_train_drop = ['Time_ID', 'Year', 'Sales']
    compet_train_drop = ['Time_ID', 'Year']
    store_sku_part_trg = store_sku_part_trg.drop(columns = sku_train_drop)
    # store_sku_part_test = store_sku_part_test.drop(columns = sku_train_drop)
    store_compet_trg = store_compet_trg.drop(columns = compet_train_drop)
    # store_compet_test = store_compet_test.drop(columns = compet_train_drop)

    # positive_features_1 = ['Price', 'Feature', 'Display', 'Pricelag' ,'Featurelag', 'Displaylag', 'Saleslag']
    # model_1.positive = positive_features_1 
    model_1.fit(store_sku_part_trg, sku_sales_train)
    sku_sales_train_rsd = sku_sales_train - model_1.predict(store_sku_part_trg)

    # positive_features_2 = [col for col in store_compet_trg.columns if col.endswith(("_Display", "_Feature", "_Price"))]
    # model_2.positive = positive_features_2
    model_2.fit(store_compet_trg, sku_sales_train_rsd)

    model_1_df = pd.DataFrame(model_1.coef_, index = store_sku_part_trg.columns, columns=[sku_id] )
    model_2_df = pd.DataFrame(model_2.coef_, index = store_compet_trg.columns, columns=[sku_id] )

    bias_term = model_1.intercept_ + model_2.intercept_

    coef_df = pd.concat([model_1_df, model_2_df])

    return coef_df, dis_mean, dis_std, bias_term

#data = pd.read_csv(file_dir)
#calendar = pd.read_csv(time_dir)

# self.store_id_input = 236117
# self.window_size = 8
# self.year_from = 2001
# self.year_to = 2005
# self.year_test = 2006
# self.alphas = np.logspace(-4, 0, 100)

# store_data =  Upload.upload_store_data()
# time_data =  Upload.upload_time_data()

class Solution:
    def __init__(self, store_id_input, window_size, year_from, year_to, year_test, alphas=np.logspace(-4, 0, 100)):
        self.store_id_input = store_id_input
        self.window_size = window_size
        self.year_from = year_from
        self.year_to = year_to
        self.year_test = year_test
        self.alphas = alphas

    def calculation(self):
        if store_data:
            data = pd.read_csv(store_data)
            
                    
        if time_data:
            calendar = pd.read_csv(time_data)
        
        
        base_features = ['Price', 'Feature', 'Display', 'Pricelag', 'Featurelag', 'Displaylag', 
                 'Saleslag', 'Sales_mov_avg', 'Halloween', 'Halloween_1', 'Thanksgiving',
                 'Thanksgiving_1', 'Christmas', 'Christmas_1', 'NewYear', 'President',
                 'President_1', 'Easter', 'Easter_1', 'Memorial', 'Memorial_1', '4thJuly', '4thJuly_1', 'Labour', 'Labour_1']

        unique_skus = list(data[data['Store_ID'] == self.store_id_input]['SKU'].unique())

        new_index_parts = []
        for code in unique_skus:
            new_index_parts.extend([
                f'{code}_Price',
                f'{code}_Display',
                f'{code}_Feature',
                f'{code}_Sales'
            ])

        # Combine the base features with the new index parts
        complete_index = base_features + new_index_parts

        # Create an empty DataFrame with the specified list of row indexes
        final_df = pd.DataFrame(index=complete_index)
        #st.write(final_df)

        df_z_score = pd.DataFrame(columns=["Mean", "Std_deviation", 'bias'])

        for sku in stqdm(unique_skus):
            df_coef, mean, std, bias = demand_coef(data, calendar, self.store_id_input, sku, self.window_size, self.year_from, self.year_to, self.alphas)
            final_df = final_df.join(df_coef, how = 'left').fillna(0)
            df_z_score.loc[sku] = [mean, std, bias]

        final_df.index = [idx.replace('Price', 'Discount') for idx in final_df.index]
        
        return final_df, df_z_score

store_data_uploader = st.file_uploader("Upload Store Data", type=['csv'])
time_data_uploader = st.file_uploader("Upload Time Data", type=['csv'])


#if st.button('Start Calculations'):
if store_data_uploader is not None and time_data_uploader is not None:
    # Assuming 'Upload' class methods are used to save or process uploaded files
    store_data = store_data_uploader  # Modify according to your actual file handling
    time_data = time_data_uploader  # Modify according to your actual file handling
    
    # Your Solution class instantiation and calculations
    #instance = Solution(store_id_input=236117, window_size=8, year_from=2001, year_to=2005, year_test=2006)
        #result = instance.calculation()  # Ensure this is callable
    #st.session_state['result'] = instance.calculation()
        
st.write('Target values for optimization')
with st.form(key='all_inputs_form_2'): 
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            store_id_input = st.text_input("Store id Input")
        with c2:
            window_size = st.selectbox('Window size',['4', '8', '16'])
        with c3:
            year_from = st.text_input('Year from')
        with c4:
            year_to = st.text_input('Year to')
        with c5:
            year_test = st.text_input('Year test')
        
        submit_button = st.form_submit_button(label='Submit')
    
if submit_button:
    instance = Solution(store_id_input=int(store_id_input), window_size=int(window_size), year_from=int(year_from), year_to=int(year_to), year_test=int(year_test))
    st.session_state['coeff'], st.session_state['z_score'] = instance.calculation()



# if st.button('Start Calculations'):
#     instance = Solution(236117, 8, 2001, 2005, 2006)
#     result = instance.calculation()
    # st.write(result)