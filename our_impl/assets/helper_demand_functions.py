
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

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

    return out_df

def sales_lag(df, sku = None, neg = True):
    out_df = df.copy()
    sales_col = f'{sku}_Sales' if sku is not None else 'Sales'
    out_df[sales_col] = -np.log(out_df[sales_col].shift(1)) if neg is True else np.log(out_df[sales_col].shift(1))
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
    max_column_name = f'{sku}_{promotype}neg' if neg is True else f'{promotype}'
    out_df[max_column_name] = -sum_values if neg is True else sum_values
    # Drop the columns used in the max calculation
    out_df.drop(columns=columns_to_max, inplace=True)
    return out_df


