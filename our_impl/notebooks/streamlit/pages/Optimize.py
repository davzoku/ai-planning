import time
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
import numpy as np
from pymoo.core.sampling import Sampling
from my_utils import utils

from pymoo.operators.crossover.pntx import (
    PointCrossover,
    SinglePointCrossover,
    TwoPointCrossover,
)
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
import streamlit as st
import pandas as pd
import sys
import os

if 'coeff' in st.session_state:
    st.success("Coefficient data loaded!")
    coeff = pd.DataFrame(st.session_state['coeff'])
else:
    coeff = pd.read_csv('assets/coefficients.csv')
    st.warning("Pre-loaded data taken")
    
if 'z_score' in st.session_state:
    st.success("Z-Score data loaded!")
    z_score = pd.DataFrame(st.session_state['z_score'])
    z_score = z_score.reset_index()
    z_score.rename(columns={z_score.columns[0]: 'SKU'}, inplace=True)
    st.write('zscore loaded',z_score)
else:
    z_score = pd.read_csv('assets/Z_scores.csv')
    st.warning("Pre loaded data loaded")

scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.append(scripts_dir)

from scripts import revenue_estimation
from scripts import pop_ga

utils.add_logo()

with st.form(key="all_inputs_form"):
    c1, c2 = st.columns(2)
    with c1:
        sku_num = st.number_input("Number of SKUs", value=0)
    with c2:
        week_horizon = st.number_input("Week Horizon", value=0)

    st.write("---")

    c3, c4, c5 = st.columns(3)
    # with c5:
    #     ndf = st.number_input("Number of display features", value=0, key="ndf")
    with c3:
        lim_pro_per_cate_xu = st.number_input(
            "Max Number of Promotion Items per Week", value=0, key="lim_pro_per_cate_xu"
        )
    with c4:
        lim_dis_per_cate_xu = st.number_input(
            "Max Number of Display Items per Week", value=0, key="lim_dis_per_cate_xu"
        )
    with c5:
        lim_fea_per_cate_xu = st.number_input(
            "Max Number of Feature Items per Week", value=0, key="lim_fea_per_cate_xu"
        )

    st.write("---")

    c6, c7 = st.columns(2)

    with c6:
        pop_size = st.number_input("GA Population Size", value=10, key="pop_size")
    with c7:
        gen_size = st.number_input(
            "GA Number of Generations", value=100, key="gen_size"
        )

    submit_button = st.form_submit_button(label="Submit")

def process_data():
    file_dir = 'assets/combined_milk_final.csv'
    time_dir = 'assets/time.csv'
    #zscore_dir = z_score
    raw = pd.read_csv(file_dir)
    calendar = pd.read_csv(time_dir)
    zscore = z_score
    #st.write('zscore', zscore)

    sales = raw[raw['Store_ID'] == 236117].copy()
    # sales['Display'] = np.maximum(sales['Display1'], sales['Display2'])
    # sales['Feature'] = np.maximum.reduce([sales['Feature1'], sales['Feature2'], sales['Feature3'], sales['Feature4']])
    sales.loc[:, 'Display'] = np.maximum(sales['Display1'], sales['Display2'])
    sales.loc[:, 'Feature'] = np.maximum.reduce([sales['Feature1'], sales['Feature2'], sales['Feature3'], sales['Feature4']])

    ## Build rolling L8W Avg Sales & L7W Sum Sales
    sales.sort_values(by=['SKU', 'Time_ID'], inplace=True)
    sales['Lag8w_avg_sls'] = sales.groupby('SKU')['Sales'].transform(lambda x: x.rolling(window=8, min_periods=1).mean())
    sales['Lag7w_sum_sls'] = sales.groupby('SKU')['Sales'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
    sales['Log_sls'] = -np.log(sales['Sales'])

    ## Calcualte Price Discount (Discount Index) from Landing Price
    ## Get lower bound price of 95 percentile prices of each SKUxYear
    lb_prices =  sales.groupby(['SKU', 'Year'])['Price'].max() * 0.95
    lb_prices = lb_prices.reset_index(name='lb_price')
    sales = pd.merge(sales, lb_prices, on=['SKU', 'Year'], how='left')
    
    ## Filter prices for top 5% and get Median Price 
    med_prices = sales[sales['Price'] >= sales['lb_price']][['SKU', 'Year', 'Price']]
    med_prices = med_prices.groupby(['SKU', 'Year'])['Price'].median().reset_index(name='med_price')
    sales = pd.merge(sales, med_prices, on=['SKU', 'Year'], how='left')
    ## Calculate Discount Index
    sales['pc_disc'] = sales['med_price'] / sales['Price']

    #st.write(sales)
    ## Apply z-standardization on discount
    sales = pd.merge(sales, zscore, on=['SKU'], how='left')
    sales['z_disc'] = ( sales['pc_disc'] - sales['Mean'] ) / sales['Std_deviation']

    ## Clean up table
    sales = sales[['SKU', 'Time_ID', 'Year', 'Sales', 'z_disc', 'Display', 'Feature', 'Log_sls', 'Lag8w_avg_sls', 'Lag7w_sum_sls']]
    sales = sales.rename(columns={'z_disc': 'Discount'})

    cal_week = calendar[['IRI Week', 'Calendar week starting on', 'Calendar week ending on']]
    cal_week = cal_week.rename(columns={'IRI Week': 'Time_ID', 
                                        'Calendar week starting on': 'Start_Date', 
                                        'Calendar week ending on': 'End_Date'})

    #cal_week.to_csv('calendar_week.csv', index=False)
    events = calendar[['IRI Week', 'Halloween', 'Halloween_1', 'Thanksgiving', 'Thanksgiving_1', 'Christmas', 'Christmas_1', 'NewYear', 'President', 'President_1', 'Easter', 'Easter_1', 'Memorial', 'Memorial_1', '4thJuly', '4thJuly_1', 'Labour', 'Labour_1']]
    events = events.rename(columns={'IRI Week': 'Time_ID'})
    events.fillna(0, inplace=True)
    
    del lb_prices
    
    return sales, med_prices, cal_week, events


def ga_demand():
    sales, med_prices, cal_week, events = process_data()
    sales = sales
    cal_week = cal_week
    events = events
    dd_coeff = coeff
    prices = med_prices
    all_skus = sorted(sales['SKU'].unique())

    time_year = sales[['Time_ID', 'Year']].copy()
    time_year = time_year.drop_duplicates()

    ## To comment out later parameters come from GA function
    sku_list = ['7_1_42365_22800', '88_6_99998_59504', '88_6_99998_59509','88_6_99998_59597', '7_1_42365_26400']
    sku_list = sorted(sku_list)
    start = 1375
    period = 8

    ## To comment later ga_output comes from GA function
    ga_output = {
        'Discount': np.random.choice(np.arange(5, 55, 5)/100, len(sku_list)*period) ,  # Random discounts
        'Feature': np.random.choice([0, 1], len(sku_list)*period),  # Random features
        'Display': np.random.choice([0, 1], len(sku_list)*period) # Random displays
    }
    ga_output = pd.DataFrame(ga_output)
    
    #combined the ga_demand function
    histr = start - 1
    end =  start + period

    idx_frame = [(SKU, Time_ID) for SKU in sku_list for Time_ID in range(start, end)]
    idx_frame = pd.DataFrame(idx_frame, columns=['SKU', 'Time_ID'])

    sales_hist = sales[(sales['SKU'].isin(sku_list)) & (sales['Time_ID']>=histr-period-5) & (sales['Time_ID']<=histr)].copy()

    ## Preapare GA dataframe
    ga_df = ga_output.copy()
    ga_df = pd.concat([idx_frame, ga_df], axis=1)
    ga_df = pd.merge(ga_df, z_score, on=['SKU'], how='left')
    ga_df['z_disc'] = (ga_df['Discount'] - ga_df['Mean']) / ga_df['Std_deviation']
    ga_df = ga_df[['SKU', 'Time_ID', 'z_disc', 'Feature', 'Display']]
    ga_df = ga_df.rename(columns={'z_disc': 'Discount'})

    ## Create Competitor Matrix
    comp_matrix_columns = [f'{sku}_{promo}' for sku in all_skus for promo in ['Discount', 'Display', 'Feature', 'Sales']]
    comp_matrix = pd.DataFrame(columns=comp_matrix_columns, index=range(len(sku_list)*period))
    comp_matrix = pd.concat([idx_frame, comp_matrix], axis=1)
    for sku in sku_list:
        for promo in ['Discount', 'Display', 'Feature']:
            neg = -1 if promo in ['Display', 'Feature'] else 1
            tmp = list(ga_df[ga_df['SKU']==sku][promo] * neg) * period
            tmp = pd.DataFrame(tmp)
            comp_matrix[sku + "_" + promo] = tmp
            comp_matrix.loc[comp_matrix['SKU'] == sku, [sku + "_" + promo]] = 0
    comp_matrix.fillna(0, inplace=True)

    # comp_matrix.head()

    revenue = []

    ## Iterate through each week through the demand function
    ## Obtain sales prediction and feed back into historical sales for picking\

    for week in range(start, end):

        dd_coeff_val = dd_coeff[sku_list].values
        year = time_year[time_year['Time_ID']==week]['Year'].values[0]
        ga_tmp = ga_df[ga_df['Time_ID']==week].copy()
        for promo in ['Discount', 'Feature', 'Display']:
            merge = sales_hist[sales_hist['Time_ID']==week-1][['SKU', promo]].copy()
            merge = merge.rename(columns={promo: promo+"lag"})
            ga_tmp = pd.merge(ga_tmp, merge, on=['SKU'], how='left')

        merge = sales_hist[sales_hist['Time_ID']==week-1][['SKU', 'Log_sls', 'Lag8w_avg_sls']].copy()
        merge = merge.rename(columns={'Log_sls': 'Saleslag', 'Lag8w_avg_sls': 'Sales_mov_avg'})
        ga_tmp = pd.merge(ga_tmp, merge, on=['SKU'], how='left')

        events_tmp = events[events['Time_ID']==week].drop(columns = 'Time_ID').copy()
        events_tmp = pd.concat([events_tmp]*len(sku_list), ignore_index=True)
        ga_tmp = pd.concat([ga_tmp, events_tmp], axis=1)

        comp_tmp = comp_matrix[comp_matrix['Time_ID']==week].drop(columns = 'Time_ID').copy()
        for sku in sku_list:
            tmp = sales_hist[(sales_hist['SKU']==sku) & (sales_hist['Time_ID']==week-1)]['Sales'].item()
            comp_tmp[sku+"_Sales"] = tmp
            comp_tmp.loc[comp_tmp['SKU'] == sku, [sku + "_Sales"]] = 0

        ga_tmp = pd.merge(ga_tmp, comp_tmp, on=['SKU'], how='left')

        ga_val = ga_tmp.drop(columns=['SKU', 'Time_ID']).values

        sales_output = np.diag(ga_val.dot(dd_coeff_val))
        prices_tmp = prices[(prices['SKU'].isin(sku_list)) & (prices['Year']==year)]['med_price'].values

        revenue.append(sum(sales_output * prices_tmp))

        ## Prep for historical insert
        prep_tmp = sales_hist[sales_hist['Time_ID']==week-1][['SKU', 'Lag7w_sum_sls']].copy()
        hist_prep = sales_hist[sales_hist['Time_ID']==week-7][['SKU', 'Sales']]
        hist_prep = hist_prep.rename(columns={'Sales': 'Lag7w_sls'})
        hist_prep = pd.merge(hist_prep, prep_tmp, on=['SKU'], how='left')
        hist_prep = hist_prep.drop(columns=['SKU'])

        ## Build historical insert
        hist_insert = ga_tmp[['SKU', 'Time_ID', 'Discount', 'Display', 'Feature']]
        hist_insert['Year'] = year
        hist_insert['Sales'] = sales_output
        hist_insert['Log_sls'] =  -np.log(hist_insert['Sales'])
        hist_insert = pd.concat([hist_insert, hist_prep], axis=1)
        hist_insert['Lag8w_avg_sls'] = ( hist_insert['Lag7w_sum_sls'] + hist_insert['Sales'] ) / 8
        hist_insert['Lag7w_sum_sls_upd'] = hist_insert['Lag7w_sum_sls'] - hist_insert['Lag7w_sls'] + hist_insert['Sales']
        hist_insert = hist_insert[['SKU', 'Time_ID', 'Year', 'Sales', 'Discount', 'Display', 'Feature', 'Log_sls', 'Lag8w_avg_sls', 'Lag7w_sum_sls_upd']]
        hist_insert = hist_insert.rename(columns={'Lag7w_sum_sls_upd': 'Lag7w_sum_sls'})
        hist_insert.fillna(0, inplace=True)

        ## Insert results into historical
        sales_hist = pd.concat([sales_hist, hist_insert], ignore_index=True)
    
    return sum(revenue)

sales, med_prices, cal_week, events = process_data()
# st.write('sales df', sales)
# st.write('price df', med_prices)
# st.write('week', cal_week)
# st.write('events', events)

if submit_button:
    # TODO: convert to frontend variables
    start_week = 1375 #search the time id from cal_week dataframe. put start week as a frontend variable with drop down list
    period = week_horizon
    #zscore = pd.read_csv("assets/Z_scores.csv")
    selected_sku_list = z_score["SKU"].tolist()

    POP_SIZE = pop_size
    GEN_SIZE = gen_size
    # Fixed Variables
    price_opt_number = 4
    ndf = 2
    CFG = {
        "sku_num": int(sku_num),  # no of sku
        "h": week_horizon,  # week horizon
        "price_opt_num": price_opt_number,  # num of pricing options dvar
        "ndf": ndf,  # num of display, feature dvar
        "lim_pro_per_cate_xu": lim_pro_per_cate_xu,  # num of promotion items upper bound
        "lim_dis_per_cate_xu": lim_dis_per_cate_xu,  # num of display items upper bound
        "lim_fea_per_cate_xu": lim_fea_per_cate_xu,  # num of feature items upper bound
        "constraint_num": 2,  # simplify to 2 high level constraints
    }
    print(f"{CFG=}")
    # st.write(CFG)
    PRICE_LIST = pop_ga.init_price_list(CFG["sku_num"], CFG["h"])
    # st.write(PRICE_LIST)

    #try:
    print(sales)
    print(f"init problem")
    # TODO: convert to use computed demand model DFs
    rev_est = revenue_estimation.RevenueEstimation(
        # sales_dir="assets/processed_sales.csv", # process_data_sls.ipynb
        # cal_dir="assets/calendar_week.csv",# process_data_sls.ipynb
        # events_dir="assets/events.csv", # process_data_sls.ipynb
        # zscore_dir="assets/Z_scores.csv", # demand func .ipynb
        # dd_coeff_dir="assets/Coefficients.csv", # demand func .ipynb
        # prices_dir="assets/prices.csv", # process_data_sls.ipynb            
        sales=sales,
        cal_week=cal_week,
        events=events,
        zscore=z_score,
        dd_coeff=coeff,
        prices=med_prices,
    )
    print('rev', rev_est)
    problem = pop_ga.PromotionOptimizationProblem(
        cfg=CFG,
        rev_est=rev_est,
        selected_sku_list=selected_sku_list,
        start_week=start_week,
        period=period,
    )
    print('prob', problem)
    algo_best_ops = GA(
        pop_size=POP_SIZE,
        sampling=pop_ga.SKUPopulationSampling(cfg=problem.cfg, pop_size=POP_SIZE),
        # mutation=pop_ga.SwapMutation(cfg=problem.cfg, prob=0.4),
        # crossover=SBX(prob=0.2, eta=15),
        mutation=PolynomialMutation(prob=0.6, eta=20),
        crossover=TwoPointCrossover(prob=0.4),
        eliminate_duplicates=True,
    )
    print('best ops', algo_best_ops)
    # label = "SBX 0.2 Swap Mutation 0.4"
    label = "TwoPointCrossover 0.4, PM 0.6"

    start_time = time.time()
    res = minimize(
        problem,
        algo_best_ops,
        ("n_gen", GEN_SIZE),
        seed=2,
        save_history=True,
        verbose=True,
    )
    end_time = time.time()

    time = end_time - start_time

    rows = CFG["sku_num"] * CFG["h"]
    cols = CFG["price_opt_num"] + CFG["ndf"]
    formatted_schedule = problem._to_discount_values(res.X.reshape((rows, cols)))

    st.dataframe(formatted_schedule)

    revenue = -res.F[0]  # reverse the sign back to pos
    formatted_revenue = f"{revenue:.2f}"
    formatted_time = f"{time:.2f}"

    markdown_text = f"""
    **Expected Revenue:** ${formatted_revenue}  

    **Time Taken to Compute:** {formatted_time} seconds
    """

    st.markdown(markdown_text)

    # Plotting history (https://pymoo.org/misc/convergence.html)
    fig, ax = plt.subplots()
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history])
    ax.plot(n_evals, opt, "--", label=label)

    ax.set_title("Analysis of Convergence")
    ax.set_xlabel("Number of Evaluations")
    ax.set_ylabel("f min")
    ax.legend(loc="upper right")
    # plt.show()
    st.pyplot(fig)
    #except:
    #    "Please input the values for the GA to run :)"