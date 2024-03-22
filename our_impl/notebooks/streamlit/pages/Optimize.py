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


def local_css(file_name):
    with open(file_name) as f:
        css = f.read()
        
    return css


css = local_css("./style.css")
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

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
sku_list = z_score["SKU"].tolist()
options = ["All"] + sku_list
num_of_sku = len(sku_list)

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

sales, med_prices, cal_week, events = process_data()
start_week_list = cal_week['Start_Date'].to_list()

with st.form(key="all_inputs_form"):
    c1, c2, c3 = st.columns(3)
    
    with c1:
        week_horizon = st.number_input("Week Horizon", value=0)

    with c2:
        selected_sku_list = st.multiselect("Select SKU", options, default="All")
        if "All" in selected_sku_list:
            if len(selected_sku_list) > 1:
                selected_sku_list = ["All"]
                st.warning("If 'All' is selected, no other selections are allowed.")
                st.multiselect("Select SKU", options, default="All")
    with c3:
        start_week = st.selectbox("Start week", start_week_list)
    
    st.write("---")

    c4, c5, c6 = st.columns(3)
    # with c5:
    #     ndf = st.number_input("Number of display features", value=0, key="ndf")
    with c4:
        lim_pro_per_cate_xu = st.number_input(
            "Max Number of Promotion Items per Week", value=0, key="lim_pro_per_cate_xu"
        )
    with c5:
        lim_dis_per_cate_xu = st.number_input(
            "Max Number of Display Items per Week", value=0, key="lim_dis_per_cate_xu"
        )
    with c6:
        lim_fea_per_cate_xu = st.number_input(
            "Max Number of Feature Items per Week", value=0, key="lim_fea_per_cate_xu"
        )

    st.write("---")

    c7, c8 = st.columns(2)

    with c7:
        pop_size = st.number_input("GA Population Size", value=10, key="pop_size")
    with c8:
        gen_size = st.number_input(
            "GA Number of Generations", value=100, key="gen_size"
        )

    submit_button = st.form_submit_button(label="Submit")


if submit_button:
    time_id = cal_week[cal_week['Start_Date'] == start_week]['Time_ID'].iloc[0]
    st.write('time ID', time_id)
    start_week = time_id 
    period = week_horizon
    #zscore = pd.read_csv("assets/Z_scores.csv")
    selected_sku_list = z_score["SKU"].tolist()

    POP_SIZE = pop_size
    GEN_SIZE = gen_size
    # Fixed Variables
    price_opt_number = 4
    ndf = 2
    CFG = {
        "sku_num": num_of_sku,  # no of sku
        "h": week_horizon,  # week horizon
        "price_opt_num": price_opt_number,  # num of pricing options dvar
        "ndf": ndf,  # num of display, feature dvar
        "lim_pro_per_cate_xu": lim_pro_per_cate_xu,  # num of promotion items upper bound
        "lim_dis_per_cate_xu": lim_dis_per_cate_xu,  # num of display items upper bound
        "lim_fea_per_cate_xu": lim_fea_per_cate_xu,  # num of feature items upper bound
        "constraint_num": 2,  # simplify to 2 high level constraints
    }
    print(f"{CFG=}")
    PRICE_LIST = pop_ga.init_price_list(CFG["sku_num"], CFG["h"])

    try:
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
        start = start_week
        end = start_week + period

        idx_frame = [(SKU, Time_ID) for Time_ID in range(start, end) for SKU in sku_list]
        idx_frame = pd.DataFrame(idx_frame, columns=["SKU", "Time_ID"])

        output_df = pd.concat([idx_frame, formatted_schedule], axis=1)
        
        st.dataframe(output_df)

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
    except:
        "Please input the values for the GA to run :)"