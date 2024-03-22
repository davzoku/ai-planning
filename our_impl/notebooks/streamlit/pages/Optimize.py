import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import (
    SinglePointCrossover,
    TwoPointCrossover,
)
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

from utils import utils


import streamlit as st

import sys
import os

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

if submit_button:
    # TODO: convert to frontend variables
    start_week = 1375
    period = week_horizon
    zscore = pd.read_csv("assets/Z_scores.csv")
    selected_sku_list = zscore["SKU"].tolist()

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

    try:
        print(f"init problem")
        # TODO: convert to use computed demand model DFs
        rev_est = revenue_estimation.RevenueEstimation(
            sales_dir="assets/processed_sales.csv",
            cal_dir="assets/calendar_week.csv",
            events_dir="assets/events.csv",
            zscore_dir="assets/Z_scores.csv",
            dd_coeff_dir="assets/Coefficients.csv",
            prices_dir="assets/prices.csv",
        )
        problem = pop_ga.PromotionOptimizationProblem(
            cfg=CFG,
            rev_est=rev_est,
            selected_sku_list=selected_sku_list,
            start_week=start_week,
            period=period,
        )
        algo_best_ops = GA(
            pop_size=POP_SIZE,
            sampling=pop_ga.SKUPopulationSampling(cfg=problem.cfg, pop_size=POP_SIZE),
            # mutation=pop_ga.SwapMutation(cfg=problem.cfg, prob=0.4),
            # crossover=SBX(prob=0.2, eta=15),
            mutation=PolynomialMutation(prob=0.6, eta=20),
            crossover=TwoPointCrossover(prob=0.4),
            eliminate_duplicates=True,
        )
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
    except:
        "Please input the values for the GA to run :)"
