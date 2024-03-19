
import time
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
import numpy as np
from pymoo.core.sampling import Sampling

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

def init_price_list(x: int, y: int) -> np.ndarray:
    """
    Initialize the price list based on the given dimensions.

    Parameters:
    - x (int): Number of SKUs.
    - y (int): Time horizon.

    Returns:
    - np.ndarray: An array of price values.
    """
    size = x * y
    values = np.arange(size * 10, 0, -10)
    return values

class SKUPopulationSampling(Sampling):
    """
    Custom sampling method for generating the initial population of all feasible solutions
    with uniqueness check and trial limit.
    """

    def __init__(self, cfg, pop_size, max_trials=10):
        super().__init__()
        self.cfg = cfg  # Configuration dictionary containing problem parameters
        self.pop_size = pop_size  # The size of the population to generate
        self.max_trials = max_trials  # Maximum trials to find a unique solution

    def _do(self, problem=None, n_samples=None, **kwargs):
        # Configuration parameters
        pop_size = self.pop_size
        nc = self.cfg["sku_num"]
        h = self.cfg["h"]
        price_ga = self.cfg["price_opt_num"]
        ndf = self.cfg["ndf"]
        lim_pro_per_cate_xu = self.cfg["lim_pro_per_cate_xu"]
        lim_dis_per_cate_xu = self.cfg["lim_dis_per_cate_xu"]
        lim_fea_per_cate_xu = self.cfg["lim_fea_per_cate_xu"]

        # Initialize the population set to ensure uniqueness
        unique_solutions = set()
        sku_pop = np.zeros((pop_size, nc * h * (price_ga + ndf)), dtype=int)

        s = 0
        while s < pop_size:
            trials = 0
            while trials < self.max_trials:
                x_sku_pop = np.zeros((nc * h, (price_ga + ndf)), dtype=int)
                start_row = 0
                for pop_t in range(h):
                    end_row = start_row + nc
                    sku_indices = np.random.permutation(nc)

                    num_discounted = np.random.randint(0, lim_pro_per_cate_xu + 1)
                    for sku_idx in sku_indices[:num_discounted]:
                        discount_idx = np.random.choice(price_ga)
                        x_sku_pop[start_row + sku_idx, discount_idx] = 1

                    discounted_skus = sku_indices[:num_discounted]
                    num_display = np.random.randint(
                        0, min(lim_dis_per_cate_xu, num_discounted) + 1
                    )
                    num_feature = np.random.randint(
                        0, min(lim_fea_per_cate_xu, num_discounted) + 1
                    )
                    sample_d = np.random.choice(
                        discounted_skus, num_display, replace=False
                    )
                    sample_f = np.random.choice(
                        discounted_skus, num_feature, replace=False
                    )

                    for sku_idx in sample_d:
                        x_sku_pop[start_row + sku_idx, price_ga] = 1
                    for sku_idx in sample_f:
                        x_sku_pop[start_row + sku_idx, price_ga + 1] = 1

                    start_row = end_row

                # Flatten and convert to tuple for hashability
                flat_solution = tuple(x_sku_pop.flatten())
                if flat_solution not in unique_solutions:
                    unique_solutions.add(flat_solution)
                    sku_pop[s, :] = np.array(flat_solution)
                    s += 1  # Increment if a unique solution is found
                    break  # Exit the trial loop
                else:
                    trials += 1  # Increment trial count and try again

        return sku_pop


class SwapMutation(Mutation):
    """
    Custom mutation operator for the GA where we swap rows in the candiate solution (2d array)
    """

    def __init__(self, cfg, prob=0.9):
        super().__init__()
        self.prob = prob
        self.cfg = cfg
        self.rows = self.cfg["sku_num"] * self.cfg["h"]

    def _do(self, problem, pop, **kwargs):
        # print(type(pop), pop.shape, pop)
        pop_reshaped = pop.reshape(
            (-1, self.rows, self.cfg["price_opt_num"] + self.cfg["ndf"])
        )

        # print(f"{len(pop)=}, {pop_reshaped.shape=}")
        # no of candidates to mutate
        num_to_mutate = int(len(pop) * self.prob)
        # print(f"{num_to_mutate=}")

        # random select indices of candidates to mutate
        indices_to_mutate = np.random.choice(len(pop), num_to_mutate, replace=False)

        for idx in indices_to_mutate:
            # Randomly choose two rows to swap
            row_idx1, row_idx2 = np.random.choice(self.rows, 2, replace=False)

            # Swap the rows
            pop_reshaped[idx, [row_idx1, row_idx2]] = pop_reshaped[
                idx, [row_idx2, row_idx1]
            ]

        mutated_pop = pop_reshaped.reshape((-1, pop.shape[1]))
        return mutated_pop


class SwapCrossover(Crossover):
    """
    Custom crossover operator for the GA where we swap rows across candiate solutions (2d array)
    """

    def __init__(self, cfg, prob=0.9, n_rows_to_swap=1, n_offsprings=2):
        super().__init__(2, 2)
        self.prob = prob
        self.cfg = cfg
        self.rows = self.cfg["sku_num"] * self.cfg["h"]
        self.n_rows_to_swap = n_rows_to_swap

    def _do(self, problem, X, **kwargs):
        p1, p2 = X

        # Reshape parents into 3D arrays
        p1_reshaped = p1.reshape(
            (-1, self.rows, self.cfg["price_opt_num"] + self.cfg["ndf"])
        )
        p2_reshaped = p2.reshape(
            (-1, self.rows, self.cfg["price_opt_num"] + self.cfg["ndf"])
        )

        # Randomly select solutions to perform crossover
        do_crossover = np.random.random() < self.prob

        # random crossover with the same row
        # if do_crossover:
        # # Randomly choose rows to swap
        # row_indices = np.random.choice(
        #     self.rows, self.n_rows_to_swap, replace=False
        # )

        # # Swap the rows between parents
        # for row_idx in row_indices:
        #     p1_reshaped[0, row_idx], p2_reshaped[0, row_idx] = (
        #         p2_reshaped[0, row_idx],
        #         p1_reshaped[0, row_idx],
        #     )

        if do_crossover:
            # Randomly choose rows to swap
            row_indices_p1 = np.random.choice(
                self.rows, self.n_rows_to_swap, replace=False
            )
            row_indices_p2 = np.random.choice(
                self.rows, self.n_rows_to_swap, replace=False
            )
            random_swaps = zip(row_indices_p1, row_indices_p2)
            # debug
            # print(row_indices_p1, row_indices_p2)
            # for pair in random_swaps:
            #     print(pair)
            # Swap the selected rows between parents
            for idx_p1, idx_p2 in random_swaps:
                p1_reshaped[0, idx_p1], p2_reshaped[0, idx_p2] = (
                    p2_reshaped[0, idx_p2].copy(),
                    p1_reshaped[0, idx_p1].copy(),
                )

        Q = np.copy(X)

        # Reshape back to 2D arrays
        Q[0] = p1_reshaped.reshape(-1, p1.shape[1])
        Q[1] = p2_reshaped.reshape(-1, p2.shape[1])

        return Q


class PromotionOptimizationProblem(ElementwiseProblem):
    def __init__(self):
        self.cfg = CFG
        self.sku_ndvar = (
            self.cfg["price_opt_num"] + self.cfg["ndf"]
        )  # no of dvar per sku, cols
        self.rows = (
            self.cfg["sku_num"] * self.cfg["h"]
        )  # no of rows in the 2d array of candidate soln
        self.n_var = self.rows * self.sku_ndvar
        self.sku_num = self.cfg["sku_num"]
        self.price_opt_num = self.cfg["price_opt_num"]

        super().__init__(
            n_var=self.n_var,
            n_obj=1,
            n_constr=self.cfg["sku_num"] * self.cfg["h"] * self.cfg["constraint_num"],
            xl=np.zeros(self.n_var),
            xu=np.ones(self.n_var),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        can_sol = x.reshape((self.rows, self.sku_ndvar))  # candidate solution

        ### CONSTRAINTS
        cv = self._calculate_constraints(can_sol)
        out["G"] = cv

        # If there are any violations, set the objective value to a large negative value
        if np.any(cv > 0):
            out["F"] = -np.inf
            return

        ### PROFIT CALCULATION
        profit = self._calculate_profit(can_sol)
        out["F"] = -profit

    def _calculate_constraints(self, can_sol):
        # cv: constraints per sku
        cv1 = np.zeros(can_sol.shape[0], dtype=int)
        for i, row in enumerate(can_sol):
            # constraint 1: at most 1 price flag
            if np.sum(row[: self.price_opt_num]) not in [0, 1]:
                # print("constaint 1")
                cv1[i] = 1
            # constraint 2: no display or feature if no promo
            if np.all(row[: self.price_opt_num] == 0) and (row[4] == 1 or row[5] == 1):
                # print("constaint 2")
                cv1[i] = 1

        # relax constraints work for default mutation, crossover
        # cv = cv1
        # cv2: constraints per week
        cv2 = np.zeros(can_sol.shape[0], dtype=int)
        promo_count = display_count = feature_count = 0
        # print(f"{can_sol=}, {self.sku_num=}")
        for i in range(0, len(can_sol), self.sku_num):
            # print(f"{i=}")
            # for j in range(0, len(can_sol), cfg['sku_num']):
            #     promo_count = np.sum(can_sol[i:i+cfg['sku_num'], :cfg['price_opt_num']])
            #     display_count = np.sum(can_sol[i:i+cfg['sku_num'], cfg['price_opt_num']:5])
            #     feature_count = np.sum(can_sol[i:i+cfg['sku_num'], 5:])
            # for j in range(0, len(can_sol), self.sku_num):

            promo_array = can_sol[i : i + self.sku_num, : self.price_opt_num]
            display_array = can_sol[
                i : i + self.sku_num, self.price_opt_num : self.price_opt_num + 1
            ]
            feature_array = can_sol[i : i + self.sku_num, self.price_opt_num + 1 :]
            # print(f"x {promo_array=}")
            # print(f"{promo_array.shape=}")
            # print(f"x {display_array=}")
            # print(f"{display_array.shape=}")
            # print(f"x {feature_array=}")
            # print(f"{feature_array.shape=}")
            promo_count = np.sum(promo_array)
            display_count = np.sum(display_array)
            feature_count = np.sum(feature_array)
            # promo_count += np.sum(row[:4])
            #     # promo_count += np.sum(row[:4])
            # display_count += np.sum(row[4:5])
            # feature_count += np.sum(row[5:])
            # print(promo_count, display_count, feature_count)
            # for row in can_sol[i:i+self.sku_num]:
            #     promo_count += np.sum(row[:self.price_opt_num])
            #     display_count += np.sum(row[self.price_opt_num:5])
            #     feature_count += np.sum(row[5:])
            # print(promo_count, display_count, feature_count)
            # promo count must be less than lim_pro_per_cate_xu
            if promo_count > self.cfg["lim_pro_per_cate_xu"]:
                # print(promo_count)
                for j in range(i, i + self.sku_num):
                    # print("constaint 3")
                    cv2[j] = 1
            # dis count must be less than lim_dis_per_cate_xu
            elif display_count > self.cfg["lim_dis_per_cate_xu"]:
                # print(display_count)
                for j in range(i, i + self.sku_num):
                    # print("constaint 4")
                    cv2[j] = 1
            # fea count must be less than lim_fea_per_cate_xu
            elif feature_count > self.cfg["lim_fea_per_cate_xu"]:
                # print(display_count)
                for j in range(i, i + self.sku_num):
                    # print("constaint 5")
                    cv2[j] = 1
        # Combine all constraint violations
        cv = np.concatenate((cv1, cv2))
        return cv

    def _calculate_profit(self, can_sol):
        profit = 0
        discounted_values = np.zeros_like(PRICE_LIST)

        # TODO: For loop to cater to demand function calculation,  also to include display and feature effect on profit
        for i in range(len(can_sol)):
            if np.all(can_sol[i, : self.price_opt_num] == 0):
                discounted_values[i] = PRICE_LIST[i]
            else:
                discount_factor = (
                    0.8
                    if np.array_equal(can_sol[i, :4], [0, 0, 0, 1])
                    else (
                        0.6
                        if np.array_equal(can_sol[i, :4], [0, 0, 1, 0])
                        else (
                            0.4
                            if np.array_equal(can_sol[i, :4], [0, 1, 0, 0])
                            else (
                                0.2
                                if np.array_equal(can_sol[i, :4], [1, 0, 0, 0])
                                else 1
                            )
                        )
                    )
                )
                discounted_values[i] = discount_factor * PRICE_LIST[i]

        # Calculate profit
        profit = np.sum(discounted_values)
        return profit


with st.form(key='all_inputs_form'):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pop_size = st.number_input("Pop size", value=0)
    with c2:
        sku_num = st.text_input('SKU number')
    with c3:
        week_horizon = st.number_input('Week horizon', value=0)
    with c4:
        price_opt_number = st.number_input('Pricing option number', value=0)
    
    st.write("---")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        ndf = st.number_input('Number of display features', value=0, key='ndf')
    with c6:
        lim_pro_per_cate_xu = st.number_input('Number of promotion items upper bound', value=0, key='lim_pro_per_cate_xu')
    with c7:
        lim_dis_per_cate_xu = st.number_input('Number of display items upper bound', value=0, key='lim_dis_per_cate_xu')
    with c8:
        lim_fea_per_cate_xu = st.number_input('Number of feature items upper bound', value=0, key='lim_fea_per_cate_xu')

    submit_button = st.form_submit_button(label='Submit')
    
if submit_button:
    POP_SIZE = pop_size
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
    
    #st.write(CFG)
    PRICE_LIST = init_price_list(CFG["sku_num"], CFG["h"])
    #st.write(PRICE_LIST)


if __name__ == "__main__":
    try:
        problem = PromotionOptimizationProblem()
        algo_custom_ops = GA(
            pop_size=100,
            sampling=SKUPopulationSampling(cfg=problem.cfg, pop_size=POP_SIZE),
            mutation=SwapMutation(cfg=problem.cfg, prob=0.9),
            crossover=SwapCrossover(cfg=problem.cfg, prob=0.9, n_rows_to_swap=10),
            eliminate_duplicates=True,
        )

        algo_vanilla_ops = GA(
            pop_size=100,
            sampling=SKUPopulationSampling(cfg=problem.cfg, pop_size=POP_SIZE),
            crossover=PointCrossover(prob=0.8, n_points=2),
            mutation=PolynomialMutation(prob=0.3, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        start_time_custom = time.time()
        res_custom = minimize(
            problem, algo_custom_ops, ("n_gen", 10), seed=2, save_history=True, verbose=True
        )
        end_time_custom = time.time()

        start_time_vanilla = time.time()
        res_vanilla = minimize(
            problem,
            algo_vanilla_ops,
            ("n_gen", 10),
            seed=2,
            save_history=True,
            verbose=True,
        )
        end_time_vanilla = time.time()

        time_custom = end_time_custom - start_time_custom
        time_vanilla = end_time_vanilla - start_time_vanilla

        st.success(f"Custom operators time: {time_custom} seconds")
        st.success(f"Vanilla operators time: {time_vanilla} seconds")
        #print(f"Custom operators time: {time_custom} seconds")
        #print(f"Vanilla operators time: {time_vanilla} seconds")
        st.success(f"Custom operators solution quality: {res_custom.F}")
        st.success(f"Vanilla operators solution quality: {res_vanilla.F}")
        #print(f"Custom operators solution quality: {res_custom.F}")
        #print(f"Vanilla operators solution quality: {res_vanilla.F}")

        # Plotting history (https://pymoo.org/misc/convergence.html)
        fig, ax = plt.subplots()
        n_evals = np.array([e.evaluator.n_eval for e in res_custom.history])
        opt = np.array([e.opt[0].F for e in res_custom.history])
        ax.plot(n_evals, opt, "--", label='res_custom')

        n_evals = np.array([e.evaluator.n_eval for e in res_vanilla.history])
        opt = np.array([e.opt[0].F for e in res_vanilla.history])
        ax.plot(n_evals, opt, "--", label='res_vanilla')

        ax.set_title("Convergence of res_custom & res_vanilla")
        ax.set_xlabel('n_eval')
        ax.set_ylabel('f_min')
        ax.legend(loc='upper right')
        #plt.show()
        st.pyplot(fig)
    except:
        'Please input the values for the GA to run :)'