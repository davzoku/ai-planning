import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.sampling import Sampling

from pymoo.operators.crossover.pntx import (
    SinglePointCrossover,
    TwoPointCrossover,
)
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.mutation.bitflip import BFM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover

import pop_ga
import revenue_estimation

from pop_ga import SwapCrossover, SwapMutation, SKUPopulationSampling

GEN_SIZE = 100
POP_SIZE = 10
CFG = {
    "sku_num": 36,  # no of sku
    "h": 8,  # week horizon
    "price_opt_num": 4,  # num of pricing options dvar
    "ndf": 2,  # num of display, feature dvar
    "lim_pro_per_cate_xu": 15,  # num of promotion items upper bound
    "lim_dis_per_cate_xu": 10,  # num of display items upper bound
    "lim_fea_per_cate_xu": 5,  # num of feature items upper bound
    "constraint_num": 2,  # simplify to 2 high level constraints
}

start_week = 1375
period = 8
zscore = pd.read_csv("../assets/Z_scores.csv")
selected_sku_list = zscore["SKU"].tolist()

crossover_configs = [
    {"operator": SBX, "prob": 0.2, "eta": 15},
    {"operator": SBX, "prob": 0.4, "eta": 15},
    {"operator": SBX, "prob": 0.6, "eta": 15},
    {"operator": SBX, "prob": 0.8, "eta": 15},
    {"operator": SinglePointCrossover, "prob": 0.2},
    {"operator": SinglePointCrossover, "prob": 0.4},
    {"operator": SinglePointCrossover, "prob": 0.6},
    {"operator": SinglePointCrossover, "prob": 0.8},
    {"operator": TwoPointCrossover, "prob": 0.2},
    {"operator": TwoPointCrossover, "prob": 0.4},
    {"operator": TwoPointCrossover, "prob": 0.6},
    {"operator": TwoPointCrossover, "prob": 0.8},
    {"operator": SwapCrossover, "prob": 0.2, "cfg": CFG},
    {"operator": SwapCrossover, "prob": 0.4, "cfg": CFG},
    {"operator": SwapCrossover, "prob": 0.6, "cfg": CFG},
    {"operator": SwapCrossover, "prob": 0.8, "cfg": CFG},
]

mutation_configs = [
    {"operator": BFM, "prob": 0.2},
    {"operator": BFM, "prob": 0.4},
    {"operator": BFM, "prob": 0.6},
    {"operator": BFM, "prob": 0.8},
    {"operator": PolynomialMutation, "prob": 0.2, "eta": 20},
    {"operator": PolynomialMutation, "prob": 0.4, "eta": 20},
    {"operator": PolynomialMutation, "prob": 0.6, "eta": 20},
    {"operator": PolynomialMutation, "prob": 0.8, "eta": 20},
    {"operator": SwapMutation, "prob": 0.2, "cfg": CFG},
    {"operator": SwapMutation, "prob": 0.4, "cfg": CFG},
    {"operator": SwapMutation, "prob": 0.6, "cfg": CFG},
    {"operator": SwapMutation, "prob": 0.8, "cfg": CFG},
]

best_combinations = [
    ("SBX", 0.2, "BFM", 0.6),  # x
    ("SBX", 0.8, "PolynomialMutation", 0.8),  # all same
    ("SBX", 0.2, "SwapMutation", 0.4),  # x
    ("SinglePointCrossover", 0.8, "BFM", 0.8),  # x
    ("SinglePointCrossover", 0.2, "PolynomialMutation", 0.8),  # x
    ("SinglePointCrossover", 0.2, "SwapMutation", 0.8),  # x
    ("TwoPointCrossover", 0.6, "BFM", 0.6),  # x
    ("TwoPointCrossover", 0.4, "PolynomialMutation", 0.6),  # x
    ("TwoPointCrossover", 0.8, "SwapMutation", 0.8),  # x
    ("SwapCrossover", 0.6, "BFM", 0.2),  # x
    ("SwapCrossover", 0.8, "PolynomialMutation", 0.2),  # x
    ("SwapCrossover", 0.8, "SwapMutation", 0.4),  # x
]

if __name__ == "__main__":
    rev_est = revenue_estimation.RevenueEstimation()
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
        mutation=pop_ga.SwapMutation(cfg=problem.cfg, prob=0.4),
        crossover=SBX(prob=0.2, eta=15),
        eliminate_duplicates=True,
    )

    tested_combinations = set()

    results = []
    total_time = 0
    for crossover_config in crossover_configs:
        for mutation_config in mutation_configs:
            start_time = time.time()
            # Generate a unique identifier for the current pair
            pair_id = (
                crossover_config["operator"].__name__,
                crossover_config["prob"],
                mutation_config["operator"].__name__,
                mutation_config["prob"],
            )
            print(pair_id)
            if pair_id in best_combinations:

                # label = f"{crossover_config["operator"].__name__} {crossover_config["prob"]} {mutation_config["operator"].__name__} {mutation_config["prob"]}"

                label = "_".join(map(str, pair_id))
                # Skip this combination if it has already been tested
                # if pair_id in tested_combinations:
                #     continue

                # Prepare operator configurations
                crossover_kwargs = crossover_config.copy()
                mutation_kwargs = mutation_config.copy()

                # Extract operators
                crossover_operator = crossover_kwargs.pop("operator")
                mutation_operator = mutation_kwargs.pop("operator")

                # Configure and run the GA instance
                ga_instance = GA(
                    pop_size=POP_SIZE,
                    sampling=SKUPopulationSampling(cfg=CFG, pop_size=POP_SIZE),
                    crossover=crossover_operator(**crossover_kwargs),
                    mutation=mutation_operator(**mutation_kwargs),
                    eliminate_duplicates=True,
                )

                result = minimize(
                    problem,
                    ga_instance,
                    ("n_gen", GEN_SIZE),
                    seed=2,
                    save_history=True,
                    verbose=True,
                )

                elapsed_time = time.time() - start_time
                total_time += elapsed_time

                # Store results
                results.append(
                    {
                        "crossover": crossover_operator.__name__,
                        "crossover_prob": crossover_config["prob"],
                        "mutation": mutation_operator.__name__,
                        "mutation_prob": mutation_config["prob"],
                        "result": result,
                        "objective_value": result.F[0],
                        "elapsed_time": elapsed_time,
                    }
                )

                n_evals = np.array([e.evaluator.n_eval for e in result.history])
                opt = np.array([e.opt[0].F for e in result.history])
                plt.plot(n_evals, opt, "--", label=label)

                # # Mark this pair as tested
                # tested_combinations.add(pair_id)

    print(results)
    sorted_results = sorted(results, key=lambda x: x["objective_value"])
    with open("results.pkl", "wb") as f:
        pickle.dump(sorted_results, f)

    for result in sorted_results:
        print("Crossover Operator:", result["crossover"])
        print("Crossover Probability:", result["crossover_prob"])
        print("Mutation Operator:", result["mutation"])
        print("Mutation Probability:", result["mutation_prob"])
        print("Objective Value:", result["objective_value"])
        print("Time:", result["elapsed_time"])
        print()  # Add an empty line for better readability

    # After all permutations, print the total time taken
    print("Total time for all permutations (best):", total_time)
    plt.title("Analysis of Convergence")
    plt.xlabel("n_eval")
    plt.ylabel("f_min")
    plt.legend(loc="upper right")
    plt.show()
