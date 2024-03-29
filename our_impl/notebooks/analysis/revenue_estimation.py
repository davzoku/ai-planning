import pandas as pd
import numpy as np

import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=(SettingWithCopyWarning))

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated",
)

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in log"
)


class RevenueEstimation:
    def __init__(
        self,
        ### COMMENT OUT LATER FOR FRONT-END ###
        sales_dir="../assets/processed_sales.csv",  # process_data_sls.ipynb
        cal_dir="../assets/calendar_week.csv",  # process_data_sls.ipynb
        events_dir="../assets/events.csv",  # process_data_sls.ipynb
        zscore_dir="../assets/Z_scores.csv",  # demand func .ipynb
        dd_coeff_dir="../assets/Coefficients.csv",  # demand func .ipynb
        prices_dir="../assets/prices.csv",  # process_data_sls.ipynb
        ### COMMENT OUT LATER FOR FRONT-END ###
        sales=None,
        cal_week=None,
        events=None,
        zscore_tmp=None,  ### UPDATED
        zscore=None,
        dd_coeff=None,
        dd_bias=None,  ### UPDATED
        prices=None,
    ):
        
        ### COMMENT OUT LATER FOR FRONT-END ###
        self.sales = pd.read_csv(sales_dir) if sales is None else sales
        self.cal_week = pd.read_csv(cal_dir) if cal_week is None else cal_week
        self.events = pd.read_csv(events_dir) if events is None else events
        self.zscore_tmp = pd.read_csv(zscore_dir) if zscore_tmp is None else zscore_tmp
        ### COMMENT OUT LATER FOR FRONT-END ###
        
        ## UNCCOMMENT OUT LATER FOR FRONT-END ###
        # self.sales = sales
        # self.cal_week = cal_week
        # self.events = events
        # self.zscore_tmp = zscore_tmp
        ## UNCCOMMENT OUT LATER FOR FRONT-END ###

        # self.zscore_tmp = self.zscore_tmp.rename(columns={"Unnamed: 0": "SKU"})
        self.zscore_tmp = self.zscore_tmp.sort_values(by="SKU")
        self.zscore = (
            self.zscore_tmp[["SKU", "Mean", "Std_deviation"]]
            if zscore is None
            else zscore
        )
        self.dd_bias = self.zscore_tmp[["SKU", "bias"]] if dd_bias is None else dd_bias

        self.dd_coeff = (
            pd.read_csv(dd_coeff_dir).drop(columns="Unnamed: 0")
            if dd_coeff is None
            else dd_coeff
        )
        self.prices = pd.read_csv(prices_dir) if prices is None else prices
        self.all_skus = self.sales["SKU"].unique()
        self.time_year = self.sales[["Time_ID", "Year"]].copy().drop_duplicates()

    def fitness_demand(
        self, ga_output: pd.DataFrame, sku_list: list, start: float, period: int
    ) -> float:

        histr = start - 1
        end = start + period

        sku_list = sorted(sku_list)

        idx_frame = [
            (SKU, Time_ID) for Time_ID in range(start, end) for SKU in sku_list
        ]
        idx_frame = pd.DataFrame(idx_frame, columns=["SKU", "Time_ID"])

        sales_hist = self.sales[
            (self.sales["SKU"].isin(sku_list))
            & (self.sales["Time_ID"] >= histr - period - 5)
            & (self.sales["Time_ID"] <= histr)
        ].copy()

        ## Preapare GA dataframe
        ga_df = ga_output.copy()
        ga_df["Discount"] = 1 + ga_df["Discount"]
        ga_df = pd.concat([idx_frame, ga_df], axis=1)
        ga_df = pd.merge(ga_df, self.zscore, on=["SKU"], how="left")
        ga_df["z_disc"] = (ga_df["Discount"] - ga_df["Mean"]) / ga_df["Std_deviation"]
        # ga_df["z_disc"] = ga_df["z_disc"] * -1
        ga_df["z_disc"] = ga_df["z_disc"]
        ga_df = ga_df[["SKU", "Time_ID", "z_disc", "Feature", "Display"]]
        ga_df = ga_df.rename(columns={"z_disc": "Discount"})

        ## Create Competitor Matrix
        comp_matrix_columns = [
            f"{sku}_{promo}"
            for sku in self.all_skus
            for promo in ["Discount", "Display", "Feature", "Sales"]
        ]
        comp_matrix = pd.DataFrame(
            columns=comp_matrix_columns, index=range(len(sku_list) * period)
        )
        comp_matrix = pd.concat([idx_frame, comp_matrix], axis=1)
        for sku in sku_list:
            for promo in ["Discount", "Display", "Feature"]:
                ### UPDATED REMOVE NEG
                tmp = list(ga_df[ga_df["SKU"] == sku][promo]) * period
                tmp = pd.DataFrame(tmp)
                comp_matrix[sku + "_" + promo] = tmp
                comp_matrix.loc[comp_matrix["SKU"] == sku, [sku + "_" + promo]] = 0
        comp_matrix.fillna(0, inplace=True)

        zscore_tmp_ = self.zscore[self.zscore['SKU'].isin(sku_list)]

        revenue = []

        ## Iterate through each week through the demand function
        ## Obtain sales prediction and feed back into historical sales for picking

        for week in range(start, end):

            dd_coeff_val = self.dd_coeff[sku_list].values
            year = self.time_year[self.time_year["Time_ID"] == week]["Year"].values[0]
            ga_tmp = ga_df[ga_df["Time_ID"] == week].copy()
            for promo in ["Discount", "Feature", "Display"]:
                merge = sales_hist[sales_hist["Time_ID"] == week - 1][
                    ["SKU", promo]
                ].copy()
                merge = merge.rename(columns={promo: promo + "lag"})
                ga_tmp = pd.merge(ga_tmp, merge, on=["SKU"], how="left")

            ### CHANGE Log_sls to Sales
            merge = sales_hist[sales_hist["Time_ID"] == week - 1][
                ["SKU", "Sales", "Lag8w_avg_sls"]
            ].copy()
            merge = merge.rename(
                columns={"Sales": "Saleslag", "Lag8w_avg_sls": "Sales_mov_avg"}
            )
            ga_tmp = pd.merge(ga_tmp, merge, on=["SKU"], how="left")

            events_tmp = (
                self.events[self.events["Time_ID"] == week]
                .drop(columns="Time_ID")
                .copy()
            )
            events_tmp = pd.concat([events_tmp] * len(sku_list), ignore_index=True)
            ga_tmp = pd.concat([ga_tmp, events_tmp], axis=1)

            comp_tmp = (
                comp_matrix[comp_matrix["Time_ID"] == week]
                .drop(columns="Time_ID")
                .copy()
            )
            for sku in sku_list:
                tmp = sales_hist[
                    (sales_hist["SKU"] == sku) & (sales_hist["Time_ID"] == week - 1)
                ]["Sales"].item()
                comp_tmp[sku + "_Sales"] = tmp
                comp_tmp.loc[comp_tmp["SKU"] == sku, [sku + "_Sales"]] = 0

            ga_tmp = pd.merge(ga_tmp, comp_tmp, on=["SKU"], how="left")

            ga_val = ga_tmp.drop(columns=["SKU", "Time_ID"]).values

            sales_output = np.diag(ga_val.dot(dd_coeff_val))

            ### ADD MODEL BIAS
            bias_tmp = self.dd_bias[self.dd_bias["SKU"].isin(sku_list)]["bias"].values
            sales_output = sales_output + bias_tmp
            sales_output[sales_output < 0] = 0

            prices_tmp = self.prices[
                (self.prices["SKU"].isin(sku_list)) & (self.prices["Year"] == year)
            ]["med_price"].values

            discounts_tmp = 2 - (ga_tmp['Discount'].values * zscore_tmp_['Std_deviation'].values + zscore_tmp_['Mean'].values)
            other_costs_tmp = ga_tmp['Feature'].values * 5 + ga_tmp['Display'].values * 10

            revenue.append(sum(sales_output * prices_tmp * discounts_tmp - other_costs_tmp))

            ## Prep for historical insert
            prep_tmp = sales_hist[sales_hist["Time_ID"] == week - 1][
                ["SKU", "Lag7w_sum_sls"]
            ].copy()
            hist_prep = sales_hist[sales_hist["Time_ID"] == week - 7][["SKU", "Sales"]]
            hist_prep = hist_prep.rename(columns={"Sales": "Lag7w_sls"})
            hist_prep = pd.merge(hist_prep, prep_tmp, on=["SKU"], how="left")
            hist_prep = hist_prep.drop(columns=["SKU"])

            ## Build historical insert
            hist_insert = ga_tmp[["SKU", "Time_ID", "Discount", "Display", "Feature"]]
            hist_insert["Year"] = year
            hist_insert["Sales"] = sales_output
            # hist_insert["Log_sls"] = hist_insert["Sales"] ## NOT USED ANYMORE
            hist_insert = pd.concat([hist_insert, hist_prep], axis=1)
            hist_insert["Lag8w_avg_sls"] = (
                hist_insert["Lag7w_sum_sls"] + hist_insert["Sales"]
            ) / 8
            hist_insert["Lag7w_sum_sls_upd"] = (
                hist_insert["Lag7w_sum_sls"]
                - hist_insert["Lag7w_sls"]
                + hist_insert["Sales"]
            )
            hist_insert = hist_insert[
                [
                    "SKU",
                    "Time_ID",
                    "Year",
                    "Sales",
                    "Discount",
                    "Display",
                    "Feature",
                    # "Log_sls",  ## NOT USED ANYMORE
                    "Lag8w_avg_sls",
                    "Lag7w_sum_sls_upd",
                ]
            ]
            hist_insert = hist_insert.rename(
                columns={"Lag7w_sum_sls_upd": "Lag7w_sum_sls"}
            )
            hist_insert.fillna(0, inplace=True)

            ## Insert results into historical
            sales_hist = pd.concat([sales_hist, hist_insert], ignore_index=True)

        return sum(revenue)


if __name__ == "__main__":
    rev_est = RevenueEstimation()

    # full 36
    sku_list = rev_est.zscore["SKU"].tolist()
    sku_list = sorted(sku_list)
    print(f"{sku_list=}")
    start = 1375
    period = 8

    ga_output = {
        "Discount": np.random.choice(
            np.arange(5, 55, 5) / 100, len(sku_list) * period
        ),  # Random discounts
        "Feature": np.random.choice([0, 1], len(sku_list) * period),  # Random features
        "Display": np.random.choice([0, 1], len(sku_list) * period),  # Random displays
    }
    ga_output = pd.DataFrame(ga_output)

    revenue = rev_est.fitness_demand(ga_output, sku_list, start, period)
    print(revenue)
