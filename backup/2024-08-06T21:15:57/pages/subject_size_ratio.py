import streamlit as st
import altair as alt
from altair import datum

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from collections import defaultdict

import sys
# import path
# directory = path.path(__file__).abspath()
# setting path
sys.path.append("..")
from basic import *
eps = 1e-8

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    with st.sidebar:
        with st.form("Predefined Configurations:"):
            # selected_configuration = st.selectbox("Select a predefined configuration: ", predefined_configurations.keys())
            apply_log = st.selectbox("Apply log transformation (log(value+1)): ", [True, False], index=1)
            total_trials = st.number_input("Total number of simulation trials: ", value=100)
            total_subjs = st.number_input("Total number of subjects: ", value=30)
            st.form_submit_button("Submit")
            # apply_log = False

    with st.form("Configuration:"):
        col1,col2 = st.columns(2)
        col1.write("Group1:")
        gt_between_subj_mean1    = col1.number_input("Ground truth between subject mean (group1): ",
                                                     value=2.5,format="%.5f")
        gt_between_subj_var1     = col1.number_input("Ground truth between subject variance (group1): ",
                                                     value=2.0,format="%.5f")
        gt_within_subj_var_mean1 = col1.number_input("Ground truth mean over within subject variance (group1): ",
                                                     value=0.0,format="%.5f")
        gt_within_subj_var_var1  = col1.number_input("Ground truth variance over within subject variance (group1): ",
                                                     value=1.0,format="%.5f")

        col2.write("Group2:")
        gt_between_subj_mean2    = col2.number_input("Ground truth between subject mean (group2): ",
                                                     value=3.1,format="%.5f")
        gt_between_subj_var2     = col2.number_input("Ground truth between subject variance (group2): ",
                                                     value=2.9,format="%.5f")
        gt_within_subj_var_mean2 = col2.number_input("Ground truth mean over within subject variance (group2): ",
                                                     value=0.0,format="%.5f")
        gt_within_subj_var_var2  = col2.number_input("Ground truth variance over within subject variance (group2): ",
                                                     value=1.0,format="%.5f")
        st.form_submit_button("Submit")

    outputs = defaultdict(list)
    ratio_dict = {"1:1": 0.5, "2:1": 2/3, "5:1": 5/6}
    for m in range(2,12):
        for seed in range(0, total_trials):
            for name_ratio, ratio in ratio_dict.items():
                scale = 1
                shift = 0
                n_subj1 = int(total_subjs * ratio)
                n_subj2 = total_subjs - n_subj1
                seed = seed * 10
                df1, act_mean1, act_std1 = stats_synthesize(
                    gt_between_subj_mean1,
                    gt_between_subj_var1,
                    gt_within_subj_var_mean1,
                    gt_within_subj_var_var1,
                    m = m,
                    n_subj = n_subj1,
                    groupid = 1,
                    apply_log = apply_log,
                    seed = seed,
                    shift=shift,
                    scale=scale
                )

                df2, act_mean2, act_std2 = stats_synthesize(
                    gt_between_subj_mean2,
                    gt_between_subj_var2,
                    gt_within_subj_var_mean2,
                    gt_within_subj_var_var2,
                    m = m,
                    n_subj = n_subj2,
                    groupid = 2,
                    apply_log = apply_log,
                    seed = seed,
                    shift=shift,
                    scale=scale
                )

                # Calculate t-test between two groups given the $between\_subj\_mean$ and $Var$ obstained from sampled values.
                tstat, pvalue = stats.ttest_ind_from_stats(act_mean1, act_std1, n_subj1,
                                                           act_mean2, act_std2, n_subj2)
                outputs["pvalue"].append(pvalue)
                outputs["M"].append(m)
                outputs["ratio"].append(name_ratio)

    df_stats = pd.DataFrame(outputs)
    # target = "Shift"
    target = "ratio"
    base_chart = alt.Chart(df_stats).transform_joinaggregate(
        total='count(*)',
        groupby=[target, "M"]
    ).transform_calculate(
        pct='1 / datum.total'
    )
    bar_chart = base_chart.mark_bar(opacity=0.7).encode(
        alt.X("pvalue:Q").bin(extent=[0, 1], step=0.05),
        alt.Y("sum(pct):Q").title("Percentage").stack(None),
        alt.Color(f"{target}:N").title(None),
        alt.Row(f"{target}:N").title("Subject size ratio (group1:group2)").header(labelAngle=0),
        alt.Column(f"M:N").title("M").header(labelAngle=0)

    ).properties(
        width=150,
        height=150
    ).resolve_scale(
        x='independent'
    ).configure_range(
    category={"scheme": "category10"}
    )

    # .transform_filter(
    #     (datum.ratio == "1:1")
    # )

    st.altair_chart(bar_chart, theme="streamlit")
