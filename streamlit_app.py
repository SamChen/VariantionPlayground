import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from collections import defaultdict

from basic import *
eps = 1e-8

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    latext = r'''
    ## General process:

    For each group:

    1. Use ground truth $between\_subj\_mean$ and $between\_subj\_var$ to sample the mean value for each subject

    2. Use ground truth mean of within-subject variance and variance of within-subject variance to sample the $within\_subj\_var$ for each subject.

    3. For each subject, we sample $M$ values/measurements from a normal distribution with the subject's mean and $within\_subj\_var$.

    4. Measure the actual $within\_subj\_var$ and $between\_subj\_var$ from sampled values.

    5. Calculate the overall variance: $ Var = \frac{1}{N} (\frac{2*within\_subj\_var}{M} + between\_subj\_var) $ where $N$ refers to the total number of subjects and $M$ refers to the total number of measurements per subject.
        -    $\mu_n = \frac{1}{M}\sum_{m=1}^{M} measurement_{(n,m)}$
        -    $var_n = \frac{1}{M}\sum_{m=1}^{M} (measurement_{(n,m)}-\mu_n)^2$
        -    $within\_subj\_var=\frac{1}{N}\sum_{n=1}^{N} var_n$
        -    $between\_subj\_mean=\bar{\mu}=\frac{1}{N}\sum_{n=1}^{N} \mu_n$
        -    $between\_subj\_var=\frac{1}{N}\sum_{n=1}^{N} (\mu_n - \bar{\mu})^2$

    Calculate t-test between two groups given the $between\_subj\_mean$ and $Var$ obtained from sampled values.

    '''
    st.write(latext)

    st.write("## Playground")
    predefined_configurations = {
        "mimic pTau":{
            "default_gt_between_subj_mean1": 1.7,
            "default_gt_between_subj_var1": 0.52,
            "default_gt_within_subj_var_mean1": 0.1,
            "default_gt_within_subj_var_var1": 0.05,
            "default_gt_between_subj_mean2": 1.7 * 1.1,
            "default_gt_between_subj_var2": 0.52,
            "default_gt_within_subj_var_mean2": 0.1,
            "default_gt_within_subj_var_var2": 0.05,
        },

        "mimic pTau X10":{
            "default_gt_between_subj_mean1": 1.7 * 10,
            "default_gt_between_subj_var1": 0.52 * 10,
            "default_gt_within_subj_var_mean1": 0.1 * 10,
            "default_gt_within_subj_var_var1": 0.05 * 10,
            "default_gt_between_subj_mean2": 1.7 * 1.1 * 10,
            "default_gt_between_subj_var2": 0.52 * 10,
            "default_gt_within_subj_var_mean2": 0.1 * 10,
            "default_gt_within_subj_var_var2": 0.05 * 10,
        },

        "mimic GFAP":{
            "default_gt_between_subj_mean1": 70.2,
            "default_gt_between_subj_var1": 1126.6,
            "default_gt_within_subj_var_mean1": 255.5,
            "default_gt_within_subj_var_var1": 557058.0,
            "default_gt_between_subj_mean2": 70.2 * 1.1,
            "default_gt_between_subj_var2": 1126.6,
            "default_gt_within_subj_var_mean2": 255.5,
            "default_gt_within_subj_var_var2": 557058.0,
        },
        "mimic NfL":{
            "default_gt_between_subj_mean1":    24.6,
            "default_gt_between_subj_var1":     12.1,
            "default_gt_within_subj_var_mean1": 3.4,
            "default_gt_within_subj_var_var1":  20.5,
            "default_gt_between_subj_mean2":    24.6 * 1.1,
            "default_gt_between_subj_var2":     12.1,
            "default_gt_within_subj_var_mean2": 3.4,
            "default_gt_within_subj_var_var2":  20.5,
        },
        "mimic Flt1":{
            "default_gt_between_subj_mean1":    134.8,
            "default_gt_between_subj_var1":     22091.3,
            "default_gt_within_subj_var_mean1": 468.1,
            "default_gt_within_subj_var_var1":  1402354.4,
            "default_gt_between_subj_mean2":    134.8 * 1.1,
            "default_gt_between_subj_var2":     22091.3,
            "default_gt_within_subj_var_mean2": 468.1,
            "default_gt_within_subj_var_var2":  1402354.4,
        },
    }
    with st.sidebar:
        with st.form("Predefined Configurations:"):
            selected_configuration = st.selectbox("Select a predefined configuration: ", predefined_configurations.keys())
            apply_log = st.selectbox("Apply log transformation (log(value+1)): ", [True, False], index=1)
            total_trials = st.number_input("Total number of simulation trials: ", value=100)
            st.form_submit_button("Submit")
            # apply_log = False

    default_configuration = predefined_configurations[selected_configuration]
    with st.form("Configuration:"):
        col1,col2 = st.columns(2)
        col1.write("Group1:")
        n_subj1                  = col1.number_input("Total number of subjets for each group (group1): ", value=10)
        gt_between_subj_mean1    = col1.number_input("Ground truth between subject mean (group1): ",
                                                     value=default_configuration["default_gt_between_subj_mean1"],format="%.5f")
        gt_between_subj_var1     = col1.number_input("Ground truth between subject variance (group1): ",
                                                     value=default_configuration["default_gt_between_subj_var1"],format="%.5f")
        gt_within_subj_var_mean1 = col1.number_input("Ground truth mean over within subject variance (group1): ",
                                                     value=default_configuration["default_gt_within_subj_var_mean1"],format="%.5f")
        gt_within_subj_var_var1  = col1.number_input("Ground truth variance over within subject variance (group1): ",
                                                     value=default_configuration["default_gt_within_subj_var_var1"],format="%.5f")

        col2.write("Group2:")
        n_subj2                  = col2.number_input("Total number of subjets for each group (group2): ", value=10)
        gt_between_subj_mean2    = col2.number_input("Ground truth between subject mean (group2): ",
                                                     value=default_configuration["default_gt_between_subj_mean2"],format="%.5f")
        gt_between_subj_var2     = col2.number_input("Ground truth between subject variance (group2): ",
                                                     value=default_configuration["default_gt_between_subj_var2"],format="%.5f")
        gt_within_subj_var_mean2 = col2.number_input("Ground truth mean over within subject variance (group2): ",
                                                     value=default_configuration["default_gt_within_subj_var_mean2"],format="%.5f")
        gt_within_subj_var_var2  = col2.number_input("Ground truth variance over within subject variance (group2): ",
                                                     value=default_configuration["default_gt_within_subj_var_var2"],format="%.5f")
        st.form_submit_button("Submit")

    # gt_between_subj_mean_ratio    =  gt_between_subj_mean2   / gt_between_subj_mean1
    # gt_between_subj_var_ratio     =  gt_between_subj_var2    / gt_between_subj_var1
    # gt_within_subj_var_mean_ratio =  gt_within_subj_var_mean2/ gt_within_subj_var_mean1
    # gt_within_subj_var_var_ratio  =  gt_within_subj_var_var2 / gt_within_subj_var_var1

    outputs = defaultdict(list)
    for m in range(2,12):
        for seed in range(0, total_trials):
            seed = seed * 10
            df1, act_mean1, act_std1 = stats_synthesize(
                gt_between_subj_mean1, gt_between_subj_var1,
                gt_within_subj_var_mean1, gt_within_subj_var_var1,
                m = m,
                n_subj = n_subj1,
                groupid = 1,
                apply_log = apply_log,
                seed = seed,
            )

            df2, act_mean2, act_std2 = stats_synthesize(
                gt_between_subj_mean2, gt_between_subj_var2,
                gt_within_subj_var_mean2, gt_within_subj_var_var2,
                m = m,
                n_subj = n_subj2,
                groupid = 2,
                apply_log = apply_log,
                seed = seed,
            )

            # Calculate t-test between two groups given the $between\_subj\_mean$ and $Var$ obstained from sampled values.
            tstat, pvalue = stats.ttest_ind_from_stats(act_mean1, act_std1, n_subj1,
                                                       act_mean2, act_std2, n_subj2)
            outputs["pvalue"].append(pvalue)
            outputs["M"].append(m)

    df_stats = pd.DataFrame(outputs)
    # st.write(df_stats)
    # st.pyplot(sns.displot(df, x="pvalue", hue="m", palette="tab10"))
    # cols = st.columns(2)
    # cols[0].write(df1.groupby("subid")["value"].describe())
    # cols[0].write(df2.groupby("subid")["value"].describe())

    # g = sns.displot(df_stats, x="pvalue", col="m")
    # cols[1].pyplot(g)
    # .transform_density(
    #     'pvalue',
    #     groupby=['M'],
    #     as_=['pvalue', 'density'],
    #     extent=[0, 1],
    #     counts=False,
    # )
    bar_chart = alt.Chart(df_stats).transform_joinaggregate(
        total='count(*)',
        groupby=["M"]
    ).transform_calculate(
        pct='1 / datum.total'
    ).mark_bar().encode(
        alt.X("pvalue:Q").bin(extent=[0, 1], step=0.05),
        alt.Y("sum(pct):Q").title("Percentage"),
        alt.Color("M:N")
    ).properties(
        width=200,
        height=200
    ).facet(
        facet="M:N",
        columns=5
    ).resolve_scale(
        x='independent'
    )
    st.altair_chart(bar_chart, theme="streamlit")

    # st.pyplot(sns.displot(x="pvalue", hue="M", col="M", kind="kde", data=df_stats))

    # st.write(f'''
    # |                               |   Value ratio (group2 / group1)                         |
    # |:----------------------------------------------------------:|:--------------------------:|
    # |Ground truth between subject mean                 | {gt_between_subj_mean_ratio}    |
    # |Ground truth between subject variance             | {gt_between_subj_var_ratio}     |
    # |Ground truth mean over within subject variance    | {gt_within_subj_var_mean_ratio} |
    # |Ground truth variance over within subject variance| {gt_within_subj_var_var_ratio } |
    #              ''')
