import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from collections import defaultdict

def fn_variance_diff(within_subj_var, between_subj_var, sample_size):
    return (2*within_subj_var) / sample_size + between_subj_var


def generate_samples(
    between_subj_mean, between_subj_var,
    within_subj_var_mean, within_subj_var_std,
    m, n_subj, groupid, seed):

    np.random.seed(seed)
    # 1. Use `ground truth` $between\_subj\_mean$ and $between\_subj\_var$ to sample the mean value for each subject
    subj_means = np.random.normal(between_subj_mean,
                                  np.sqrt(between_subj_var),
                                  size=n_subj)
    subj_means = np.abs(subj_means)

    # 2. Use `ground truth` **mean over within subject variance** and **variance over within subject variance** to sample the $within\_subj\_var$ for each subject.
    subj_vars = np.random.normal(within_subj_var_mean,
                                 np.sqrt(within_subj_var_std),
                                 size=n_subj)
    subj_vars = np.abs(subj_vars)

    samples = defaultdict(list)
    for idx, (subj_mean, subj_var) in enumerate(zip(subj_means, subj_vars)):
        np.random.seed(seed+idx+1)

        # 3. For each subject, we sample $M$ values/measurements based on that subject's **mean** and $within\_subj\_var$.
        subj_sample = np.random.normal(subj_mean, np.sqrt(subj_var), m)

        samples["groupid"].extend([f"{groupid}" for i in subj_sample])
        samples["subid"].extend([f"{groupid}_{idx}" for i in subj_sample])
        samples["value"].extend(subj_sample)
    return pd.DataFrame(samples)


def stats_synthesize(
    between_subj_mean, between_subj_var,
    within_subj_var_mean, within_subj_var_std,
    m,
    n_subj,
    groupid,
    apply_log,
    seed,
):
    group = generate_samples(
        between_subj_mean, between_subj_var,
        within_subj_var_mean, within_subj_var_std,
        m        = m       ,
        n_subj   = n_subj  ,
        groupid  = groupid ,
        seed     = seed
    )

    # * optional
    if apply_log:
        group["value"] = group["value"].apply(lambda x: np.log(x+1))

    # 4. Measure the `actual` $within\_subj\_var$ and $between\_subj\_var$ from sampled values.
    act_within_subj_var=group.groupby("subid")["value"].var().mean()
    act_between_subj_var=group.groupby("subid")["value"].mean().var()
    # 5. Calculate the overall variance: $ Var = \frac{1}{N} \frac{\frac{2}{M}*within\_subj\_var}{between\_subj\_var} $ where $N$ refers to the total number of subjects and $M$ refers total number of measurements per subject.
    act_std = np.sqrt(fn_variance_diff(within_subj_var=act_within_subj_var,
                                       between_subj_var=act_between_subj_var,
                                       sample_size=m) / n_subj)
    # 6. Between subject mean: $between\_subj\_mean=\frac{1}{N}*\sum_{n=1}^{N} \mu$
    act_mean = group.groupby("subid")["value"].mean().mean()
    return group, act_mean, act_std

eps = 1e-8

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    # data = defaultdict(list)
    # m = st.slider("Sample size range 1 ($m$):",1, 10, value=10)
    # values = [0.1, 0.5, 1.0, 5.0, 10.0]
    # values = [0.1, 0.5, 1.0]
    # ratioes = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10]
    # for ratio, within_subj_var in product(ratioes, values):
    #     between_subj_var = within_subj_var / ratio
    #     for m in range(1,11):
    #         var_diff = fn_variance_diff(within_subj_var, between_subj_var, m)
    #         # var_diff_s2 = fn_variance_diff(within_subj_var, between_subj_var, s2)
    #         # ratio = var_diff_s1 / var_diff_s2
    #         data["Within_subj_var"].append(within_subj_var)
    #         data["Between_subj_var"].append(between_subj_var)
    #         data["Pair"].append(f"{within_subj_var}_{between_subj_var}")
    #         data["Variance"].append(var_diff)
    #         data["m"].append(m)
    #         data["Ratio"].append(within_subj_var / between_subj_var)

    # df = pd.DataFrame(data)
    # st.write(df)
    # latext = r'''
    #          $$
    #          Ratio = \frac{within\_subj\_var}{between\_subj\_var}
    #          $$
    #          '''
    # st.write(latext)

    # chart = alt.Chart(df).mark_line(point=True).encode(
    #     alt.X("m"),
    #     alt.Y("Variance"),
    #     alt.Color("Pair:N", legend=alt.Legend(columns=2, symbolLimit=0)),
    # ).properties(
    #     width=250,
    #     height=250
    # ).facet(
    #     facet="Ratio:N",
    #     columns=3
    # ).resolve_axis(
    #     x='independent',
    #     y='independent',
    # )
    # st.altair_chart(chart)

    latext = r'''
             ## General process:

             For each group:

             1. Use `ground truth` $between\_subj\_mean$ and $between\_subj\_var$ to sample the mean value for each subject

             2. Use `ground truth` **mean over within subject variance** and **variance over within subject variance** to sample the $within\_subj\_var$ for each subject.
             3. For each subject, we sample $M$ values/measurements based on that subject's **mean** and $within\_subj\_var$.

             4. Measure the `actual` $within\_subj\_var$ and $between\_subj\_var$ from sampled values.

             5. Calculate the overall variance: $ Var = \frac{1}{N} \frac{\frac{2}{M}*within\_subj\_var}{between\_subj\_var} $ where $N$ refers to the total number of subjects and $M$ refers total number of measurements per subject.
                 -    $var_n = \frac{1}{N}*\sum_{m=1}^{M} (measurement_{(n,m)} - \bar{\mu_n})^2$
                 -    $within\_subj\_var=\frac{1}{N}*\sum_{n=1}^{N} var_n$
                 -    $between\_subj\_var=\frac{1}{N}*\sum_{n=1}^{N} (\mu_n - \bar{\mu})^2 $

             6. Between subject mean: $between\_subj\_mean=\frac{1}{N}*\sum_{n=1}^{N} \mu$

             Calculate t-test between two groups given the $between\_subj\_mean$ and $Var$ obstained from sampled values.

             --------------------------------------------------
             We repeat the above process for 100 times for each $M$ where $M\in[1,10]$ to see the distribution of p-value.
             `We choose the repeat time to be 100 for fast rendering speed. It is a hacking choice.`

             '''
    st.write(latext)

    st.write("## Playground")
    predefined_configurations = {
        "mimic pTau":{
            "default_gt_between_subj_mean1": 1.7,
            "default_gt_between_subj_var1": 0.52,
            "default_gt_within_subj_var_mean1": 0.1,
            "default_gt_within_subj_var_var1": 0.05,
            "default_gt_between_subj_mean2": 2.0,
            "default_gt_between_subj_var2": 0.52,
            "default_gt_within_subj_var_mean2": 0.1,
            "default_gt_within_subj_var_var2": 0.05,
        },

        "mimic GFAP":{
            "default_gt_between_subj_mean1": 70.2,
            "default_gt_between_subj_var1": 1126.6,
            "default_gt_within_subj_var_mean1": 255.5,
            "default_gt_within_subj_var_var1": 557058.0,
            "default_gt_between_subj_mean2": 60.0,
            "default_gt_between_subj_var2": 1126.6,
            "default_gt_within_subj_var_mean2": 255.5,
            "default_gt_within_subj_var_var2": 557058.0,
        }
    }
    with st.sidebar:
        with st.form("Predefined Configurations:"):
            selected_configuration = st.selectbox("Select a predefined configuration: ", predefined_configurations.keys())
            n_subj = st.number_input("Total number of subjets for each group: ", value=10)
            # apply_log = st.selectbox("Apply log transformation (log(value+1)): ", [True, False], index=1)
            st.form_submit_button("Submit")
            apply_log = False

    default_configuration = predefined_configurations[selected_configuration]
    with st.form("Configuration:"):
        col1,col2 = st.columns(2)
        col1.write("Group1:")
        gt_between_subj_mean1    = col1.number_input("Ground truth between subject mean (group1): ",
                                                     value=default_configuration["default_gt_between_subj_mean1"],format="%.2f")
        gt_between_subj_var1     = col1.number_input("Ground truth between subject variance (group1): ",
                                                     value=default_configuration["default_gt_between_subj_var1"],format="%.2f")
        gt_within_subj_var_mean1 = col1.number_input("Ground truth mean over within subject variance (group1): ",
                                                     value=default_configuration["default_gt_within_subj_var_mean1"],format="%.2f")
        gt_within_subj_var_var1  = col1.number_input("Ground truth variance over within subject variance (group1): ",
                                                     value=default_configuration["default_gt_within_subj_var_var1"],format="%.2f")

        col2.write("Group2:")
        gt_between_subj_mean2    = col2.number_input("Ground truth between subject mean (group2): ",
                                                     value=default_configuration["default_gt_between_subj_mean2"],format="%.2f")
        gt_between_subj_var2     = col2.number_input("Ground truth between subject variance (group2): ",
                                                     value=default_configuration["default_gt_between_subj_var2"],format="%.2f")
        gt_within_subj_var_mean2 = col2.number_input("Ground truth mean over within subject variance (group2): ",
                                                     value=default_configuration["default_gt_within_subj_var_mean2"],format="%.2f")
        gt_within_subj_var_var2  = col2.number_input("Ground truth variance over within subject variance (group2): ",
                                                     value=default_configuration["default_gt_within_subj_var_var2"],format="%.2f")
        st.form_submit_button("Submit")

    outputs = defaultdict(list)
    for m in range(2,12):
        for seed in range(0, 100):
            seed = seed * 10
            df1, act_mean1, act_std1 = stats_synthesize(
                gt_between_subj_mean1, gt_between_subj_var1,
                gt_within_subj_var_mean1, gt_within_subj_var_var1,
                m = m,
                n_subj = n_subj,
                groupid = 1,
                apply_log = apply_log,
                seed = seed,
            )

            df2, act_mean2, act_std2 = stats_synthesize(
                gt_between_subj_mean2, gt_between_subj_var2,
                gt_within_subj_var_mean2, gt_within_subj_var_var2,
                m = m,
                n_subj = n_subj,
                groupid = 2,
                apply_log = apply_log,
                seed = seed,
            )

            # Calculate t-test between two groups given the $between\_subj\_mean$ and $Var$ obstained from sampled values.
            tstat, pvalue = stats.ttest_ind_from_stats(act_mean1, act_std1, n_subj,
                                                       act_mean2, act_std2, n_subj)
            outputs["pvalue"].append(pvalue)
            outputs["M"].append(m)

    df_stats = pd.DataFrame(outputs)
    # st.pyplot(sns.displot(df, x="pvalue", hue="m", palette="tab10"))
    # cols = st.columns(2)
    # cols[0].write(df1.groupby("subid")["value"].describe())
    # cols[0].write(df2.groupby("subid")["value"].describe())

    # g = sns.displot(df_stats, x="pvalue", col="m")
    # cols[1].pyplot(g)
    bar_chart = alt.Chart(df_stats).mark_bar().encode(
        alt.X("pvalue:Q").bin(extent=[0, 1], step=0.05),
        alt.Y("count()", scale=alt.Scale(domain=[0, 100])),
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
    st.altair_chart(bar_chart)

