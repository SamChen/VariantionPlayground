import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from collections import defaultdict

# import sys
# sys.path.append("../")
# from basic import generate_samples, fn_variance_diff
import statsmodels.formula.api as smf
eps = 1e-8

# from rpy2.robjects import r, globalenv, conversion, default_converter
# from rpy2.robjects.packages import importr, data
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.conversion import localconverter
# import rpy2.robjects.packages as rpackages

# @st.cache_data
def lmer_r(data, formula):
    output = {}
    # sleepstudy = pd.read_csv("sleepstudy.csv")
    # st.write(sleepstudy)
    with conversion.localconverter(default_converter + pandas2ri.converter):
        # stats     = importr("stats")
        # lme4_r    = importr('lme4')
        base      = importr('base')
        lme4_test = importr('lmerTest')

        r_out = lme4_test.lmer(formula, dat=data)
        # r_out = lme4_test.lmer("Reaction~Days + (Days|Subject)", dat=sleepstudy)
        summary = base.summary(r_out)
        # st.write(lme4_test.get_coefmat(r_out))
    return summary["coefficients"][-1,-1]

# @st.cache_data
def lmer_py(data, formula):
    md = smf.mixedlm(formula,
                     data=data,
                     groups=data["Subid"],
                     re_formula="1",
                     # vc_formula={"ithMeasurement": "0+ithMeasurement"}
                     )
    mdf = md.fit(method=["powell", "lbfgs"])
    assert mdf.converged == True
    # mdf = md.fit()
    return mdf

def generate_samples(
    between_subj_mean, between_subj_var,
    within_subj_var_left, within_subj_var_right,
    m, n_subj, groupid, seed):

    np.random.seed(seed)
    # 1. Use `ground truth` $between\_subj\_mean$ and $between\_subj\_var$ to sample the mean value for each subject
    subj_means = np.random.normal(between_subj_mean,
                                  np.sqrt(between_subj_var),
                                  size=n_subj)
    subj_means = np.abs(subj_means)

    np.random.seed(seed+1)
    # 2. Use `ground truth` **mean over within subject variance** and **variance over within subject variance** to sample the $within\_subj\_var$ for each subject.
    subj_vars = np.random.uniform(low=within_subj_var_left,
                                  high=within_subj_var_right,
                                  size=n_subj)
    subj_vars = np.abs(subj_vars)

    samples = defaultdict(list)
    for idx, (subj_mean, subj_var) in enumerate(zip(subj_means, subj_vars)):
        np.random.seed(seed+2+idx)

        # 3. For each subject, we sample $M$ values/measurements based on that subject's **mean** and $within\_subj\_var$.
        subj_sample = np.random.normal(subj_mean, np.sqrt(subj_var), m)

        # samples["groupid"].extend([f"{groupid}" for i in subj_sample])
        # samples["groupid"].extend([groupid for i in subj_sample])
        samples["Phaseid"].extend([groupid for i in subj_sample])
        # samples["subid"].extend([f"{groupid}_{idx}" for i in subj_sample])
        # samples["subid"].extend([f"{idx}" for i in subj_sample])
        samples["Subid"].extend([idx for i in subj_sample])
        samples["value"].extend(subj_sample)
        samples["ithMeasurement"].extend([i+groupid*m for i in range(1,m+1)])

    return pd.DataFrame(samples)

def stats_synthesize(
    between_subj_mean, between_subj_var,
    within_subj_var_left, within_subj_var_right,
    m,
    n_subj,
    groupid,
    apply_log,
    seed,
    shift=0,
    scale=1
):
    group = generate_samples(
        between_subj_mean, between_subj_var,
        within_subj_var_left, within_subj_var_right,
        m        = m       ,
        n_subj   = n_subj  ,
        groupid  = groupid ,
        seed     = seed
    )

    group["value"] = group["value"].apply(lambda x: x*scale+shift)
    # * optional
    if apply_log:
        group["value"] = group["value"].apply(lambda x: np.log(x+1))

    # # 4. Measure the `actual` $within\_subj\_var$ and $between\_subj\_var$ from sampled values.
    # act_within_subj_mean=group.groupby("subid")["value"].mean()
    # act_within_subj_var=group.groupby("subid")["value"].var()
    # # act_between_subj_var=group.groupby("subid")["value"].mean().var()

    return group
    # return group, act_mean, act_std



if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.write("## Playground")
    predefined_configurations = {
        "mimic pTau":{
            "default_gt_between_subj_mean1": 0.41,
            "default_gt_between_subj_var1": 0.053,
            "default_gt_within_subj_var_left1":  0.053 * 0.1,
            "default_gt_within_subj_var_right1": 0.053 * 0.5,

            "default_gt_between_subj_mean2": 0.34,
            "default_gt_between_subj_var2": 0.014,
            "default_gt_within_subj_var_left2":  0.014 * 0.1,
            "default_gt_within_subj_var_right2": 0.014 * 0.5,
        },
    }
    with st.sidebar:
        with st.form("Predefined Configurations:"):
            selected_configuration = st.selectbox("Select a predefined configuration: ", predefined_configurations.keys())
            apply_log = st.selectbox("Apply log transformation (log(value+1)): ", [True, False], index=1)
            total_trials = st.number_input("Total number of simulation trials: ", value=100)
            sample_size = st.number_input("Sample size: ", value=18)
            st.form_submit_button("Submit")
            # apply_log = False

    default_configuration = predefined_configurations[selected_configuration]
    with st.form("Configuration:"):
        lmer_formula = st.text_input("Formula for linear mixed model (fixed effects): ", value="value ~  Phaseid")
        target_var  = st.text_input("Variable that we are interested in", value="Phaseid")
        col1,col2 = st.columns(2)
        col1.write("Phase 1:")
        # n_subj1                  = col1.number_input("Total number of subjets for each Phase (Phase1): ", value=10)
        gt_between_subj_mean1    = col1.number_input("Ground truth between subject mean (Phase1): ",
                                                     value=default_configuration["default_gt_between_subj_mean1"],format="%.5f")
        gt_between_subj_var1     = col1.number_input("Ground truth between subject variance (Phase1): ",
                                                     value=default_configuration["default_gt_between_subj_var1"],format="%.5f")
        gt_within_subj_var_left1 = col1.number_input("Ground truth lower boundary within subject variance (Phase1): ",
                                                     value=default_configuration["default_gt_within_subj_var_left1"],format="%.5f")
        gt_within_subj_var_right1 = col1.number_input("Ground truth upper boundary within subject variance (Phase1): ",
                                                     value=default_configuration["default_gt_within_subj_var_right1"],format="%.5f")

        col2.write("Phase 2:")
        # n_subj2                  = col2.number_input("Total number of subjets for each Phase (Phase2): ", value=10)
        gt_between_subj_mean2    = col2.number_input("Ground truth between subject mean (Phase2): ",
                                                     value=default_configuration["default_gt_between_subj_mean2"],format="%.5f")
        gt_between_subj_var2     = col2.number_input("Ground truth between subject variance (Phase2): ",
                                                     value=default_configuration["default_gt_between_subj_var2"],format="%.5f")
        gt_within_subj_var_left2 = col2.number_input("Ground truth lower boundary within subject variance (Phase2): ",
                                                     value=default_configuration["default_gt_within_subj_var_left2"],format="%.5f")
        gt_within_subj_var_right2 = col2.number_input("Ground truth upper boundary within subject variance (Phase2): ",
                                                     value=default_configuration["default_gt_within_subj_var_right2"],format="%.5f")
        st.form_submit_button("Submit")


    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    outputs = defaultdict(list)
    m0 = 1
    mE = 11
    for m in range(m0, mE):
        for seed in range(0, total_trials):
            percent_complete = ((m-m0) * total_trials + seed+1) / (10 * total_trials)
            my_bar.progress(percent_complete, text=progress_text)

            seed = seed * 100
            group1 = stats_synthesize(
                gt_between_subj_mean1, gt_between_subj_var1,
                gt_within_subj_var_left1, gt_within_subj_var_right1,
                m = m,
                n_subj = sample_size,
                groupid = 0,
                apply_log = apply_log,
                seed = seed,
            )

            group2 = stats_synthesize(
                gt_between_subj_mean2, gt_between_subj_var2,
                gt_within_subj_var_left2, gt_within_subj_var_right2,
                m = m,
                n_subj = sample_size,
                groupid = 1,
                apply_log = apply_log,
                seed = seed+50,
                # seed = seed,
            )

            df = pd.concat([group1, group2]).reset_index(drop=True)
            # pvalue = lmer_r(df, lmer_formula)
            # outputs["pvalue"].append(pvalue)
            # st.write("R: ", pvalue)
            out = lmer_py(df, lmer_formula)
            # st.write(out.summary())
            # break
            outputs["pvalue"].append(out.pvalues.loc[target_var])
            outputs["M"].append(m)

    df_stats = pd.DataFrame(outputs)
    # df_stats.to_csv(f"statistic_estimation_{sample_size}.csv")

    bar_chart = alt.Chart(df_stats).transform_joinaggregate(
        total='count(*)',
        groupby=["M"]
    ).transform_calculate(
        pct='1 / datum.total'
    ).mark_bar().encode(
        alt.X("pvalue:Q").bin(extent=[0, 1], step=0.05),
        alt.Y("sum(pct):Q", scale=alt.Scale(domain=[0, 1.0])).title("Percentage"),
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
    # st.write(df_stats)
