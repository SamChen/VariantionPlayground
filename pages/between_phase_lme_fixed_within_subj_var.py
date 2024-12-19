import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

import yaml
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from collections import defaultdict

import statsmodels.formula.api as smf

import os
import sys
# Get the absolute path to the root folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
src_path = os.path.join(root_path, 'src')
sys.path.append(src_path)
import simulation

EPS = 1e-8

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


def draw_bar(df, group, sorted_groups, title, xname="Pvalue", fontsize = 18):
    base = alt.Chart().transform_joinaggregate(
        total='count(*)',
        groupby=[group]
    ).transform_calculate(
        pct='1 / datum.total'
    )
    bar = base.mark_bar().encode(
        alt.X(f"{xname}:Q").bin(extent=[0, 1], step=0.05).title("Pvalue"),
        alt.Y("sum(pct):Q", scale=alt.Scale(domain=[0, 1.0])).title("Percentage"),
        alt.Color(f"{group}:N", sort=sorted_groups)
    )
    rule = base.mark_rule(color="red").encode(
        # alt.Y("sum(pct):Q", scale=alt.Scale(domain=[0, 1.0])).title("Percentage"),
        alt.Y("threshold:Q"),
    )

    bar_chart = alt.layer(bar, rule).transform_calculate(
        threshold="0.8"
    ).properties(
        width=300,
        height=300,
    ).facet(
        # facet="M:N",
        alt.Facet(f"{group}:N", sort=sorted_groups, header=alt.Header(labelFontSize=fontsize, titleFontSize=fontsize)),
        columns=3,
        data=df,
        title=title,
    ).resolve_scale(
        x='independent'
    ).configure_axis(
        labelFontSize = fontsize,
        titleFontSize = fontsize,
    ).configure_title(
        fontSize= fontsize+6
    )
    return bar_chart


def draw_line(df, pvalue_threshold = 0.05):
    df = df.copy()
    df[f"pvalue lower than {pvalue_threshold}"] = df["pvalue"] <= pvalue_threshold
    sign_df = df.groupby(["M","sample_size"])[f"pvalue lower than {pvalue_threshold}"].aggregate(["sum", "count"]).reset_index()
    sign_df["Percentage"] = sign_df["sum"] / sign_df["count"]

    base = alt.Chart(sign_df)
    line = base.mark_line() + base.mark_circle()
    line = line.encode(
        alt.X("M:O"),
        alt.Y("Percentage", scale=alt.Scale(domain=[0, 1.0])),
        alt.Color("sample_size:O", scale=alt.Scale(scheme="set2"))
    )
    rule = base.mark_rule(color="red").encode(
        # alt.Y("sum(pct):Q", scale=alt.Scale(domain=[0, 1.0])).title("Percentage"),
        alt.Y("threshold:Q"),
    )
    chart = alt.layer(line, rule).transform_calculate(
        threshold="0.8"
    ).properties(
        width=1000,
        height=500,
    )
    return chart

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.write("## Playground")
    with open("predefined_biomarker_config.yaml", 'r') as f:
        predefined_configurations = yaml.safe_load(f)["crossover_study"]

    with st.sidebar:
        with st.form("Predefined Configurations:"):
            selected_configuration = st.selectbox("Select a predefined configuration: ", predefined_configurations.keys())
            total_trials = st.number_input("Total number of simulation trials: ", value=20)
            with_dependency = st.selectbox("Apply dependency simulation: ", [True, False], index=0)
            # sample_size = st.number_input("Sample size: ", value=18)
            sample_sizes = st.pills("Sample size: ", options=[18,24,32,48], selection_mode="multi", default=[18, 24, 32, 48])
            st.form_submit_button("Submit")
            # apply_log = False
            # apply_log = st.selectbox("Apply log transformation (log(value+1)): ", [True, False], index=1)

    default_configuration = predefined_configurations[selected_configuration]
    with st.form("Configuration:"):
        lmer_formula = st.text_input("Formula for linear mixed model (fixed effects): ", value="value ~  Phaseid")
        target_var  = st.text_input("Variable that we are interested in", value="Phaseid")
        col1,col2 = st.columns(2)
        col1.write("Phase 1:")
        # n_subj1                  = col1.number_input("Total number of subjets for each group (group1): ", value=10)
        gt_between_subj_mean1    = col1.number_input("Ground truth between subject mean (Phase1): ",
                                                     value=default_configuration["default_gt_between_subj_mean1"],format="%.5f")
        gt_between_subj_var1     = col1.number_input("Ground truth between subject variance (Phase1): ",
                                                     value=default_configuration["default_gt_between_subj_var1"],format="%.5f")
        gt_within_subj_var_pct1  = col1.number_input("Ground truth within-subject var % out of the within-Phase variability (Phase1): ",
                                                     value=default_configuration["default_gt_within_subj_percentage1"],format="%.2f")

        col2.write("Phase 2:")
        # n_subj2                  = col2.number_input("Total number of subjets for each Phase (Phase2): ", value=10)
        gt_between_subj_mean2    = col2.number_input("Ground truth between subject mean (Phase2): ",
                                                     value=default_configuration["default_gt_between_subj_mean2"],format="%.5f")
        gt_between_subj_var2     = col2.number_input("Ground truth between subject variance (Phase2): ",
                                                     value=default_configuration["default_gt_between_subj_var2"],format="%.5f")
        gt_within_subj_var_pct2  = col2.number_input("Ground truth within-subject var % out of the within-Phase variability (Phase2): ",
                                                     value=default_configuration["default_gt_within_subj_percentage2"],format="%.2f")
        st.form_submit_button("Submit")

    gt_within_subj_var_value1 = gt_between_subj_var1 * gt_within_subj_var_pct1 / 100
    gt_within_subj_var_value2 = gt_between_subj_var2 * gt_within_subj_var_pct2 / 100
    st.write("The within-subj var:")
    st.write("Phase1: ", gt_within_subj_var_value1, ", Phase2: ", gt_within_subj_var_value2)

    outputs = defaultdict(list)
    m0 = 1
    mE = 11
    # mE = 7
    for sample_size in sample_sizes:
        progress_text = f"Operation on sample size {sample_size} is in progress. Please wait."
        my_bar = st.progress(0.0, text=progress_text)
        for m in range(m0, mE):
            for trial in range(0, total_trials):
                percent_complete = ((m-m0) * total_trials + trial+1) / ((mE-m0) * total_trials)
                my_bar.progress(percent_complete, text=progress_text)

                seed = trial * 100000
                group1, sampled_between_subj_means = simulation.stats_synthesize_ind(
                    gt_between_subj_mean1, gt_between_subj_var1,
                    gt_within_subj_var_value1,
                    m = m,
                    n_subj = sample_size,
                    groupid = 0,
                    seed = seed,
                )
                assert len(sampled_between_subj_means) == sample_size, f"{len(sampled_between_subj_means)}, {sample_size}"

                if with_dependency:
                    group2, _ = simulation.stats_synthesize_dep(
                        # sampled_between_subj_means,
                        # (gt_between_subj_mean2 - gt_between_subj_mean1), gt_between_subj_var1+gt_between_subj_var2,
                        sampled_between_subj_means,
                        (gt_between_subj_mean2 - gt_between_subj_mean1), gt_between_subj_var2,
                        gt_within_subj_var_value2,
                        m = m,
                        n_subj = sample_size,
                        groupid = 1,
                        # seed = seed+50000,
                        seed = seed,
                    )
                else:
                    group2, _ = simulation.stats_synthesize_ind(
                        gt_between_subj_mean2, gt_between_subj_var2,
                        gt_within_subj_var_value2,
                        m = m,
                        n_subj = sample_size,
                        groupid = 1,
                        # seed = seed+50000,
                        seed = seed,
                    )

                group1, group2 = pd.DataFrame(group1), pd.DataFrame(group2)

                df = pd.concat([group1, group2]).reset_index(drop=True)
                # pvalue = lmer_r(df, lmer_formula)
                # outputs["pvalue"].append(pvalue)
                out = lmer_py(df, lmer_formula)
                # st.write(out.pvalues)
                outputs["pvalue"].append(out.pvalues.loc[target_var])
                outputs["coefficient"].append(out.params.loc[target_var])
                outputs["sample_size"].append(sample_size)
                outputs["M"].append(m)

    df_stats = pd.DataFrame(outputs)
    df_stats["M"] = df_stats["M"].apply(lambda x: "1 var=0" if x == 0.5 else int(x))
    df_stats["pvalue"] = df_stats.apply(lambda x: 1.0 if x["coefficient"] > 0 else x["pvalue"], axis=1)
    # sorted_M = ['1 var=0'] + [str(i) for i in range(m0, mE)]
    sorted_M = [str(i) for i in range(m0, mE)]
    # df_stats.to_csv(f"statistic_estimation_{sample_size}.csv")
    group = "M"

    chart = draw_line(df_stats, pvalue_threshold=0.05)
    st.altair_chart(chart, theme=None)

    # bar_chart = draw_bar(df_stats, group, sorted_groups=sorted_M, title=f"Sample size: {sample_size}",
    #                      xname="pvalue")

    # bar_chart.save(f"outputs/pTau217_n{sample_size}.png", ppi=400)
    # df_stats.to_csv(f"outputs/pTau217_n{sample_size}.csv")
    # st.altair_chart(bar_chart, theme="streamlit")
