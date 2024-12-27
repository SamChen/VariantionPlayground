import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from collections import defaultdict

from concurrent.futures import ProcessPoolExecutor
import os
import sys
# Get the absolute path to the root folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
src_path = os.path.join(root_path, 'src')
sys.path.append(src_path)
import simulation

import statsmodels.formula.api as smf
eps = 1e-8


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


def run(cfg, sample_size, m, seed):
    seed = seed * 100000
    group1, sampled_between_subj_means = simulation.stats_synthesize_ind(
        cfg["gt_between_subj_mean1"], cfg["gt_between_subj_var1"],
        cfg["gt_within_subj_var_value1"],
        m = m,
        n_subj = sample_size,
        groupid = 0,
        seed = seed,
    )
    assert len(sampled_between_subj_means) == sample_size, f"{len(sampled_between_subj_means)}, {sample_size}"

    if cfg["with_dependency"]:
        group2, _ = simulation.stats_synthesize_dep(
            sampled_between_subj_means,
            simulation.cal_between_phase_diff_mean(cfg["gt_between_subj_mean1"], cfg["gt_between_subj_mean2"]), cfg["gt_within_subj_var_value2"],
            cfg["gt_within_subj_var_value2"],
            m = m,
            n_subj = sample_size,
            groupid = 1,
            seed = seed+50000,
            # seed = seed,
        )
    else:
        group2, _ = simulation.stats_synthesize_ind(
            cfg["gt_between_subj_mean2"], cfg["gt_between_subj_var2"],
            cfg["gt_within_subj_var_value2"],
            m = m,
            n_subj = sample_size,
            groupid = 1,
            seed = seed+50000,
            # seed = seed,
        )

    group1, group2 = pd.DataFrame(group1), pd.DataFrame(group2)
    df = pd.concat([group1, group2]).reset_index(drop=True)
    out = lmer_py(df, cfg["lmer_formula"])

    outputs = {}
    outputs["sample_size_pergroup"] = sample_size
    outputs["sample_size"] = sample_size
    outputs["pvalue"]      = out.pvalues.loc[cfg["target_var"]]
    outputs["coefficient"] = out.params.loc[cfg["target_var"]]
    outputs["M"]           = m
    return outputs


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
    import yaml
    with open("predefined_biomarker_config.yaml", 'r') as f:
        predefined_configurations = yaml.safe_load(f)["parallel_study"]
    with st.sidebar:
        with st.form("Predefined Configurations:"):
            selected_configuration = st.selectbox("Select a predefined configuration: ", predefined_configurations.keys())
            apply_log = st.selectbox("Apply log transformation (log(value+1)): ", [True, False], index=1)
            total_trials = st.number_input("Total number of simulation trials: ", value=20)
            # sample_size = st.number_input("Sample size: ", value=18)
            sample_sizes = st.pills("Sample size: ", options=[18,24,32,48], selection_mode="multi", default=[18,24,32,48])
            st.form_submit_button("Submit")
            # apply_log = False

    default_configuration = predefined_configurations[selected_configuration]
    with st.form("Configuration:"):
        lmer_formula = st.text_input("Formula for linear mixed model (fixed effects): ", value="value ~  Groupid")
        target_var  = st.text_input("Variable that we are interested in", value="Groupid")
        col1,col2 = st.columns(2)
        col1.write("Control group:")
        # n_subj1                  = col1.number_input("Total number of subjets for each group (group1): ", value=10)
        gt_between_subj_mean1    = col1.number_input("Ground truth between subject mean (Control): ",
                                                     value=default_configuration["default_gt_between_subj_mean1"],format="%.5f")
        gt_between_subj_var1     = col1.number_input("Ground truth between subject variance (Control): ",
                                                     value=default_configuration["default_gt_between_subj_var1"],format="%.5f")
        gt_within_subj_var_pct1  = col1.number_input("Ground truth within-subject var % out of the within-Phase variability (Control): ",
                                                     value=default_configuration["default_gt_within_subj_percentage1"],format="%.2f")

        col2.write("Experiment group:")
        # n_subj2                  = col2.number_input("Total number of subjets for each Phase (Phase2): ", value=10)
        gt_between_subj_mean2    = col2.number_input("Ground truth between subject mean (Experiment): ",
                                                     value=default_configuration["default_gt_between_subj_mean2"],format="%.5f")
        gt_between_subj_var2     = col2.number_input("Ground truth between subject variance (Experiment): ",
                                                     value=default_configuration["default_gt_between_subj_var2"],format="%.5f")
        gt_within_subj_var_pct2  = col2.number_input("Ground truth within-subject var % out of the within-Phase variability (Experiment): ",
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
    m = 1
    ratios = [i*10 for i in range(1, 11)]
    cfg = {}
    total_trials           = total_trials
    sample_sizes           = sample_sizes

    cfg["lmer_formula"] = "value ~  Phaseid"
    cfg["target_var"]   = "Phaseid"
    cfg["gt_between_subj_mean1"]    = gt_between_subj_mean1
    cfg["gt_between_subj_var1"]     = gt_between_subj_var1
    cfg["gt_between_subj_mean2"]    = gt_between_subj_mean2
    cfg["gt_between_subj_var2"]     = gt_between_subj_var2
    cfg["gt_within_subj_var_value1"] = gt_within_subj_var_value1
    cfg["gt_within_subj_var_value2"] = gt_within_subj_var_value2
    cfg["with_dependency"] = False

    configs = []
    # For parallel design, we assume the total number of repeated measurement, m, will always be 1.
    m = 1
    for sample_size in sample_sizes:
            for seed in range(0, total_trials):
                configs.append((cfg, sample_size, m, seed))

    def fun_tmp(param):
        return run(*param)

    with ProcessPoolExecutor() as executor:
        outputs = list(executor.map(fun_tmp, configs))

    df_stats = pd.DataFrame(outputs)
    df_stats["M"] = df_stats["M"].apply(lambda x: "1 var=0" if x == 0.5 else int(x))
    # sorted_M = ['1 var=0'] + [str(i) for i in range(m0, mE)]
    sorted_M = [str(i) for i in range(m0, mE)]
    df_stats.to_csv(f"statistic_estimation_parallel_{selected_configuration.split('_')[0]}.csv")
    group = "M"

    chart = draw_line(df_stats, pvalue_threshold=0.05)
    st.altair_chart(chart, theme=None)

    # bar_chart = draw_bar(df_stats, group, sorted_groups=sorted_M, title=f"Sample size: {sample_size}",
    #                      xname="pvalue")

    # bar_chart.save(f"outputs/pTau217_n{sample_size}.png", ppi=400)
    # df_stats.to_csv(f"outputs/pTau217_n{sample_size}.csv")
    # st.altair_chart(bar_chart, theme="streamlit")
