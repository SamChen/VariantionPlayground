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
# sys.path.append("src/")
import simulation

EPS = 1e-8

def describe_with_variance(x):
    s = x.describe()
    s['variance'] = x.var()
    return s

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.write("## Playground")
    with open("predefined_biomarker_config.yaml", 'r') as f:
        predefined_configurations = yaml.safe_load(f)["crossover_study"]

    with st.sidebar:
        with st.form("Predefined Configurations:"):
            selected_configuration = st.selectbox("Select a predefined configuration: ", predefined_configurations.keys())
            total_trials = st.number_input("Total number of simulation trials: ", value=1)
            sample_size = st.number_input("Sample size per trial: ", value=1000)
            with_dependency = st.selectbox("Apply dependency simulation: ", [True, False], index=0)
            st.form_submit_button("Submit")
            apply_log = False
            # apply_log = st.selectbox("Apply log transformation (log(value+1)): ", [True, False], index=1)

    default_configuration = predefined_configurations[selected_configuration]
    with st.form("Configuration:"):
        # lmer_formula = st.text_input("Formula for linear mixed model (fixed effects): ", value="value ~  Phaseid")
        # target_var  = st.text_input("Variable that we are interested in", value="Phaseid")
        col1,col2 = st.columns(2)
        col1.write("Phase 1:")
        # n_subj1                  = col1.number_input("Total number of subjets for each group (group1): ", value=10)
        gt_between_subj_mean1    = col1.number_input("Ground truth between subject mean (Phase1): ",
                                                     value=default_configuration["default_gt_between_subj_mean1"],format="%.5f")
        gt_between_subj_var1     = col1.number_input("Ground truth between subject variance (Phase1): ",
                                                     value=default_configuration["default_gt_between_subj_var1"],format="%.5f")
        # gt_within_subj_var_pct1  = col1.number_input("Ground truth within-subject var % out of the within-Phase variability (Phase1): ",
        #                                              value=default_configuration["default_gt_within_subj_percentage1"],format="%.2f")

        col2.write("Phase 2:")
        # n_subj2                  = col2.number_input("Total number of subjets for each Phase (Phase2): ", value=10)
        gt_between_subj_mean2    = col2.number_input("Ground truth between subject mean (Phase2): ",
                                                     value=default_configuration["default_gt_between_subj_mean2"],format="%.5f")
        gt_between_subj_var2     = col2.number_input("Ground truth between subject variance (Phase2): ",
                                                     value=default_configuration["default_gt_between_subj_var2"],format="%.5f")
        # gt_within_subj_var_pct2  = col2.number_input("Ground truth within-subject var % out of the within-Phase variability (Phase2): ",
        #                                              value=default_configuration["default_gt_within_subj_percentage2"],format="%.2f")
        st.form_submit_button("Submit")

    # st.write("The within-subj var:")
    # st.write("Phase1: ", gt_within_subj_var_value1, ", Phase2: ", gt_within_subj_var_value2)

    outputs = defaultdict(list)
    m = 1

    # for gt_within_subj_var_pct in range(0, 101, 10):
    for trial in range(0, total_trials):
        for gt_within_subj_var_pct in range(100, -1, -10):
            gt_within_subj_var_value1 = gt_between_subj_var1 * gt_within_subj_var_pct / 100
            gt_within_subj_var_value2 = gt_between_subj_var2 * gt_within_subj_var_pct / 100
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
                    # (gt_between_subj_mean2 - gt_between_subj_mean1+0.01), gt_between_subj_var1+gt_between_subj_var2,
                    sampled_between_subj_means,
                    (gt_between_subj_mean2 - gt_between_subj_mean1), gt_between_subj_var2,
                    # Use the following if we want to mimic Donanemab's outcomes
                    # (gt_between_subj_mean2 - gt_between_subj_mean1+0.01), gt_between_subj_var2,
                    gt_within_subj_var_value2,
                    m = m,
                    n_subj = sample_size,
                    groupid = 1,
                    seed = seed+50000,
                    # seed = seed,
                )
            else:
                group2, _ = simulation.stats_synthesize_ind(
                    gt_between_subj_mean2, gt_between_subj_var2,
                    gt_within_subj_var_value2,
                    m = m,
                    n_subj = sample_size,
                    groupid = 1,
                    seed = seed+50000,
                    # seed = seed,
                )

            group1, group2 = pd.DataFrame(group1), pd.DataFrame(group2)
            df = pd.merge(group1, group2, on="Subid", suffixes=("_phase1", "_phase2"))
            # if gt_within_subj_var_pct == 100:
            #     mask = (df["value_phase2"] > 0) & (df["value_phase1"] > 0)
            # df = df[(df["value_phase2"] > 0) & (df["value_phase1"] > 0)]
            # df = df[mask]
            act_sample_size = len(df)
            outputs["sample_size"].extend([act_sample_size]*act_sample_size)
            # outputs["reduce"].extend(df["reduce"])
            outputs["value_phase1"].extend(df["value_phase1"])
            outputs["value_phase2"].extend(df["value_phase2"])
            outputs["ratio"].extend([gt_within_subj_var_pct]*act_sample_size)

    df_stats = pd.DataFrame(outputs)
    # shift = abs(min(df["value_phase1"].min(), df["value_phase2"].min())) + EPS
    # df_stats["value_phase1"] = df_stats["value_phase1"] + shift
    # df_stats["value_phase2"] = df_stats["value_phase2"] + shift
    df_stats["reduce"] = df_stats["value_phase2"].apply(lambda x: np.log10(x)) - df_stats["value_phase1"].apply(lambda x: np.log10(x))
    # df_stats["reduce"] = df_stats["value_phase2"] - df_stats["value_phase1"]

    outputs = list()
    for ratio in range(0, 101, 10):
        tmp = df_stats[df_stats["ratio"]==ratio]["reduce"]
        s = tmp.describe()
        s['variance'] = tmp.var()
        o = s.to_dict()
        o["ratio"] = f"{ratio/100}"
        outputs.append(o)
    df_stats = pd.DataFrame(outputs)
    columns = ["ratio", "count","mean","variance","std","min","25%","50%","75%","max"]
    st.write(df_stats[columns].style.format({col: "{:.3f}" for col in ["mean","variance","std"]}))
    # st.write(df_stats[columns], column_config=st.column_config.NumberColumn(format="%.3f"))
