import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

import yaml
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import os
import argparse
import sys
# Get the absolute path to the root folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src folder to sys.path
src_path = os.path.join(root_path, 'src')
sys.path.append(src_path)
import simulation

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

def run(cfg, sample_size, m_group1, m_group2, seed):
    seed = seed * 100000
    group1, sampled_between_subj_means = simulation.stats_synthesize_ind(
        cfg["gt_between_subj_mean1"], cfg["gt_between_subj_var1"],
        cfg["gt_within_subj_var_value1"],
        m = m_group1,
        n_subj = sample_size,
        groupid = 0,
        seed = seed,
    )
    assert len(sampled_between_subj_means) == sample_size, f"{len(sampled_between_subj_means)}, {sample_size}"

    if cfg["with_dependency"]:
        group2, _ = simulation.stats_synthesize_dep(
            sampled_between_subj_means,
            (cfg["gt_between_subj_mean2"] - cfg["gt_between_subj_mean1"])*cfg["efficacy"], cfg["gt_within_subj_var_value2"],
            cfg["gt_within_subj_var_value2"],
            m = m_group2,
            n_subj = sample_size,
            groupid = 1,
            seed = seed+50000,
        )
    else:
        gt_between_subj_mean2 = cfg["gt_between_subj_mean1"] + (cfg["gt_between_subj_mean2"] - cfg["gt_between_subj_mean1"])*cfg["efficacy"]
        group2, _ = simulation.stats_synthesize_ind(
            gt_between_subj_mean2, cfg["gt_between_subj_var2"],
            cfg["gt_within_subj_var_value2"],
            m = m_group2,
            n_subj = sample_size,
            groupid = 1,
            seed = seed+50000,
        )

    group1, group2 = pd.DataFrame(group1), pd.DataFrame(group2)

    df = pd.concat([group1, group2]).reset_index(drop=True)
    out = lmer_py(df, cfg["lmer_formula"])

    outputs = {}
    outputs["sample_size_pergroup"] = sample_size
    outputs["sample_size"] = sample_size
    outputs["M"]           = f"{m_group1}_{m_group2}"
    # outputs["M"]           = m

    out = lmer_py(df, cfg["lmer_formula"])
    outputs["pvalue"]      = out.pvalues.loc[cfg["target_var"]]

    # out = stats.pearsonr(group1["value"], group2["value"])
    # outputs["pearsonr"]    = out.statistic
    # outputs["pearsonr_p"]  = out.pvalue
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple program demonstrating argparse.")

    # Positional arguments (required)
    parser.add_argument("--output-folder", default="outputs", help="Path to the output file")

    # TODO: add more arguments
    # # Optional arguments
    # parser.add_argument("", choices=["process", "analyze", "report"], default="process", help="Operating mode (default: process)")
    # parser.add_argument("-m", "--mode", choices=["process", "analyze", "report"], default="process", help="Operating mode (default: process)")
    parser.add_argument("--num-trials", type=int, default=20, help="Total number of trials (default: 20)")

    # # Mutually exclusive group
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument("--with_dependency", action="store_true", help="Run in fast mode (disables some features)")
    # group.add_argument("--slow", action="store_true", help="Run in slow mode (enables extra features)")


    args = parser.parse_args()

    # Accessing the arguments
    output_folder          = args.output_folder
    # Experiment configuration
    total_trials           = args.num_trials
    with_dependency        = False
    efficacy               = 1.0
    sample_sizes           = [18,24,32,48,56,60,65,68]
    lmer_formula           = "value ~ Phaseid"
    target_var             = "Phaseid"

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    with open("../predefined_biomarker_config.yaml", 'r') as f:
        predefined_configurations = yaml.safe_load(f)["crossover_study"]

    selected_configurations = predefined_configurations.keys()

    selected_configuration = "pTau217_(U/mL)"

    ratios = [i*10 for i in range(1, 11)]
    for ratio in ratios:
        cfg = {}
        cfg["with_dependency"] = with_dependency
        cfg["efficacy"]        = efficacy
        cfg["lmer_formula"]    = lmer_formula
        cfg["target_var"]      = target_var

        default_configuration = predefined_configurations[selected_configuration]
        cfg["gt_between_subj_mean1"]    = default_configuration["default_gt_between_subj_mean1"]
        cfg["gt_between_subj_var1"]     = default_configuration["default_gt_between_subj_var1"]
        gt_within_subj_var_pct1         = ratio
        cfg["gt_between_subj_mean2"]    = default_configuration["default_gt_between_subj_mean2"]
        cfg["gt_between_subj_var2"]     = default_configuration["default_gt_between_subj_var2"]
        gt_within_subj_var_pct2         = ratio
        cfg["gt_within_subj_var_value1"] = cfg["gt_between_subj_var1"] * gt_within_subj_var_pct1 / 100
        cfg["gt_within_subj_var_value2"] = cfg["gt_between_subj_var2"] * gt_within_subj_var_pct2 / 100

        configs = []
        # m0 = 1
        # mE = 11
        total_trials = total_trials
        sample_sizes = sample_sizes
        for sample_size in sample_sizes:
            for seed in range(0, total_trials):
                # for m in range(m0, mE):
                configs.append((cfg, sample_size, 2, 4, seed))

        def fun_tmp(param):
            return run(*param)
        with ProcessPoolExecutor() as executor:
            outputs = list(executor.map(fun_tmp, configs))

        df_stats = pd.DataFrame(outputs)
        df_stats.to_csv(os.path.join(output_folder, f"statistic_estimation_crossover_{selected_configuration.split('_')[0]}_{ratio}.csv"))
