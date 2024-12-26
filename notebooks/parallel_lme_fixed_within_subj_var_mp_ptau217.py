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
# import concurrent
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

# from rpy2.robjects import r, globalenv, conversion, default_converter
# from rpy2.robjects.packages import importr, data
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.conversion import localconverter
# import rpy2.robjects.packages as rpackages
# def lmer_r(data, formula):
#     output = {}
#     with conversion.localconverter(default_converter + pandas2ri.converter):
#         # stats     = importr("stats")
#         # lme4_r    = importr('lme4')
#         base      = importr('base')
#         lme4_test = importr('lmerTest')
#
#         r_out = lme4_test.lmer(formula, dat=data)
#         summary = base.summary(r_out)
#     return summary["coefficients"][-1,-1]

def lmer_py(data, formula):
    md = smf.mixedlm(formula,
                     data=data,
                     groups=data["Subid"],
                     re_formula="1",
                     # vc_formula={"ithMeasurement": "0+ithMeasurement"}
                     )
    mdf = md.fit(method=["powell", "lbfgs"])
    assert mdf.converged == True
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
            (cfg["gt_between_subj_mean2"] - cfg["gt_between_subj_mean1"])*cfg["efficacy"], cfg["gt_within_subj_var_value2"],
            cfg["gt_within_subj_var_value2"],
            m = m,
            n_subj = sample_size,
            groupid = 1,
            seed = seed+50000,
        )
    else:
        gt_between_subj_mean2 = cfg["gt_between_subj_mean1"] + (cfg["gt_between_subj_mean2"] - cfg["gt_between_subj_mean1"])*cfg["efficacy"]
        group2, _ = simulation.stats_synthesize_ind(
            gt_between_subj_mean2, cfg["gt_between_subj_var2"],
            cfg["gt_within_subj_var_value2"],
            m = m,
            n_subj = sample_size,
            groupid = 1,
            seed = seed+50000,
        )

    group1, group2 = pd.DataFrame(group1), pd.DataFrame(group2)

    df = pd.concat([group1, group2]).reset_index(drop=True)

    outputs = {}
    outputs["sample_size_pergroup"] = sample_size
    outputs["sample_size"] = sample_size * 2
    outputs["M"]           = m

    out = lmer_py(df, cfg["lmer_formula"])
    outputs["pvalue"]      = out.pvalues.loc[cfg["target_var"]]
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple program demonstrating argparse.")

    # Positional arguments (required)
    parser.add_argument("--output-folder", default="outputs", help="Path to the output file")
    parser.add_argument("--num-trials", type=int, default=20, help="Total number of trials (default: 20)")

    # TODO: add more arguments
    args = parser.parse_args()


    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    with open("../predefined_biomarker_config.yaml", 'r') as f:
        predefined_configurations = yaml.safe_load(f)["parallel_study"]
    selected_configurations = predefined_configurations.keys()

    selected_configuration = "pTau217_(U/mL)"

    # Accessing the arguments
    output_folder          = args.output_folder
    total_trials           = args.num_trials
    # Experiment configuration
    with_dependency        = False
    efficacy               = 1.0
    sample_sizes           = [18,24,32,48,56,60,65,68]
    lmer_formula           = "value ~ Phaseid"
    target_var             = "Phaseid"

    ratios = [i*10 for i in range(1, 11)]
    for ratio in ratios:
        cfg = {}
        cfg["with_dependency"] = with_dependency
        cfg["efficacy"]        = efficacy
        total_trials           = total_trials
        sample_sizes           = sample_sizes
        cfg["lmer_formula"]    = lmer_formula
        cfg["target_var"]      = target_var

        default_configuration = predefined_configurations[selected_configuration]
        cfg["gt_between_subj_mean1"]     = default_configuration["default_gt_between_subj_mean1"]
        cfg["gt_between_subj_var1"]      = default_configuration["default_gt_between_subj_var1"]
        gt_within_subj_var_pct1          = ratio
        cfg["gt_between_subj_mean2"]     = default_configuration["default_gt_between_subj_mean2"]
        cfg["gt_between_subj_var2"]      = default_configuration["default_gt_between_subj_var2"]
        gt_within_subj_var_pct2          = ratio
        cfg["gt_within_subj_var_value1"] = cfg["gt_between_subj_var1"] * gt_within_subj_var_pct1 / 100
        cfg["gt_within_subj_var_value2"] = cfg["gt_between_subj_var2"] * gt_within_subj_var_pct2 / 100

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
        df_stats.to_csv(os.path.join("outputs", f"statistic_estimation_parallel_{selected_configuration.split('_')[0]}_{ratio}.csv"))
