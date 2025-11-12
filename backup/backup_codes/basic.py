# import streamlit as st
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

    np.random.seed(seed+1)
    # 2. Use `ground truth` **mean over within subject variance** and **variance over within subject variance** to sample the $within\_subj\_var$ for each subject.
    subj_vars = np.random.normal(within_subj_var_mean,
                                 np.sqrt(within_subj_var_std),
                                 size=n_subj)
    subj_vars = np.abs(subj_vars)

    samples = defaultdict(list)
    for idx, (subj_mean, subj_var) in enumerate(zip(subj_means, subj_vars)):
        np.random.seed(seed+2+idx)

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
    shift=0,
    scale=1
):
    group = generate_samples(
        between_subj_mean, between_subj_var,
        within_subj_var_mean, within_subj_var_std,
        m        = m       ,
        n_subj   = n_subj  ,
        groupid  = groupid ,
        seed     = seed
    )

    group["value"] = group["value"].apply(lambda x: x*scale+shift)
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
