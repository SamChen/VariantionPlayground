import numpy as np
import scipy.stats as stats
from itertools import product
from collections import defaultdict
# from typing import List, Dict
from numba import njit
from numba.core import types
from numba.typed import Dict, List

EPS = 1e-4
SHIFT = 1000

def conditional_decorator(condition, decorator):
    """Applies a decorator if the condition is True, otherwise does nothing."""
    def decorator_wrapper(func):
        if condition:
            return decorator(func)
        return func  # Return the original function if condition is False
    return decorator_wrapper

def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

NUMBA_MODE = True

# # TODO: unify the sample synthesis process.
# def basic_synthesis_data(subj_means, subj_vars, seed, groupid, m):
#     """
#     """
#     assert len(subj_vars) == len(subj_means)
#     samples = defaultdict(list)
#     for idx, (subj_mean, subj_var) in enumerate(zip(subj_means, subj_vars)):
#         np.random.seed(seed+2+idx)
#
#         # 3. For each subject, we sample $M$ values/measurements based on that subject's **mean** and $within\_subj\_var$.
#         if subj_var == 0:
#             subj_sample = [subj_mean for i in range(m)]
#         else:
#             subj_sample = np.random.normal(subj_mean, np.sqrt(subj_var), m)
#
#         samples["Phaseid"].extend([groupid for i in subj_sample])
#         samples["Subid"].extend([idx for i in subj_sample])
#         samples["value"].extend(subj_sample)
#         samples["ithMeasurement"].extend([i+groupid*m for i in range(1,m+1)])
#     return samples



# @njit(parallel=True)
@njit
def basic_synthesis_data_numba(subj_means, subj_vars, seed, groupid, m):
    """
    Generates synthetic data for multiple subjects.

    Args:
        subj_means (np.ndarray): Array of subject means.
        subj_vars (np.ndarray): Array of subject variances.
        seed (int): Random seed.
        groupid (int): Group identifier.
        m (int): Number of measurements per subject.

    Returns:
        tuple: A tuple containing lists of arrays for Phaseid, Subid, value, and ithMeasurement.
    """
    n_subjects = len(subj_means)
    assert len(subj_vars) == n_subjects

    # Pre-allocate NumPy arrays for efficiency within Numba
    samples_Phaseid = np.empty((n_subjects, m), dtype=np.int64)
    samples_Subid = np.empty((n_subjects, m), dtype=np.int64)
    samples_value = np.empty((n_subjects, m), dtype=np.float64)
    samples_ithMeasurement = np.empty((n_subjects, m), dtype=np.int64)

    for idx in range(n_subjects):
        np.random.seed(seed + 2 + idx)  # Seed each subject separately

        subj_mean = subj_means[idx]
        subj_var = subj_vars[idx]

        if subj_var == 0:
            subj_sample = np.full(m, subj_mean) # Use np.full for efficiency
        else:
            subj_sample = np.random.normal(subj_mean, np.sqrt(subj_var), m)
            # subj_sample = stats.truncnorm.rvs(EPS, np.inf,
            #                                   loc=subj_mean,
            #                                   scale=np.sqrt(subj_var),
            #                                   size=m)
        samples_Phaseid[idx, :] = groupid
        samples_Subid[idx, :] = idx
        samples_value[idx, :] = subj_sample
        samples_ithMeasurement[idx, :] = np.arange(1 + groupid * m, m + 1 + groupid * m)

    return samples_Phaseid, samples_Subid, samples_value, samples_ithMeasurement


def convert_lists2dict(samples_Phaseid, samples_Subid, samples_value, samples_ithMeasurement):
    samples = defaultdict(list)
    for idx in range(len(samples_Phaseid)):
        sp, ss, sv, si = samples_Phaseid[idx], samples_Subid[idx], samples_value[idx], samples_ithMeasurement[idx]
        samples["Phaseid"].extend(sp)
        samples["Subid"].extend(ss)
        samples["value"].extend(sv)
        samples["ithMeasurement"].extend(si)
    return samples



@njit
def generate_samples_ind(
    curr_between_subj_mean, curr_between_subj_var,
    curr_within_subj_var_value,
    m, n_subj, groupid, seed):

    np.random.seed(seed)
    # 1. Use `ground truth` $between\_subj\_mean$ and $between\_subj\_var$ to sample the mean value for each subject
    subj_means = np.random.normal(curr_between_subj_mean,
                                  np.sqrt(curr_between_subj_var),
                                  size=n_subj)
    subj_means = np.abs(subj_means) + EPS

    # 2. Use `ground truth` **mean over within subject variance** and **variance over within subject variance** to sample the $within\_subj\_var$ for each subject.
    np.random.seed(seed+1)
    subj_vars = [curr_within_subj_var_value for i in range(n_subj)]

    # 3. For each subject, we sample $M$ values/measurements based on that subject's **mean** and $within\_subj\_var$.
    samples = basic_synthesis_data_numba(subj_means, subj_vars, seed, groupid, m)

    return samples, subj_means



def stats_synthesize_ind(
    curr_between_subj_mean, curr_between_subj_var,
    curr_within_subj_var_value,
    m,
    n_subj,
    groupid,
    seed,
):
    group, subj_means = generate_samples_ind(
        curr_between_subj_mean, curr_between_subj_var,
        curr_within_subj_var_value,
        m        = m       ,
        n_subj   = n_subj  ,
        groupid  = groupid ,
        seed     = seed
    )
    if isinstance(group, tuple):
        group = convert_lists2dict(*group)
    return group, subj_means


# TODO: add solution for variance sampling using uniform distribution
@njit
def generate_samples_dep(
    prev_between_subj_means,
    between_phase_diff_mean, between_phase_diff_var,
    curr_within_subj_var_value,
    m, n_subj, groupid, seed):
    """
    TODO: add explanations
    """

    np.random.seed(seed)
    # 1. Use `ground truth` $between\_subj\_mean$ and $between\_subj\_var$ to sample the mean value for each subject
    subj_diffs = np.random.normal(between_phase_diff_mean,
                                  np.sqrt(between_phase_diff_var),
                                  size=n_subj)

    # TODO: add complete sanity check
    if isinstance(prev_between_subj_means, list):
        assert len(prev_between_subj_means) == len(subj_diffs), f"{len(prev_between_subj_means)}, {len(subj_diffs)}"
    elif isinstance(prev_between_subj_means, float):
        pass

    # 2. Use `ground truth` **mean over within subject variance** and **variance over within subject variance** to sample the $within\_subj\_var$ for each subject.
    subj_means = np.abs(prev_between_subj_means + subj_diffs)

    np.random.seed(seed+1)
    subj_vars = [curr_within_subj_var_value for i in range(n_subj)]

    # 3. For each subject, we sample $M$ values/measurements based on that subject's **mean** and $within\_subj\_var$.
    samples = basic_synthesis_data_numba(subj_means, subj_vars, seed, groupid, m)
    # samples = basic_synthesis_data(subj_means, subj_vars, seed, groupid, m)

    return samples, subj_means

def stats_synthesize_dep(
    prev_between_subj_means,
    between_phase_diff_mean, between_phase_diff_var,
    curr_within_subj_var_value,
    m,
    n_subj,
    groupid,
    seed,
):
    group, subj_means = generate_samples_dep(
        prev_between_subj_means,
        between_phase_diff_mean, between_phase_diff_var,
        curr_within_subj_var_value,
        m        = m       ,
        n_subj   = n_subj  ,
        groupid  = groupid ,
        seed     = seed
    )
    if isinstance(group, tuple):
        group = convert_lists2dict(*group)
    return group, subj_means


#
def cal_between_phase_diff_mean(previous_phase_mean, current_phase_mean):
    return current_phase_mean - previous_phase_mean
