#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
from scipy import stats

def compute_mean_and_uncertainty(series):
    """Compute mean and 95% confidence interval half-width for a numeric series."""
    values = series.dropna()
    n = len(values)
    if n == 0:
        return None, None, 0
    mean_val = values.mean()
    sem = stats.sem(values)  # standard error of the mean
    ci = stats.t.interval(0.95, n-1, loc=mean_val, scale=sem)
    half_width = (ci[1] - ci[0]) / 2
    return mean_val, half_width, n

def process_file(file_path, label):
    """Process one dataset and return results for eta_CPM/Q=2, eta_CIM/Q=2, eta_CIM/Q=1."""
    df = pd.read_csv(file_path)

    results = {}

    # eta_CPM when Q=2
    subset_cpm = df[df["Q"] == 2]["eta_CPM"]
    mean_cpm, unc_cpm, n_cpm = compute_mean_and_uncertainty(subset_cpm)
    results[f"{label}_eta_CPM_Q2"] = (mean_cpm, unc_cpm, n_cpm)

    # eta_CIM when Q=2
    subset_cim_q2 = df[df["Q"] == 2]["eta_CIM"]
    mean_cim_q2, unc_cim_q2, n_cim_q2 = compute_mean_and_uncertainty(subset_cim_q2)
    results[f"{label}_eta_CIM_Q2"] = (mean_cim_q2, unc_cim_q2, n_cim_q2)

    # eta_CIM when Q=1
    subset_cim_q1 = df[df["Q"] == 1]["eta_CIM"]
    mean_cim_q1, unc_cim_q1, n_cim_q1 = compute_mean_and_uncertainty(subset_cim_q1)
    results[f"{label}_eta_CIM_Q1"] = (mean_cim_q1, unc_cim_q1, n_cim_q1)

    return results

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define input files
    files = {
        "B1": os.path.join(script_dir, "B1_full.csv"),
        "B2": os.path.join(script_dir, "B2_full.csv"),
        "B3": os.path.join(script_dir, "B3_full.csv"),
    }

    all_results = {}

    for label, path in files.items():
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        res = process_file(path, label)
        all_results.update(res)

    # Print nicely
    for key, (mean, unc, n) in all_results.items():
        if mean is None:
            print(f"{key}: No valid data")
        else:
            print(f"{key}: {mean:.6f} ± {unc:.6f} (95% CI, n={n})")

if __name__ == "__main__":
    main()

