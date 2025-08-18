#!/usr/bin/env python3
"""
make_inv_eff_heatmaps.py

Usage
    python make_inv_eff_heatmaps.py  <results_folder>

<results_folder> is the directory that contains the CSV files produced by the
main sweep script (e.g. CIM_sensitivity_results_2025-08-14_02-31-07).

The script builds one PNG per X:
    inv_eff_heatmap_X3.png , … , inv_eff_heatmap_X10.png
"""

import sys, re
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# config for the slice we visualise
R_FIXED     = 0.4
NOISE_FIXED = 0.10
CMAP        = "viridis"           # higher is better (closer to 1)
# ----------------------------------------------------------------------

if len(sys.argv) != 2:
    print("Usage:  python make_inv_eff_heatmaps.py  <results_folder>")
    sys.exit(1)

root = Path(sys.argv[1]).expanduser().resolve()
csv_files = sorted(root.glob("sweep_X*.csv"))
if not csv_files:
    print("No sweep_X*.csv files found under", root)
    sys.exit(1)

for csv in csv_files:
    df = pd.read_csv(csv)
    X_match = re.search(r"X(\d+)", csv.name)
    X = int(X_match.group(1)) if X_match else -1

    df["inv_eff"] = 1.0 / df["eff"]       # new metric

    sl = df[(df.r == R_FIXED) & (df.noise == NOISE_FIXED)]
    if sl.empty:
        print(f"[warn] no rows for X={X} at r={R_FIXED}, σ={NOISE_FIXED}")
        continue

    pivot = sl.pivot(index="cac_dt", columns="mu", values="inv_eff")
    plt.figure(figsize=(4, 3))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=CMAP, vmin=0, vmax=1)
    plt.title(f"  Efficiency of parameters Configuration  \n for X={X}  and fixed (r={R_FIXED}, σ={NOISE_FIXED})")
    plt.xlabel("μ")
    plt.ylabel("Δt")
    plt.tight_layout()
    out_png = root / f"inv_eff_heatmap_X{X}.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print("saved", out_png)
