#!/usr/bin/env python3
"""
Builds one PNG per X:
    efficiency_heatmap_X3.png , … , efficiency_heatmap_X10.png
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

DATA_DIR = Path("DATA")
RES_DIR = Path("ASSESSOR_PLOTS")
csv_files = sorted(DATA_DIR.glob("sweep_X*.csv"))
if not csv_files:
    print("No sweep_X*.csv files found under", root)
    sys.exit(1)

for csv in csv_files:
    df = pd.read_csv(csv)
    X_match = re.search(r"X(\d+)", csv.name)
    X = int(X_match.group(1)) if X_match else -1

    df["inv_eff"] = 1.0 / df["eff"]       # convert stored inverse to efficiency

    sl = df[(df.r == R_FIXED) & (df.noise == NOISE_FIXED)]
    if sl.empty:
        print(f"[warn] no rows for X={X} at r={R_FIXED}, σ={NOISE_FIXED}")
        continue

    pivot = sl.pivot(index="cac_dt", columns="mu", values="inv_eff")
    plt.figure(figsize=(4, 3))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=CMAP, vmin=0, vmax=1)
    plt.title(r"$\eta_{CIM}$ per configuration "
              f"\nfor X={X} and $r=0.4,\;\sigma=0.10$", pad=15)
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta t$")
    plt.tight_layout()
    out_png = RES_DIR / f"efficiency_heatmap_X{X}.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print("saved", out_png)
