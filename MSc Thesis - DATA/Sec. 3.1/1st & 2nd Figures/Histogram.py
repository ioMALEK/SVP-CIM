#!/usr/bin/env python3
# plot_mean_norm_hist.py
#
# Linear-scale histogram of mean_norm for the fixed settings
#     r = 0.4   and   σ = 0.10
#
# The figure is saved to  TO_PLOT/mean_norm_hist.png

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- user-config ----------------------------------------------------
DATA_DIR  = Path("TO_PLOT")
CSV_PATH  = DATA_DIR / "stats.csv"
R_FIXED   = 0.4
SIG_FIXED = 0.10
N_BINS    = 20                      # histogram resolution
PLOT_KW   = dict(dpi=200, bbox_inches="tight")

# --- load & basic checks -------------------------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

df = (pd.read_csv(CSV_PATH)
        .rename(columns={"pivot_r":"r",
                         "pivot_sig":"sigma"}))

need = {"mean_norm", "r", "sigma"}
if (missing := need - set(df.columns)):
    raise ValueError("CSV missing columns: " + ", ".join(missing))

sub = df[(df["r"].round(3) == R_FIXED) &
         (df["sigma"].round(3) == SIG_FIXED)]

if sub.empty:
    raise ValueError(f"No rows with r={R_FIXED}, σ={SIG_FIXED}")

vals = sub["mean_norm"]

# --- plotting -------------------------------------------------------
sns.set(style="whitegrid", context="talk")
plt.figure(figsize=(8,5))
sns.histplot(vals, bins=N_BINS, color="tab:blue", kde=False)

plt.xlabel("Norm CIM outputs")
plt.ylabel("count")
plt.title(rf"Norms CIM outputs across all optimised configurations, "
          rf"with $\sigma = {SIG_FIXED:.2f}$",
          pad=15)

out_png = DATA_DIR / "mean_norm_hist.png"
plt.savefig(out_png, **PLOT_KW)
plt.close()
print("•", out_png.name, "saved in", DATA_DIR)
