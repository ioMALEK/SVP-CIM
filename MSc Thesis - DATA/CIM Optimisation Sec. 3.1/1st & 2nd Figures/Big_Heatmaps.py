#!/usr/bin/env python3
# plot_stats_heatmaps.py  – y-tick labels horizontal
#
# • heat_eta_fixed_A.png  (dt 0.02–0.04)
# • heat_eta_fixed_B.png  (dt 0.05–0.07)
# • heat_eta_best.png

from pathlib import Path
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# I/O ---------------------------------------------------------------
DATA_DIR = Path("DATA")
RES_DIR = Path("ASSESSOR_PLOTS")
CSV_IN   = DATA_DIR / "stats.csv"
if not CSV_IN.exists():
    raise FileNotFoundError(CSV_IN)

# load & rename -----------------------------------------------------
df = (pd.read_csv(CSV_IN)
        .rename(columns={"pivot_dt":"dt",
                         "pivot_mu":"mu",
                         "pivot_r":"r",
                         "pivot_sig":"sigma",
                         "eff":"eta"}))

required = {"dt","mu","r","sigma","eta"}
if (missing := required - set(df.columns)):
    raise ValueError("CSV missing: " + ", ".join(missing))

# plot style --------------------------------------------------------
sns.set(style="white", context="talk")
PLOT_KW = dict(dpi=200, bbox_inches="tight")

def save(fig, name):
    fig.savefig(RES_DIR / name, **PLOT_KW)
    plt.close(fig)
    print("•", name, "saved")

def fmt_dt(idx):          # Δt to 3 decimals
    return [f"{v:.3f}" for v in idx]

def plot_heat(pivot, title, fname, cbar_label):
    rows, cols = pivot.shape
    fig = plt.figure(figsize=(1.2*cols+3, 0.55*rows+3))
    ax  = sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis",
                      annot_kws={"size":10}, square=True,
                      cbar_kws=dict(label=cbar_label))
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\Delta t$")
    ax.set_title(title, pad=25)
    # y-tick labels horizontal
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    save(fig, fname)

# ------------------------------------------------------------------
# 1) fixed r = 0.4, σ = 0.10, split into bands
# ------------------------------------------------------------------
fixed = df[(df["r"].round(3)==0.4) & (df["sigma"].round(3)==0.10)]
if fixed.empty:
    warnings.warn("No rows r=0.4 σ=0.10 – skipping fixed maps")
else:
    bands = {"A": (0.02, 0.04),
             "B": (0.05, 0.07)}
    for tag, (lo, hi) in bands.items():
        sub = fixed[(fixed["dt"] >= lo-1e-6) & (fixed["dt"] <= hi+1e-6)]
        if sub.empty:
            warnings.warn(f"No rows in Δt band {tag}")
            continue
        pivot = (sub.groupby(["dt","mu"])["eta"]
                   .mean()
                   .unstack("mu")
                   .sort_index())
        pivot.index = fmt_dt(pivot.index)
        plot_heat(pivot,
                  rf"$\eta_{{CIM}}$ for $r=0.4,\;\sigma=0.10$ "
                  rf"(Δt {lo:.2f}–{hi:.2f})",
                  f"heat_eta_fixed_{tag}.png",
                  r"$\eta_{CIM}$")

# ------------------------------------------------------------------
# 2) best η_CIM per (Δt, μ)
# ------------------------------------------------------------------
pivot_best = (df.groupby(["dt","mu"])["eta"]
                .max()
                .unstack("mu")
                .sort_index())
pivot_best.index = fmt_dt(pivot_best.index)
plot_heat(pivot_best,
          r"Best $\eta_{CIM}$ per $(\Delta t, \mu)$ across all $r,\sigma$",
          "heat_eta_best.png",
          r"max $\eta_{CIM}$")

print("Done – heat-maps saved in", DATA_DIR)
