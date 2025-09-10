#!/usr/bin/env python3
"""
make_assessor_plots_1.py · 2025-09-07 

"""

from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ───────── CLI / folders ─────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "DATA",
               help="Folder that contains PER_BRI/")
args = p.parse_args()

HERE, DATA_DIR = Path(__file__).resolve().parent, args.data_dir.resolve()
PER_BRI, OUT_DIR = DATA_DIR / "PER_BRI", HERE / "ASSESSOR_PLOTS"

if not PER_BRI.exists():
    sys.exit(f"❌  Missing folder {PER_BRI}")

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir()

def save(fig: plt.Figure, name: str):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=300)
    plt.close(fig)

sns.set_theme(style="whitegrid")

# ───────── Load PER_BRI (0–128) ─────────
def bri_no(p: Path) -> int: return int(p.stem.split("_")[1])

dfs: list[pd.DataFrame] = []
for fp in sorted(PER_BRI.glob("bri_*.csv"), key=bri_no):
    if bri_no(fp) > 128:
        continue
    df = pd.read_csv(fp)
    df.columns = df.columns.str.strip()
    df.rename(columns={"BRI_time": "BRI time",
                       "B_avg":    "mean_pool"}, inplace=True)
    need = {"global_best", "mean_pool", "eta_CIM_max",
            "BRI time", "sparsity", "replacements"}
    miss = need.difference(df.columns)
    if miss:
        raise ValueError(f"{fp.name}: missing {', '.join(miss)}")
    df["BRI"] = bri_no(fp)
    dfs.append(df)

if not dfs:
    sys.exit("❌  No PER_BRI files in range 0–128")

d = pd.concat(dfs, ignore_index=True)
d["runtime_cum"] = d["BRI time"].cumsum()

# ───────── aliases ─────────
bri        = d["BRI"]
rt         = d["runtime_cum"]
avg_norm   = d["mean_pool"].astype(float)
best_norm  = d["global_best"].astype(float)
eta_cim    = d["eta_CIM_max"].astype(float)
sparsity   = d["sparsity"].astype(float)
repl_count = d["replacements"].astype(float)

# dual-axis helper
b_arr, r_arr = bri.to_numpy(), rt.to_numpy()
def b2r(x): return np.interp(x, b_arr, r_arr)
def r2b(x): return np.interp(x, r_arr, b_arr)
def add_rt_axis(ax):
    sec = ax.secondary_xaxis("top", functions=(b2r, r2b))
    sec.set_xticks(np.linspace(r_arr.min(), r_arr.max(), 6).astype(int))
    sec.set_xlabel("CPM Runtime (s)")

# generic combined builder
def combined(curves, title, fname, yscale="linear", after=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for y, c, ls, lab in curves:
        ax.plot(bri, y, color=c, ls=ls, lw=2, label=lab)
    if after:
        after(ax)
    ax.set_xlabel("BRI Number")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.set_yscale(yscale)
    ax.set_title(title, pad=30)
    add_rt_axis(ax)
    ax.legend()
    save(fig, fname)

# ───────── Figure 1 – norms + fixed asymptote ─────────
Q1_LEVEL = 1.45e45  # fixed value

def decorate_norms(ax):
    # make sure the level is visible
    ymin, ymax = ax.get_ylim()
    if Q1_LEVEL < ymin:
        ax.set_ylim(Q1_LEVEL * 0.8, ymax)
    # dotted line
    ax.axhline(Q1_LEVEL, color="red", ls=":", lw=1.5, zorder=0)
    # inline label
    x_label = bri.min() + 0.10 * (bri.max() - bri.min())
    label_text = r"$1.45\times10^{45}$ – Shortest Norm achieved with $Q=1$"
    ax.text(x_label, Q1_LEVEL * 1.03, label_text,
            color="red", fontsize=10, ha="left", va="bottom")

combined(
    [(best_norm, "red", "-",  "shortest norm so far"),
     (avg_norm,  "red", "--", "reduced-basis average norm")],
    "Shortest norm found and average norm of the reduced basis\n"
    "at each BRI, with runtime (s)",
    "combined_norms_dual_x.png",
    yscale="log",
    after=decorate_norms,
)

# ───────── Figure 2 – η_CIM ─────────
combined(
    [(eta_cim, "red", "-", r"$\eta_{CIM}$")],
    r"$\eta_{CIM}$ at each BRI, with runtime (s)",
    "combined_eta_CIM_dual_x.png",
)

# ───────── Figure 3 – replacements ───
combined(
    [(repl_count, "blue", "-", "replacement count")],
    "Replacement count at each BRI, with runtime (s)",
    "combined_replacements_dual_x.png",
)

# ───────── Figure 4 – sparsity ────────
combined(
    [(sparsity, "black", "-", "sparsity")],
    "Sparsity at each BRI, with runtime (s)",
    "combined_sparsity_dual_x.png",
)

# ───────── Figure 5 – sparsity + η_CIM ─
combined(
    [(sparsity, "black", "-", "sparsity"),
     (eta_cim,  "red",   "-", r"$\eta_{CIM}$")],
    "Sparsity (black) and $\\eta_{CIM}$ (red)\n"
    "at each BRI, with runtime (s)",
    "combined_sparsity_etaCIM_dual_x.png",
)

print(f"✅  Five combined figures written to {OUT_DIR.resolve()}")
