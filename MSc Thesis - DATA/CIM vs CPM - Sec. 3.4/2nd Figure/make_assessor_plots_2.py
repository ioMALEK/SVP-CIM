#!/usr/bin/env python3
"""
make_assessor_plots_2.py · 2025-09-07 

"""

from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ───────── CLI & folders ─────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "DATA",
               help="Folder that contains Deg_count.csv and PER_BRI/")
args = p.parse_args()

HERE      = Path(__file__).resolve().parent
DATA_DIR  = args.data_dir.resolve()
PER_BRI   = DATA_DIR / "PER_BRI"
OUT_DIR   = HERE / "ASSESSOR_PLOTS"

if not (DATA_DIR / "Deg_count.csv").exists():
    sys.exit(f"❌  Missing Deg_count.csv in {DATA_DIR}")
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

# ───────── Figure A – degeneracies ─────────
deg_df = pd.read_csv(DATA_DIR / "Deg_count.csv")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(deg_df["BRI"], deg_df["deg_removed"], marker="o")
ax.set(xlabel="BRI Number", ylabel="Degeneracies Removed Number",
       title="Degeneracies removed per Basis Replacement Iterations (BRI)")
save(fig, "A_deg_removed_vs_BRI.png")

# ───────── Load PER_BRI (0–70) ─────────
def bri_no(p: Path) -> int: return int(p.stem.split("_")[1])

records: list[pd.DataFrame] = []
for fp in sorted(PER_BRI.glob("bri_*.csv"), key=bri_no):
    if bri_no(fp) > 70:
        continue
    df = pd.read_csv(fp)
    df.columns = df.columns.str.strip()
    df.rename(columns={"BRI_time": "BRI time",
                       "BRITime":  "BRI time",
                       "BRI Time": "BRI time",
                       "elapsed_sec": "BRI time"}, inplace=True)
    need = {"global_best", "mean_pool", "eta_CPM_max",
            "eta_CIM_max", "BRI time"}
    if need.difference(df.columns):
        raise ValueError(f"{fp.name}: missing required columns")
    df["BRI"] = bri_no(fp)
    records.append(df)

d = pd.concat(records, ignore_index=True)
d["runtime_cum"] = d["BRI time"].cumsum()

# aliases
bri, rt = d["BRI"], d["runtime_cum"]
avg, best = d["mean_pool"], d["global_best"]
eta_cpm, eta_cim = d["eta_CPM_max"], d["eta_CIM_max"]

# runtime axis helper
def b2r(x): return np.interp(x, bri, rt)
def r2b(x): return np.interp(x, rt, bri)
def add_rt(ax):
    sec = ax.secondary_xaxis("top", functions=(b2r, r2b))
    sec.set_xticks(np.linspace(rt.min(), rt.max(), 6).astype(int))
    sec.set_xlabel("CPM Runtime (s)")

# generic combined plot
def combined(title, curves, fname, yscale="linear", after=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for y, c, ls, lab in curves:
        ax.plot(bri, y, color=c, ls=ls, lw=2, label=lab)
    ax.set_xlabel("BRI Number")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.set_yscale(yscale)
    if after:
        after(ax)                     # decorate after log-scale set
    ax.set_title(title, pad=30)
    add_rt(ax)
    ax.legend()
    save(fig, fname)

# ───────── Combined norms with Q=2 line ─────────
Q2 = 1.49e41
def q2_line(ax):
    ax.axhline(Q2, color="red", ls=":", lw=1.5)
    x_lbl = bri.min() + 0.10 * (bri.max() - bri.min())
    ax.text(x_lbl, Q2 * 1.03,
            r"$1.49\times10^{41}$ – Shortest Norm achieved with $Q=2$",
            color="red", fontsize=10, ha="left", va="bottom")

combined(
    "Shortest norm found and average norm of the reduced basis at each BRI,\nwith runtime (s)",
    [(best, "red", "-",  "shortest norm so far"),
     (avg,  "red", "--", "reduced-basis average norm")],
    "E_combined_norms_dual_x.png",
    yscale="log",
    after=q2_line,
)

# ───────── Combined η_CPM & η_CIM ─────────
combined("$η_{CPM}$ at each BRI, with runtime (s)",
         [(eta_cpm, "black", "-", r"$\eta_{CPM}$")],
         "F_combined_eta_CPM_dual_x.png")

combined("$η_{CIM}$ at each BRI, with runtime (s)",
         [(eta_cim, "red", "-", r"$\eta_{CIM}$")],
         "G_combined_eta_CIM_dual_x.png")

print(f"✅  Plots written to {OUT_DIR.resolve()}")
