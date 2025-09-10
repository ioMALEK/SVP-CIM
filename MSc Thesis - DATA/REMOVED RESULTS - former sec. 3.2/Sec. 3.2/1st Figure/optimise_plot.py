#!/usr/bin/env python3
"""
optimise_plot.py  –  post-processing for CIM-SVP sweeps
Produces per-dimension figures and overall figures (efficiency,
absolute efficiency, #trials).  Histogram x-axis is logarithmic.
"""
from __future__ import annotations
import argparse, json, re, warnings
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# silence seaborn ↔ pandas deprecation chatter
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*use_inf_as_na option is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*length-1 tuple to get_group.*")

sns.set_theme(style="whitegrid")

# ─── figure helpers ────────────────────────────────────────────────
def make_hist(norms, opt_norm, out):
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.set_xscale("log")
    if np.all(norms == norms[0]):
        ax.axvline(norms[0], lw=6, color="steelblue")
    else:
        bins = min(40, max(1, np.unique(norms).size - 1))
        sns.histplot(norms, bins=bins, color="steelblue",
                     log_scale=(True, False))
    ax.axvline(opt_norm, ls="--", c="red", label="shortest")
    ax.set_xlabel("norm (log scale)"); ax.set_ylabel("count"); ax.legend()
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def make_conv(curve, out):
    if curve.shape[0] < 2: return
    plt.figure(figsize=(6, 3))
    plt.plot(curve["number"], curve["value"], marker="o")
    plt.xlabel("trial"); plt.ylabel("efficiency"); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()

def make_param_evolution(curve, param_cols, out):
    if len(param_cols) < 2: return
    long = curve.melt(id_vars="number", value_vars=param_cols,
                      var_name="param", value_name="val")
    sns.lineplot(data=long, x="number", y="val", hue="param", marker="o")
    plt.legend(bbox_to_anchor=(1.05, 1.0)); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()

# ─── per-dimension processing – returns None if data incomplete ─────
def per_dimension(tag: str, dim: int):
    data_dir = Path("runs")/"data"/tag/"cim"/f"X{dim:02d}"
    if not (data_dir/"trials.csv").is_file() or not (data_dir/"spin_norms.csv").is_file():
        print(f"[WARN] incomplete data for X{dim:02d}")
        return None

    trials = pd.read_csv(data_dir/"trials.csv")
    norms  = pd.read_csv(data_dir/"spin_norms.csv")["norm"].to_numpy()
    meta   = json.load(open(data_dir/"README.json"))
    opt_norm = meta.get("opt_norm", np.min(norms))

    # parameter columns
    param_map = {c: c.replace("params_", "") for c in trials.columns
                 if c.startswith("params_")}
    trials = trials.rename(columns=param_map)
    param_cols = list(param_map.values())

    plot_dir = Path("runs")/"plots"/tag/f"X{dim:02d}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    make_hist(norms, opt_norm,               plot_dir/"hist.png")
    make_conv(trials[["number","value"]],    plot_dir/"convergence.png")
    make_param_evolution(trials, param_cols, plot_dir/"params.png")

    b_min = meta.get("b_min_norm", np.nan)
    gap   = b_min - opt_norm if np.isfinite(b_min) else np.nan
    abs_eff = trials["value"].max() * gap if np.isfinite(gap) else np.nan
    return trials["value"].max(), len(trials), abs_eff

# ─── aggregate figures ─────────────────────────────────────────────
def aggregate(tag: str):
    base = Path("runs")/"data"/tag/"cim"
    if not base.exists():
        print(f"[WARN] no data for tag {tag}"); return

    dims_all = sorted(int(re.findall(r"\d+", p.name)[0]) for p in base.glob("X*"))
    dims, best_eff, trial_counts, abs_eff = [], [], [], []

    for d in dims_all:
        res = per_dimension(tag, d)
        if res is None: continue
        eff, ntr, abs_e = res
        dims.append(d); best_eff.append(eff); trial_counts.append(ntr); abs_eff.append(abs_e)

    if not dims:
        print("[WARN] nothing to aggregate"); return

    out_dir = Path("runs")/"plots"/tag/f"Overall_X{dims[0]:02d}_X{dims[-1]:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 3))
    sns.lineplot(x=dims, y=best_eff, marker="o")
    plt.xlabel("dimension"); plt.ylabel("best efficiency"); plt.tight_layout()
    plt.savefig(out_dir/"eff_vs_dim.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 3))
    sns.barplot(x=dims, y=trial_counts, color="steelblue")
    plt.xlabel("dimension"); plt.ylabel("# Optuna trials"); plt.tight_layout()
    plt.savefig(out_dir/"trials_vs_dim.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 3))
    sns.lineplot(x=dims, y=abs_eff, marker="o")
    plt.xlabel("dimension"); plt.ylabel("absolute efficiency"); plt.tight_layout()
    plt.savefig(out_dir/"abs_eff_vs_dim.png", dpi=150); plt.close()

    print(f"[INFO] overall figures → {out_dir}")

# ─── CLI ────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--tag", required=True)
    pa.add_argument("--dim", type=int)
    pa.add_argument("--all", action="store_true")
    args = pa.parse_args()

    if args.dim is not None:
        per_dimension(args.tag, args.dim)
    if args.all:
        aggregate(args.tag)

if __name__ == "__main__":
    main()