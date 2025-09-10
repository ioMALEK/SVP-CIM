#!/usr/bin/env python3
"""
analyze_cim_results.py

Purpose:
- Post-process existing CIM run outputs to produce rich "efficiency" and percentile analyses.
- Uses uniform-over-spins Monte Carlo per X to define a reference CDF for norms.
- Compares CIM output norms to the uniform reference:
  - Best/mean percentile per X
  - Percentile distribution per X (hist/CDF/heatmap)
  - Min/Max attainable norms (MC estimates)
  - Quantiles vs X, "within factor of MC-min" vs X
  - Convergence curves and convergence-time overlay (approx. via sim_time)
  - Throughput vs X
  - Distribution shape summaries (violins/boxen), tail CDFs
  - Efficiency metrics (Jensen–Shannon divergence, AUC gap)
  - Bands mass, modal norm/peak
  - Optional degeneracy histogram for small X (exact enumeration)

Where to place this file:
- Put this script in the project root (same level as CIM_Performance_worst_case.py).
- It expects to find:
  - results directory: CIM_performance_worst_case_results/<timestamp>/data
  - lattice basis files: svp_lattices/dim{dim}_seed{seed}.txt

Axis conventions:
- Any axis with "norm" uses log scale.
- Any axis with "X" uses linear scale.

Usage (example):
  python3 analyze_cim_results.py --dim 50 --seed 17 --results_dir CIM_performance_worst_case_results/2025-08-10_00-23-57 --mc_samples 50000
"""

import os
import re
import ast
import csv
import sys
import glob
import math
import time
import argparse
import contextlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from decimal import Decimal, getcontext
getcontext().prec = 100

# ----------------- IO helpers -----------------
def load_local_lattice(dim, seed, folder="svp_lattices"):
    path = os.path.join(folder, f"dim{dim}_seed{seed}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Lattice file {path} not found.")
    with open(path, "r") as f:
        text = f.read()
    try:
        B = ast.literal_eval(text)
        B = np.array(B, dtype=object)  # Python ints
    except Exception:
        rows = []
        for line in text.strip().splitlines():
            nums = re.findall(r"-?\d+", line)
            if nums:
                rows.append(list(map(int, nums)))
        B = np.array(rows, dtype=object)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"Loaded basis has shape {B.shape}, expected square.")
    return B

def vec_sq_norm_int(vec):
    return int(sum(int(x) * int(x) for x in vec))

def latest_results_dir(parent="CIM_performance_worst_case_results"):
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"Results folder '{parent}' not found.")
    subdirs = [d for d in glob.glob(os.path.join(parent, "*")) if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError(f"No subfolders inside '{parent}'.")
    subdirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    return subdirs[0]

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

# ----------------- Gram builders -----------------
def gram_from_basis_rows_int(B):
    K = B.shape[0]
    G = np.empty((K, K), dtype=object)
    for i in range(K):
        bi = B[i]
        for j in range(i, K):
            bj = B[j]
            s = 0
            for t in range(bi.shape[0]):
                s += int(bi[t]) * int(bj[t])
            G[i, j] = s
            G[j, i] = s
    return G

def gram_from_basis_rows_float_and_scale(B):
    """
    Return (G_float64_normalized, scale=max_abs_entry_of_int_Gram)
    """
    G_int = gram_from_basis_rows_int(B)
    K = G_int.shape[0]
    max_abs = max(1, max(abs(int(G_int[i, j])) for i in range(K) for j in range(K)))
    Gf = np.empty((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            Gf[i, j] = float(int(G_int[i, j]) / max_abs)
    return Gf, max_abs

# ----------------- Basis ordering -----------------
def basis_sorted_by_row_norm(B):
    sq = np.array([vec_sq_norm_int(B[i]) for i in range(B.shape[0])], dtype=object)
    perm = np.argsort(np.array([int(x) for x in sq]))
    return B[perm], perm

# ----------------- Data discovery -----------------
def find_norms_file(data_dir, X):
    """
    Find the norms file for a given X. If multiple, choose the largest (most runs).
    """
    pattern = os.path.join(data_dir, f"norms_X{X}_runs*.npy")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: (os.path.getsize(p), p))
    return files[-1]

def load_summary_if_exists(data_dir):
    """
    Returns dict[X] -> row dict with keys from header.
    """
    path = os.path.join(data_dir, "CIM_X_scan_summary.txt")
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        # normalize header keys
        header = [h.strip() for h in header]
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if not parts or len(parts) < 1:
                continue
            try:
                X = int(parts[0])
                row = dict(zip(header, parts))
                # alias sim_time
                if "sim_time_sec" not in row and "sim_time" in row:
                    row["sim_time_sec"] = row["sim_time"]
                out[X] = row
            except:
                continue
    return out

def parse_hpo_file_if_exists(data_dir, X):
    """
    Parse hyperparams_X{X}.txt if available.
    Returns list of dicts per trial with numeric fields, and best (min best_sq_norm) record.
    """
    path = os.path.join(data_dir, f"hyperparams_X{X}.txt")
    if not os.path.isfile(path):
        return [], None
    out = []
    best = None
    with open(path, "r") as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != len(header):
                continue
            rec = dict(zip(header, parts))
            # coerce numeric
            for k in ["trial","cac_time_step","cac_r","cac_mu","cac_noise","steps","rand_scaling","batch_runs","best_sq_norm"]:
                if k in rec:
                    try:
                        if k in ["trial","steps","batch_runs"]:
                            rec[k] = int(float(rec[k]))
                        else:
                            rec[k] = float(rec[k])
                    except:
                        pass
            out.append(rec)
            if "best_sq_norm" in rec:
                if best is None or (isinstance(rec["best_sq_norm"], (int,float)) and rec["best_sq_norm"] < best["best_sq_norm"]):
                    best = rec
    return out, best

# ----------------- Monte Carlo over spins -----------------
def uniform_spins_qf_samples(Gf, n_samples=50000, batch=4096, rng=None):
    """
    Draw uniform spins s ∈ {-1,1}^K and compute qf = s^T Gf s (float64), using BLAS batches.
    Returns an array of shape (n_samples,).
    """
    if rng is None:
        rng = np.random.RandomState(0)
    K = Gf.shape[0]
    out = np.empty(n_samples, dtype=np.float64)
    wptr = 0
    while wptr < n_samples:
        m = min(batch, n_samples - wptr)
        S = rng.choice([-1.0, 1.0], size=(m, K)).astype(np.float64, copy=False)
        SG = S @ Gf
        qf = np.einsum("ij,ij->i", SG, S, dtype=np.float64)
        out[wptr:wptr+m] = qf
        wptr += m
    return out

# ----------------- Degeneracy (small X) -----------------
def enumerate_degeneracy_counts(G_int, x_enum_max=18):
    """
    For small X, enumerate all spins and build a histogram of degeneracy counts:
    - Compute q_int = s^T G_int s (exact int).
    - Count occurrences per unique q_int.
    - Return dict: {degeneracy_count: how_many_distinct_norms_have_this_count}
    """
    K = G_int.shape[0]
    if K > x_enum_max:
        return None  # skip
    n_states = 1 << K
    counts = {}
    G = G_int
    for state in range(n_states):
        s = np.empty(K, dtype=np.int8)
        for i in range(K):
            s[i] = 1 if (state >> i) & 1 else -1
        total = 0
        for i in range(K):
            total += int(G[i, i])
            si = int(s[i])
            for j in range(i+1, K):
                total += 2 * si * int(s[j]) * int(G[i, j])
        counts[total] = counts.get(total, 0) + 1
    deg_hist = {}
    for _, c in counts.items():
        deg_hist[c] = deg_hist.get(c, 0) + 1
    return deg_hist

# ----------------- Plot helpers -----------------
def save_png(fig, out_png):
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def set_norm_log_axis(ax, axis='x'):
    if axis == 'x':
        ax.set_xscale('log')
    else:
        ax.set_yscale('log')

def annotate_definition(ax):
    txt = "Valid output: finite, positive norm from returned spins."
    ax.text(0.01, -0.18, txt, transform=ax.transAxes, fontsize=9, va='top', ha='left')

# ----------------- Per-X analysis -----------------
def per_x_analysis(X, B_sorted, data_dir, analysis_root, folders, mc_samples, mc_batch,
                   x_enum_max, k_factors, rng, summary_row):
    os.makedirs(analysis_root, exist_ok=True)

    # Locate norms file for this X
    norms_file = find_norms_file(data_dir, X)
    if norms_file is None:
        return None

    norms_arr = np.load(norms_file, mmap_mode='r')
    cim_norms = np.asarray(norms_arr)
    cim_norms = cim_norms[np.isfinite(cim_norms) & (cim_norms > 0)]
    if cim_norms.size == 0:
        return None

    subB = B_sorted[:X]
    Gf, max_abs = gram_from_basis_rows_float_and_scale(subB)
    # MC uniform
    qf_mc = uniform_spins_qf_samples(Gf, n_samples=mc_samples, batch=mc_batch, rng=rng)
    qf_mc_sorted = np.sort(qf_mc)
    min_mc_qf = float(qf_mc_sorted[0])
    max_mc_qf = float(qf_mc_sorted[-1])
    min_mc_norm = float(Decimal(min_mc_qf * max_abs).sqrt())
    max_mc_norm = float(Decimal(max_mc_qf * max_abs).sqrt())

    # CIM -> percentiles vs uniform
    qf_cim = (cim_norms.astype(np.float64)**2) / float(max_abs)
    ranks = np.searchsorted(qf_mc_sorted, qf_cim, side='right')
    cim_percentiles = ranks / float(mc_samples)
    best_cim_norm = float(np.min(cim_norms))
    mean_cim_norm = float(np.mean(cim_norms))
    best_percentile = float(np.searchsorted(qf_mc_sorted, (best_cim_norm**2)/float(max_abs), side='right') / float(mc_samples))
    mean_percentile = float(np.searchsorted(qf_mc_sorted, (mean_cim_norm**2)/float(max_abs), side='right') / float(mc_samples))

    quantiles = {p: float(np.percentile(cim_norms, p)) for p in [10,25,50,75,90]}
    within = {k: float(np.mean(cim_norms <= k*min_mc_norm)) for k in k_factors}

    # Mode and peak
    try:
        bins_m = np.logspace(np.log10(np.min(cim_norms)), np.log10(np.max(cim_norms)), 128)
        counts_m, edges_m = np.histogram(cim_norms, bins=bins_m)
        idx_m = int(np.argmax(counts_m))
        modal_norm = float(np.sqrt(edges_m[idx_m]*edges_m[idx_m+1]))
        peak_prob = float(counts_m[idx_m] / float(cim_norms.size))
    except Exception:
        modal_norm, peak_prob = float('nan'), float('nan')

    # Moments on log(norm)
    ln = np.log(cim_norms)
    skew = float(((ln - ln.mean())**3).mean() / (ln.std()**3 + 1e-12))
    kurt = float(((ln - ln.mean())**4).mean() / (ln.std()**4 + 1e-12)) - 3.0

    # Histogram (probability) PNG
    try:
        out_dir = folders['cim_hist_png']
        os.makedirs(out_dir, exist_ok=True)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        bins = np.logspace(np.log10(np.min(cim_norms)), np.log10(np.max(cim_norms)), 200)
        counts, edges = np.histogram(cim_norms, bins=bins)
        probs = counts / float(cim_norms.size)
        ax.step(edges[:-1], probs, where='post', color='royalblue', label='CIM histogram (prob.)')
        set_norm_log_axis(ax, 'x')
        ax.set_xlabel("Norm (log)")
        ax.set_ylabel("Probability")
        ax.set_title(f"CIM Output Norm Distribution (X={X})")
        annotate_definition(ax)
        ax.legend()
        save_png(fig, os.path.join(out_dir, f"cim_hist_X{X}.png"))
    except Exception:
        pass

    # CDF PNG
    try:
        out_dir = folders['cim_cdf_png']
        os.makedirs(out_dir, exist_ok=True)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        cim_sorted = np.sort(cim_norms)
        y = np.arange(1, cim_sorted.size+1)/float(cim_sorted.size)
        ax.plot(cim_sorted, y, color='darkgreen', label='CIM empirical CDF')
        set_norm_log_axis(ax, 'x')
        ax.set_xlabel("Norm (log)")
        ax.set_ylabel("Cumulative probability")
        ax.set_title(f"CIM Output Norm CDF (X={X})")
        annotate_definition(ax)
        ax.legend()
        save_png(fig, os.path.join(out_dir, f"cim_cdf_X{X}.png"))
    except Exception:
        pass

    # Tail CDF (0-10%)
    try:
        out_dir = folders['tail_cdf_png']
        os.makedirs(out_dir, exist_ok=True)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        cim_sorted = np.sort(cim_norms)
        y = np.arange(1, cim_sorted.size+1)/float(cim_sorted.size)
        mask = y <= 0.10
        ax.plot(cim_sorted[mask], y[mask], color='firebrick', label='CIM tail CDF (<=10%)')
        set_norm_log_axis(ax, 'x')
        ax.set_xlabel("Norm (log)")
        ax.set_ylabel("Cumulative probability")
        ax.set_title(f"CIM Tail CDF (X={X}, first 10%)")
        annotate_definition(ax)
        ax.legend()
        save_png(fig, os.path.join(out_dir, f"tail_cdf_X{X}.png"))
    except Exception:
        pass

    # Percentile histogram vs uniform
    try:
        out_dir = folders['percentile_hist_png']
        os.makedirs(out_dir, exist_ok=True)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        bins_p = np.linspace(0,1,51)
        ax.hist(cim_percentiles, bins=bins_p, color='slateblue', alpha=0.9, density=True)
        ax.set_xlabel("Percentile vs uniform-over-spins (0=best)")
        ax.set_ylabel("Density")
        ax.set_title(f"CIM Percentile Distribution vs Uniform (X={X})")
        ax.grid(True, alpha=0.3)
        save_png(fig, os.path.join(out_dir, f"percentiles_X{X}.png"))
    except Exception:
        pass

    # Bands stacked area (relative to min MC)
    try:
        out_dir = folders['bands_png']
        os.makedirs(out_dir, exist_ok=True)
        bands = [1.0, 1.1, 1.5, 2.0]
        probs_b = []
        prev_thr = 0.0
        for t in bands:
            thr = t * min_mc_norm
            probs_b.append(float(np.mean((cim_norms > prev_thr) & (cim_norms <= thr))))
            prev_thr = thr
        probs_b.append(float(np.mean(cim_norms > bands[-1]*min_mc_norm)))
        labels = [f"[{1.0}x,{1.1}x)" , "[1.1x,1.5x)", "[1.5x,2.0x)", f"[≥{2.0}x)"]
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.bar([X-0.4+0.2*i for i in range(len(probs_b))], probs_b, width=0.18,
               label=labels, color=['#1b9e77','#d95f02','#7570b3','#e7298a'])
        ax.set_xlabel("X (linear)")
        ax.set_ylabel("Probability")
        ax.set_title(f"Mass in Norm Bands (X={X}, rel. to MC-min)")
        ax.set_xlim(X-0.6, X+0.6)
        save_png(fig, os.path.join(out_dir, f"bands_X{X}.png"))
    except Exception:
        pass

    # Save per-X stats for aggregation and other plots
    rec = dict(
        X=X,
        valid_outputs=int(cim_norms.size),
        min_mc_norm=min_mc_norm,
        max_mc_norm=max_mc_norm,
        best_cim_norm=best_cim_norm,
        mean_cim_norm=mean_cim_norm,
        best_percentile_uniform=best_percentile,
        mean_percentile_uniform_of_mean=mean_percentile,
        quantiles=quantiles,
        within=within,
        modal_norm=modal_norm,
        peak_prob=peak_prob,
        log_skew=skew,
        log_kurt=kurt,
        cim_norms=cim_norms,          # keep for some plots (beware memory)
        cim_percentiles=cim_percentiles,
        qf_mc=qf_mc,                  # for JSD/AUC comparisons
        max_abs=max_abs,
        Gf=Gf,                        # for heatmaps grid building
    )
    # Add throughput if available
    if summary_row:
        try:
            sim_time_sec = float(summary_row.get("sim_time_sec", "nan"))
            if sim_time_sec > 0:
                rec["runs_per_sec"] = rec["valid_outputs"] / sim_time_sec
                rec["sim_time_sec"] = sim_time_sec
            else:
                rec["runs_per_sec"] = float('nan')
        except:
            rec["runs_per_sec"] = float('nan')
    else:
        rec["runs_per_sec"] = float('nan')
    return rec

# ----------------- Aggregate plots -----------------
def make_violins(records, out_dir):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    data = [r['cim_norms'] for r in records]
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    parts = ax.violinplot(data, positions=Xs, showmeans=True, widths=0.8)
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("Norm (log)")
    set_norm_log_axis(ax, 'y')
    ax.set_title("Violin plots of CIM Norms per X")
    save_png(fig, os.path.join(out_dir, "violins.png"))

def make_tail_cdf_grid(records, out_dir):
    ensure_dir(out_dir)
    for r in records:
        X = r['X']
        cim = np.sort(r['cim_norms'])
        y = np.arange(1, cim.size+1)/float(cim.size)
        mask = y <= 0.10
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.plot(cim[mask], y[mask], color='firebrick')
        set_norm_log_axis(ax, 'x')
        ax.set_xlabel("Norm (log)")
        ax.set_ylabel("Cumulative probability")
        ax.set_title(f"CIM Tail CDF (<=10%) X={X}")
        save_png(fig, os.path.join(out_dir, f"tail_cdf_X{X}.png"))

def make_tce(records, out_dir):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    ps = [1,5,10]
    colors = ['C0','C1','C2']
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    for p, c in zip(ps, colors):
        ys = []
        for r in records:
            thr = np.percentile(r['cim_norms'], p)
            vals = r['cim_norms'][r['cim_norms'] <= thr]
            ys.append(float(np.mean(vals)) if vals.size>0 else float('nan'))
        ax.plot(Xs, ys, 'o-', color=c, label=f"E[norm | ≤ {p}th%]")
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("Norm (log)")
    set_norm_log_axis(ax, 'y')
    ax.set_title("Conditional Tail Expectation vs X")
    ax.legend()
    save_png(fig, os.path.join(out_dir, "tce_vs_X.png"))

def make_moments(records, out_dir):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    skew = [r['log_skew'] for r in records]
    kurt = [r['log_kurt'] for r in records]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(Xs, skew, 'o-', label='Skewness(log norm)')
    ax.plot(Xs, kurt, 'o-', label='Excess kurtosis(log norm)')
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("Value")
    ax.set_title("Shape Statistics vs X")
    ax.legend()
    save_png(fig, os.path.join(out_dir, "moments_vs_X.png"))

def make_bottom_p(records, out_dir):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    ps = [0.01,0.05,0.10,0.20]
    colors = ['C0','C1','C2','C3']
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    for p,c in zip(ps, colors):
        ys = [float(np.mean(r['cim_percentiles'] <= p)) for r in records]
        ax.plot(Xs, ys, 'o-', label=f"P(percentile ≤ {int(p*100)}%)")
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("Probability")
    ax.set_title("Bottom-p Percentile Hit Rate vs X (uniform reference)")
    ax.legend()
    save_png(fig, os.path.join(out_dir, "bottom_p_vs_X.png"))

def js_divergence(p, q):
    m = 0.5*(p+q)
    def kl(a,b):
        mask = (a>0) & (b>0)
        return np.sum(a[mask]*np.log(a[mask]/b[mask]))
    return 0.5*kl(p,m) + 0.5*kl(q,m)

def make_jsd(records, out_dir, nbins=120):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    # Common log bins across all X
    all_norms = np.concatenate([r['cim_norms'] for r in records])
    lo = float(np.min(all_norms))
    hi = float(np.max(all_norms))
    bins = np.logspace(np.log10(lo), np.log10(hi), nbins)
    ys = []
    for r in records:
        cim = r['cim_norms']
        p, _ = np.histogram(cim, bins=bins, density=True)
        # Build uniform-over-spins via MC samples transformed to norm scale
        qf = r['qf_mc']*r['max_abs']
        q_norm = np.sqrt(np.maximum(qf,0.0))
        q, _ = np.histogram(q_norm, bins=bins, density=True)
        eps = 1e-12
        p = (p+eps); p = p/np.sum(p)
        q = (q+eps); q = q/np.sum(q)
        ys.append(js_divergence(p,q))
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(Xs, ys, 'o-')
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("Jensen–Shannon divergence")
    ax.set_title("JSD between CIM and Uniform-over-spins (per X)")
    save_png(fig, os.path.join(out_dir, "jsd_vs_X.png"))

def make_auc_gap(records, out_dir, nbins=240):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    # Common log bins
    all_norms = np.concatenate([r['cim_norms'] for r in records])
    lo = float(np.min(all_norms))
    hi = float(np.max(all_norms))
    bins = np.logspace(np.log10(lo), np.log10(hi), nbins)
    centers = np.sqrt(bins[:-1]*bins[1:])
    dlog = np.log(bins[1:]) - np.log(bins[:-1])
    ys = []
    for r in records:
        c = np.sort(r['cim_norms'])
        F_cim = np.searchsorted(c, centers, side='right')/float(c.size)
        qf = r['qf_mc']*r['max_abs']
        q_norm = np.sqrt(np.maximum(qf,0.0))
        q_norm_sorted = np.sort(q_norm)
        F_uni = np.searchsorted(q_norm_sorted, centers, side='right')/float(q_norm_sorted.size)
        gap = np.sum((F_cim - F_uni)*dlog)
        ys.append(gap)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(Xs, ys, 'o-')
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("AUC gap over log(norm)")
    ax.set_title("Integral CDF Advantage (CIM − Uniform)")
    save_png(fig, os.path.join(out_dir, "auc_gap_vs_X.png"))

def make_norm_efficiency(records, out_dir):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    y_best = []
    y_mean = []
    for r in records:
        q_norm = np.sqrt(np.maximum(r['qf_mc']*r['max_abs'],0.0))
        med_mc = float(np.median(q_norm))
        min_mc = r['min_mc_norm']
        denom = max(med_mc - min_mc, 1e-12)
        y_best.append((r['best_cim_norm'] - min_mc)/denom)
        y_mean.append((r['mean_cim_norm'] - min_mc)/denom)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(Xs, y_best, 'o-', label='Best (normalized)')
    ax.plot(Xs, y_mean, 'o-', label='Mean (normalized)')
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("(norm−MCmin)/(MCmedian−MCmin)")
    ax.set_title("Dimensionless Efficiency vs X")
    ax.legend()
    save_png(fig, os.path.join(out_dir, "norm_efficiency_vs_X.png"))

def make_heatmap_cim(records, out_dir, nbins=120):
    ensure_dir(out_dir)
    # Common log bins
    all_norms = np.concatenate([r['cim_norms'] for r in records])
    lo = float(np.min(all_norms)); hi = float(np.max(all_norms))
    bins = np.logspace(np.log10(lo), np.log10(hi), nbins)
    H = []
    for r in records:
        h, _ = np.histogram(r['cim_norms'], bins=bins, density=True)
        H.append(h)
    H = np.array(H)  # shape (numX, nbins-1)
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    im = ax.imshow(H.T, aspect='auto', origin='lower',
                   extent=[records[0]['X'], records[-1]['X'], np.log10(bins[0]), np.log10(bins[-1])],
                   cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Density")
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("log10(Norm)")
    ax.set_title("Heatmap: CIM Density over X and Norm")
    save_png(fig, os.path.join(out_dir, "heatmap_cim.png"))

def make_heatmap_percentile(records, out_dir, nbins=50):
    ensure_dir(out_dir)
    bins = np.linspace(0,1,nbins+1)
    H = []
    for r in records:
        h, _ = np.histogram(r['cim_percentiles'], bins=bins, density=True)
        H.append(h)
    H = np.array(H)  # (numX, nbins)
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    im = ax.imshow(H.T, aspect='auto', origin='lower',
                   extent=[records[0]['X'], records[-1]['X'], 0, 1],
                   cmap='magma')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Density")
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("Percentile vs uniform (0..1)")
    ax.set_title("Heatmap: CIM Percentile Density over X")
    save_png(fig, os.path.join(out_dir, "heatmap_percentile.png"))

def make_ridgeline(records, out_dir, nbins=100, offset=1.0):
    ensure_dir(out_dir)
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    y_off = 0.0
    for r in records:
        bins = np.logspace(np.log10(np.min(r['cim_norms'])), np.log10(np.max(r['cim_norms'])), nbins)
        h, edges = np.histogram(r['cim_norms'], bins=bins, density=True)
        x = np.sqrt(edges[:-1]*edges[1:])
        ax.plot(x, h + y_off, label=f"X={r['X']}")
        y_off += offset
    set_norm_log_axis(ax, 'x')
    ax.set_xlabel("Norm (log)")
    ax.set_ylabel("Density + offset")
    ax.set_title("Ridgeline plot of CIM densities by X")
    save_png(fig, os.path.join(out_dir, "ridgeline.png"))

def make_surface3d(records, out_dir, nbins=100):
    ensure_dir(out_dir)
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    Xs = np.array([r['X'] for r in records], dtype=float)
    all_norms = np.concatenate([r['cim_norms'] for r in records])
    lo = float(np.min(all_norms)); hi = float(np.max(all_norms))
    bins = np.logspace(np.log10(lo), np.log10(hi), nbins)
    Y = np.sqrt(bins[:-1]*bins[1:])
    Z = []
    for r in records:
        h, _ = np.histogram(r['cim_norms'], bins=bins, density=True)
        Z.append(h)
    Z = np.array(Z).T  # shape (nbins-1, numX)
    Xg, Yg = np.meshgrid(Xs, Y)
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xg, np.log10(Yg), Z, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("log10(Norm)")
    ax.set_zlabel("Density")
    ax.set_title("3D Surface: X vs log(Norm) vs Density")
    save_png(fig, os.path.join(out_dir, "surface3d.png"))

def make_convergence_curves(records, out_dir, downsample=200):
    ensure_dir(out_dir)
    for r in records:
        X = r['X']
        norms = r['cim_norms']
        best = np.minimum.accumulate(norms)
        if best.size > downsample:
            idx = np.linspace(0, best.size-1, downsample).astype(int)
            x = idx+1
            y = best[idx]
        else:
            x = np.arange(1, best.size+1)
            y = best
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.plot(x, y, color='C0')
        ax.set_xlabel("Runs (linear)")
        ax.set_ylabel("Best-so-far norm (log)")
        set_norm_log_axis(ax, 'y')
        ax.set_title(f"Convergence curve (X={X})")
        save_png(fig, os.path.join(out_dir, f"convergence_X{X}.png"))

def make_convergence_overlay(records, out_dir, delta=0.001, window=5000):
    ensure_dir(out_dir)
    Xs, t_plateau, best_norms = [], [], []
    for r in records:
        best = np.minimum.accumulate(r['cim_norms'])
        plateau_idx = len(best)-1
        for i in range(max(0, len(best)-window-1)):
            j = min(i+window, len(best)-1)
            if best[i] <= 0:
                continue
            if (best[i] - best[j])/best[i] <= delta:
                plateau_idx = i
                break
        # convert to seconds if runs/sec known
        if 'runs_per_sec' in r and isinstance(r['runs_per_sec'], (int,float)) and r['runs_per_sec']>0:
            t_sec = plateau_idx / r['runs_per_sec']
        else:
            t_sec = float('nan')
        Xs.append(r['X'])
        t_plateau.append(t_sec)
        best_norms.append(float(np.min(r['cim_norms'])))
    Xs = np.array(Xs, dtype=int)
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(Xs, t_plateau, 'o-', color='C0', label='Convergence time (s)')
    ax1.set_xlabel("X (linear)")
    ax1.set_ylabel("Convergence time (s)")
    ax2 = ax1.twinx()
    ax2.plot(Xs, best_norms, 'o-', color='C1', label='Best norm')
    set_norm_log_axis(ax2, 'y')
    ax2.set_ylabel("Best norm (log)")
    fig.suptitle("Convergence time and Best Norm vs X")
    save_png(fig, os.path.join(out_dir, "convergence_overlay.png"))

def make_throughput(records, out_dir):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    y = [r.get('runs_per_sec', float('nan')) for r in records]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(Xs, y, 'o-')
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("Runs/sec")
    ax.set_title("Throughput vs X")
    save_png(fig, os.path.join(out_dir, "throughput_vs_X.png"))

def make_mode(records, out_dir):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    modal = [r['modal_norm'] for r in records]
    peak = [r['peak_prob'] for r in records]
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(Xs, modal, 'o-', color='C0', label='Modal norm')
    ax1.set_xlabel("X (linear)")
    ax1.set_ylabel("Modal norm (log)")
    set_norm_log_axis(ax1, 'y')
    ax2 = ax1.twinx()
    ax2.plot(Xs, peak, 'o-', color='C1', label='Peak prob.')
    ax2.set_ylabel("Peak probability")
    fig.suptitle("Modal Norm and Peak Probability vs X")
    save_png(fig, os.path.join(out_dir, "mode_peak_vs_X.png"))

def make_best_mean(records, out_dir):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    bests = [r['best_cim_norm'] for r in records]
    means = [r['mean_cim_norm'] for r in records]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(Xs, bests, 'o-', label='Best CIM norm')
    ax.plot(Xs, means, 'o-', label='Mean CIM norm')
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("Norm (log)")
    set_norm_log_axis(ax, 'y')
    ax.set_title("CIM Best and Mean Norms vs X")
    ax.legend()
    save_png(fig, os.path.join(out_dir, "cim_best_mean_vs_X.png"))

def make_percentiles_vs_X(records, out_dir):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    y_best = [r['best_percentile_uniform'] for r in records]
    y_mean = [r['mean_percentile_uniform_of_mean'] for r in records]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(Xs, y_best, 'o-', label='Best CIM norm percentile vs uniform')
    ax.plot(Xs, y_mean, 'o-', label='Percentile of mean CIM norm vs uniform')
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("Percentile (0..1)")
    ax.set_title("CIM Percentiles vs Uniform-over-spins")
    ax.legend()
    save_png(fig, os.path.join(out_dir, "percentiles_vs_X.png"))

def make_mc_minmax(records, out_dir):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    min_mc = [r['min_mc_norm'] for r in records]
    max_mc = [r['max_mc_norm'] for r in records]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(Xs, min_mc, 'o-', label='MC min norm')
    ax.plot(Xs, max_mc, 'o-', label='MC max norm')
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("Norm (log)")
    set_norm_log_axis(ax, 'y')
    ax.set_title("Uniform-over-spins Min/Max Norm Estimates vs X")
    ax.legend()
    save_png(fig, os.path.join(out_dir, "mc_minmax_vs_X.png"))

def make_gap_min(records, out_dir):
    ensure_dir(out_dir)
    Xs = np.array([r['X'] for r in records], dtype=int)
    gaps = [max(r['best_cim_norm'] - r['min_mc_norm'], 0.0) for r in records]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(Xs, gaps, 'o-')
    ax.set_xlabel("X (linear)")
    ax.set_ylabel("Gap to MC-min (log)")
    set_norm_log_axis(ax, 'y')
    ax.set_title("Best Norm Gap to MC Min vs X")
    save_png(fig, os.path.join(out_dir, "gap_to_min_vs_X.png"))

def make_degeneracy(records, B_sorted, out_dir, x_enum_max):
    ensure_dir(out_dir)
    for r in records:
        X = r['X']
        if X > x_enum_max:
            continue
        try:
            G_int = gram_from_basis_rows_int(B_sorted[:X])
            deg_hist = enumerate_degeneracy_counts(G_int, x_enum_max=x_enum_max)
            if deg_hist is None:
                continue
            xs = np.array(sorted(deg_hist.keys()), dtype=np.int64)
            ys = np.array([deg_hist[c] for c in xs], dtype=np.int64)
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(111)
            ax.bar(xs, ys, width=np.maximum(1, xs//100), color='teal', alpha=0.85)
            ax.set_xlabel("Degeneracy count (#spin configs with identical norm)")
            ax.set_ylabel("#distinct norms")
            ax.set_title(f"Degeneracy Histogram (X={X}, exact)")
            save_png(fig, os.path.join(out_dir, f"degeneracy_X{X}.png"))
        except Exception:
            continue

def make_hpo_plots(data_dir, records, out_steps_dir, out_scatter_dir, out_response_dir):
    ensure_dir(out_steps_dir)
    ensure_dir(out_scatter_dir)
    ensure_dir(out_response_dir)
    xs, chosen_steps, best_norms = [], [], []
    all_steps, all_best = [], []
    all_r, all_mu, all_noise = [], [], []
    for r in records:
        X = r['X']
        trials, best = parse_hpo_file_if_exists(data_dir, X)
        if best is not None and isinstance(best.get("steps", None), int):
            xs.append(X); chosen_steps.append(best["steps"]); best_norms.append(r['best_cim_norm'])
        # scatter pools
        for t in trials:
            if isinstance(t.get("steps", None), int) and isinstance(t.get("best_sq_norm", None), (int,float)):
                all_steps.append(t["steps"])
                all_best.append(float(t["best_sq_norm"])**0.5)  # sqrt to get a norm-like scale
            if "cac_r" in t: all_r.append([X, t["cac_r"], float(t["best_sq_norm"])**0.5])
            if "cac_mu" in t: all_mu.append([X, t["cac_mu"], float(t["best_sq_norm"])**0.5])
            if "cac_noise" in t: all_noise.append([X, t["cac_noise"], float(t["best_sq_norm"])**0.5])
    # chosen steps vs X
    if xs:
        xs = np.array(xs, dtype=int)
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.plot(xs, chosen_steps, 'o-', color='C0', label='Chosen steps')
        ax1.set_xlabel("X (linear)")
        ax1.set_ylabel("Steps")
        ax2 = ax1.twinx()
        ax2.plot(xs, best_norms, 'o-', color='C1', label='Best CIM norm')
        set_norm_log_axis(ax2, 'y')
        ax2.set_ylabel("Best norm (log)")
        fig.suptitle("HPO Chosen Steps and Best Norm vs X")
        save_png(fig, os.path.join(out_steps_dir, "hpo_steps_vs_X.png"))
    # scatter steps vs best
    if all_steps:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.scatter(all_steps, all_best, s=10, alpha=0.5)
        ax.set_xlabel("steps (trial)")
        ax.set_ylabel("trial best norm (sqrt(best_sq_norm)) (log)")
        set_norm_log_axis(ax, 'y')
        ax.set_title("HPO steps vs trial best norm (all X pooled)")
        save_png(fig, os.path.join(out_scatter_dir, "hpo_scatter_steps.png"))
    # partial response maps (if sufficient points, coarse grid)
    # Skipping 2D interpolation to keep fast and robust.

def aggregate_core(records, analysis_root, data_dir, B_sorted, x_enum_max):
    # Core plots requested as most useful by user:
    make_best_mean(records, ensure_dir(os.path.join(analysis_root, "cim_best_mean_vs_X_png")))
    make_percentiles_vs_X(records, ensure_dir(os.path.join(analysis_root, "percentiles_vs_X_png")))
    # Per-X hist and CDF already saved inside per_x_analysis in their folders.

    # New plots:
    make_violins(records, ensure_dir(os.path.join(analysis_root, "violins_png")))
    make_tail_cdf_grid(records, ensure_dir(os.path.join(analysis_root, "tail_cdf_png")))
    make_tce(records, ensure_dir(os.path.join(analysis_root, "tce_png")))
    make_moments(records, ensure_dir(os.path.join(analysis_root, "moments_png")))
    make_bottom_p(records, ensure_dir(os.path.join(analysis_root, "bottom_p_png")))
    make_jsd(records, ensure_dir(os.path.join(analysis_root, "jsd_png")))
    make_auc_gap(records, ensure_dir(os.path.join(analysis_root, "auc_gap_png")))
    make_norm_efficiency(records, ensure_dir(os.path.join(analysis_root, "norm_efficiency_png")))
    make_heatmap_cim(records, ensure_dir(os.path.join(analysis_root, "heatmap_cim_png")))
    make_heatmap_percentile(records, ensure_dir(os.path.join(analysis_root, "heatmap_percentile_png")))
    make_ridgeline(records, ensure_dir(os.path.join(analysis_root, "ridgeline_png")))
    make_surface3d(records, ensure_dir(os.path.join(analysis_root, "surface3d_png")))
    make_convergence_curves(records, ensure_dir(os.path.join(analysis_root, "convergence_curves_png")))
    make_convergence_overlay(records, ensure_dir(os.path.join(analysis_root, "convergence_overlay_png")))
    make_throughput(records, ensure_dir(os.path.join(analysis_root, "throughput_png")))
    make_mode(records, ensure_dir(os.path.join(analysis_root, "mode_png")))
    make_mc_minmax(records, ensure_dir(os.path.join(analysis_root, "mc_minmax_vs_X_png")))
    make_gap_min(records, ensure_dir(os.path.join(analysis_root, "gap_min_png")))
    make_degeneracy(records, B_sorted, ensure_dir(os.path.join(analysis_root, "degeneracy_png")), x_enum_max)
    make_hpo_plots(data_dir, records,
                   ensure_dir(os.path.join(analysis_root, "hpo_steps_png")),
                   ensure_dir(os.path.join(analysis_root, "hpo_scatter_png")),
                   ensure_dir(os.path.join(analysis_root, "hpo_response_png")))

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze CIM results: rich percentile/efficiency plots grouped in PNG folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--results_dir", type=str, default=None, help="Path to a specific results folder (the one containing 'data'). If not set, picks latest.")
    parser.add_argument("--dim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mc_samples", type=int, default=30000, help="Uniform-over-spins Monte Carlo samples per X.")
    parser.add_argument("--mc_batch", type=int, default=4096, help="Batch size for MC matrix multiplies.")
    parser.add_argument("--x_enum_max", type=int, default=18, help="Exact degeneracy enumeration for X ≤ this; else skipped.")
    parser.add_argument("--k_factors", type=str, default="1.0,1.1,1.5,2.0", help="Comma-separated list of factors for 'within factor of min' curves.")
    args = parser.parse_args()

    # Resolve results_dir
    if args.results_dir is None:
        res_dir = latest_results_dir()
        print(f"Using latest results directory: {res_dir}")
    else:
        res_dir = args.results_dir
    data_dir = os.path.join(res_dir, "data")
    if not os.path.isdir(data_dir):
        print(f"ERROR: data folder not found at {data_dir}")
        sys.exit(1)
    analysis_root = ensure_dir(os.path.join(res_dir, "analysis"))

    # Load basis and sort rows by squared norm (same selection rule as run script)
    try:
        B = load_local_lattice(args.dim, args.seed)
    except Exception as e:
        print(f"ERROR loading lattice: {e}")
        sys.exit(1)
    B_sorted, _ = basis_sorted_by_row_norm(B)

    # Summary (optional; for sim_time / runs_per_sec)
    summary = load_summary_if_exists(data_dir)

    # Parse k_factors
    try:
        k_factors = tuple(float(x.strip()) for x in args.k_factors.split(",") if x.strip())
        if not k_factors:
            k_factors = (1.0, 1.1, 1.5, 2.0)
    except Exception:
        k_factors = (1.0, 1.1, 1.5, 2.0)

    # Discover X values with norms files
    available_X = []
    for X in range(2, args.dim + 1):
        if find_norms_file(data_dir, X):
            available_X.append(X)
    if not available_X:
        print("No norms_X*_runs*.npy files found; nothing to analyze.")
        sys.exit(0)
    print(f"Found norms for X = {available_X}")

    # Create folders for per-X PNG outputs
    folders = {
        'cim_hist_png': ensure_dir(os.path.join(analysis_root, "cim_hist_png")),
        'cim_cdf_png': ensure_dir(os.path.join(analysis_root, "cim_cdf_png")),
        'tail_cdf_png': ensure_dir(os.path.join(analysis_root, "tail_cdf_png")),
        'percentile_hist_png': ensure_dir(os.path.join(analysis_root, "percentile_hist_png")),
        'bands_png': ensure_dir(os.path.join(analysis_root, "bands_png")),
    }

    rng = np.random.RandomState(12345)
    records = []
    for X in available_X:
        try:
            rec = per_x_analysis(
                X=X,
                B_sorted=B_sorted,
                data_dir=data_dir,
                analysis_root=analysis_root,
                folders=folders,
                mc_samples=args.mc_samples,
                mc_batch=args.mc_batch,
                x_enum_max=args.x_enum_max,
                k_factors=k_factors,
                rng=rng,
                summary_row=summary.get(X, {})
            )
            if rec is not None:
                records.append(rec)
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except Exception as e:
            print(f"[X={X}] ERROR during per-X analysis: {e}")

    # Aggregate CSV
    if records:
        agg_csv = os.path.join(analysis_root, "aggregate_summary_extended.csv")
        with open(agg_csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            hdr = ["X","valid_outputs","min_mc_norm","max_mc_norm","best_cim_norm","mean_cim_norm",
                   "best_percentile_uniform","mean_percentile_uniform_of_mean",
                   "modal_norm","peak_prob","log_skew","log_kurt"]
            qps = [10,25,50,75,90]
            hdr += [f"quantile_{p}" for p in qps]
            for k in k_factors:
                hdr.append(f"within_{k}x_min_mc")
            wr.writerow(hdr)
            for r in sorted(records, key=lambda x: x['X']):
                row = [r['X'], r['valid_outputs'], r['min_mc_norm'], r['max_mc_norm'],
                       r['best_cim_norm'], r['mean_cim_norm'],
                       r['best_percentile_uniform'], r['mean_percentile_uniform_of_mean'],
                       r['modal_norm'], r['peak_prob'], r['log_skew'], r['log_kurt']]
                row += [r['quantiles'][p] for p in qps]
                row += [r['within'][k] for k in k_factors]
                wr.writerow(row)
        print(f"Aggregate summary written to: {agg_csv}")

        # Aggregate plots (PNG-only, grouped folders)
        aggregate_core(records, analysis_root, data_dir, B_sorted, args.x_enum_max)
        print(f"Aggregate figures saved under: {analysis_root}")
    else:
        print("No per-X records produced; skipping aggregates.")

if __name__ == "__main__":
    main()
