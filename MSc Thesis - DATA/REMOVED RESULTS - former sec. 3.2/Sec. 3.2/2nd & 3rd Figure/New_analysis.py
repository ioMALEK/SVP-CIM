#!/usr/bin/env python3
"""
new_analysis.py

Generates a large set of additional across-X analyses (PNG only) using prior CIM run data.
Outputs go to <results_dir>/analysis/2nd_analysis/<topic>.

What it uses:
- norms_X*_runs*.npy per X in <results_dir>/data
- hyperparams_X*.txt per X (optional)
- CIM_X_scan_summary.txt (optional)
- Lattice B from svp_lattices/dim{dim}_seed{seed}.txt

Conventions:
- Any axis with "norm" is log-scaled.
- Any axis with "X" is linear.
- Percentiles are computed w.r.t. the uniform-over-spins distribution on the same X-subspace.

Run example:
  python3 new_analysis.py --dim 50 --seed 17 --results_dir CIM_performance_worst_case_results/2025-08-10_00-23-57 --mc_samples 30000
"""

import os
import re
import ast
import csv
import sys
import glob
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
getcontext().prec = 100

# ------------- Basic IO -------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def latest_results_dir(parent="CIM_performance_worst_case_results"):
    subs = [p for p in glob.glob(os.path.join(parent, "*")) if os.path.isdir(p)]
    if not subs:
        raise FileNotFoundError("No results directories found.")
    subs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subs[0]

def load_local_lattice(dim, seed, folder="svp_lattices"):
    path = os.path.join(folder, f"dim{dim}_seed{seed}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing lattice: {path}")
    with open(path, "r") as f:
        text = f.read()
    try:
        B = ast.literal_eval(text)
        B = np.array(B, dtype=object)
    except Exception:
        rows=[]
        for line in text.strip().splitlines():
            nums = re.findall(r"-?\d+", line)
            if nums: rows.append(list(map(int, nums)))
        B = np.array(rows, dtype=object)
    if B.ndim!=2 or B.shape[0]!=B.shape[1]:
        raise ValueError(f"Basis must be square, got {B.shape}")
    return B

def vec_sq_norm_int(vec):
    return int(sum(int(x)*int(x) for x in vec))

def basis_sorted_by_row_norm(B):
    sq = np.array([vec_sq_norm_int(B[i]) for i in range(B.shape[0])], dtype=object)
    order = np.argsort(np.array([int(x) for x in sq]))
    return B[order], order

def find_norms_file(data_dir, X):
    files = glob.glob(os.path.join(data_dir, f"norms_X{X}_runs*.npy"))
    if not files: return None
    files.sort(key=lambda p:(os.path.getsize(p), p))
    return files[-1]

def load_summary(data_dir):
    path = os.path.join(data_dir, "CIM_X_scan_summary.txt")
    if not os.path.isfile(path): return {}
    out={}
    with open(path,"r") as f:
        header=[h.strip() for h in f.readline().strip().split(",")]
        for line in f:
            parts=[p.strip() for p in line.strip().split(",")]
            if not parts: continue
            try:
                X=int(parts[0]); row=dict(zip(header,parts))
                out[X]=row
            except: pass
    return out

def parse_hpo(data_dir, X):
    path = os.path.join(data_dir, f"hyperparams_X{X}.txt")
    if not os.path.isfile(path): return []
    out=[]
    with open(path,"r") as f:
        header=[h.strip() for h in f.readline().strip().split(",")]
        for line in f:
            parts=[p.strip() for p in line.strip().split(",")]
            if len(parts)!=len(header): continue
            rec=dict(zip(header,parts))
            # numeric coercion
            for k in rec:
                try:
                    if k in ["trial","steps","batch_runs"]: rec[k]=int(float(rec[k]))
                    else: rec[k]=float(rec[k])
                except: pass
            out.append(rec)
    return out

# ------------- Gram helpers -------------
def gram_int(B):
    K=B.shape[0]
    G=np.empty((K,K),dtype=object)
    for i in range(K):
        bi=B[i]
        for j in range(i,K):
            bj=B[j]
            s=0
            for t in range(bi.shape[0]):
                s+=int(bi[t])*int(bj[t])
            G[i,j]=s; G[j,i]=s
    return G

def gram_float_and_scale(B):
    G=gram_int(B)
    K=G.shape[0]
    max_abs=max(1, max(abs(int(G[i,j])) for i in range(K) for j in range(K)))
    Gf=np.empty((K,K),dtype=np.float64)
    for i in range(K):
        for j in range(K):
            Gf[i,j]=float(int(G[i,j])/max_abs)
    return Gf, max_abs

# ------------- MC reference -------------
def mc_uniform_qf(Gf, n_samples=30000, batch=4096, rng=None):
    if rng is None: rng=np.random.RandomState(0)
    K=Gf.shape[0]
    out=np.empty(n_samples,dtype=np.float64)
    ptr=0
    while ptr<n_samples:
        m=min(batch, n_samples-ptr)
        S=rng.choice([-1.0,1.0],size=(m,K)).astype(np.float64,copy=False)
        SG=S@Gf
        out[ptr:ptr+m]=np.einsum("ij,ij->i",SG,S,dtype=np.float64)
        ptr+=m
    return out

# ------------- Plot utils -------------
def save_png(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def set_log(ax, axis):
    if axis=='x': ax.set_xscale('log')
    else: ax.set_yscale('log')

# ------------- Build records -------------
def build_records(res_dir, dim, seed, mc_samples, mc_batch):
    data_dir=os.path.join(res_dir,"data")
    if not os.path.isdir(data_dir): raise FileNotFoundError(f"No data at {data_dir}")
    B=load_local_lattice(dim,seed)
    B_sorted,_=basis_sorted_by_row_norm(B)
    summary=load_summary(data_dir)
    rng=np.random.RandomState(12345)

    Xs=[]
    recs=[]
    for X in range(2,dim+1):
        nf=find_norms_file(data_dir,X)
        if nf is None: continue
        norms=np.load(nf, mmap_mode='r')
        norms=np.asarray(norms)
        norms=norms[np.isfinite(norms) & (norms>0)]
        if norms.size==0: continue
        subB=B_sorted[:X]
        Gf,scale=gram_float_and_scale(subB)
        qf_mc=mc_uniform_qf(Gf, n_samples=mc_samples, batch=mc_batch, rng=rng)
        qf_mc_sorted=np.sort(qf_mc)
        min_mc=float(Decimal(np.min(qf_mc)*scale).sqrt())
        max_mc=float(Decimal(np.max(qf_mc)*scale).sqrt())
        # percentiles vs uniform
        qf_cim=(norms.astype(np.float64)**2)/float(scale)
        pct=np.searchsorted(qf_mc_sorted, qf_cim, side='right')/float(mc_samples)
        # throughput/time
        runs_per_sec=float('nan')
        if X in summary:
            try:
                sim_time=float(summary[X].get("sim_time_sec", summary[X].get("sim_time", "nan")))
                if sim_time>0: runs_per_sec=norms.size/sim_time
            except: pass
        recs.append(dict(
            X=X, norms=norms, qf_mc=qf_mc, scale=float(scale),
            min_mc=min_mc, max_mc=max_mc, pct=pct, runs_per_sec=runs_per_sec
        ))
        Xs.append(X)
    if not recs: raise RuntimeError("No usable X found.")
    recs=sorted(recs, key=lambda r:r['X'])
    return recs, os.path.join(res_dir,"analysis","2nd_analysis")

# ------------- Metrics helpers -------------
def common_log_bins(recs, nbins=150):
    all_norms=np.concatenate([r['norms'] for r in recs])
    lo=float(np.min(all_norms)); hi=float(np.max(all_norms))
    return np.logspace(np.log10(lo), np.log10(hi), nbins)

def entropy_from_hist(data, bins):
    h,_=np.histogram(data, bins=bins, density=True)
    p=h+1e-12; p/=p.sum()
    return float(-(p*np.log(p)).sum())

def hellinger(p,q):
    return float(np.sqrt(0.5*np.sum((np.sqrt(p)-np.sqrt(q))**2)))

def ks_stat(sample, ref_sorted):
    # Two-sample KS: fast approx using ECDFs on merged unique points from ref
    s=np.sort(sample); n=len(s); m=len(ref_sorted)
    i=j=0; d=0.0
    while i<n and j<m:
        if s[i]<=ref_sorted[j]: i+=1
        else: j+=1
        d=max(d, abs(i/n - j/m))
    return float(d)

def wasserstein1(a,b):
    # 1-Wass distance on reals via sorted arrays and quantile coupling
    sa=np.sort(a); sb=np.sort(b)
    n=min(len(sa), len(sb))
    if n==0: return float('nan')
    # subsample to equal size if needed
    if len(sa)!=n: sa=sa[np.linspace(0,len(sa)-1,n).astype(int)]
    if len(sb)!=n: sb=sb[np.linspace(0,len(sb)-1,n).astype(int)]
    return float(np.mean(np.abs(sa-sb)))

# ------------- Plot set 1: Efficiency vs X -------------
def plot_best_mean_percentiles(recs, outdir):
    xs=[r['X'] for r in recs]
    best=[float(np.min(r['pct'])) for r in recs]
    mean=[float(np.mean(r['pct'])) for r in recs]
    med =[float(np.median(r['pct'])) for r in recs]
    fig=plt.figure(figsize=(10,6)); ax=fig.add_subplot(111)
    ax.plot(xs, best,'o-',label='Best percentile (lower=better)')
    ax.plot(xs, mean,'o-',label='Mean percentile')
    ax.plot(xs, med ,'o-',label='Median percentile')
    ax.set_xlabel("X (linear)"); ax.set_ylabel("Percentile (0..1)"); ax.set_title("CIM Percentiles vs Uniform")
    ax.legend(); save_png(fig, os.path.join(outdir,"percentiles_vs_X_extended.png"))

def plot_within_factor(recs, outdir, factors=(1.0,1.05,1.1,1.25,1.5)):
    xs=[r['X'] for r in recs]
    fig=plt.figure(figsize=(10,6)); ax=fig.add_subplot(111)
    for k in factors:
        ys=[float(np.mean(r['norms']<=k*r['min_mc'])) for r in recs]
        ax.plot(xs, ys, 'o-', label=f"≤ {k}×MC min")
    ax.set_xlabel("X (linear)"); ax.set_ylabel("Probability"); ax.set_title("Within-factor-of-MC-min vs X")
    ax.legend(); save_png(fig, os.path.join(outdir,"within_factor_vs_X_ext.png"))

def plot_gap_ratios(recs, outdir):
    xs=[r['X'] for r in recs]
    gap_best=[(float(np.min(r['norms']))/r['min_mc']) for r in recs]
    gap_mean=[(float(np.mean(r['norms']))/r['min_mc']) for r in recs]
    fig=plt.figure(figsize=(10,6)); ax=fig.add_subplot(111)
    ax.plot(xs, gap_best,'o-',label='Best / MC-min'); ax.plot(xs, gap_mean,'o-',label='Mean / MC-min')
    ax.set_xlabel("X (linear)"); ax.set_ylabel("Ratio (log)"); set_log(ax,'y'); ax.set_title("Normalized Gaps to MC-min")
    ax.legend(); save_png(fig, os.path.join(outdir,"gap_ratios_vs_X.png"))

def plot_tail_mass_vs_X(recs, outdir):
    xs=[r['X'] for r in recs]
    mass=[float(np.mean(r['norms'] <= np.median(np.sqrt(np.maximum(r['qf_mc']*r['scale'],0.0))))) for r in recs]
    fig=plt.figure(figsize=(10,6)); ax=fig.add_subplot(111)
    ax.plot(xs, mass,'o-'); ax.set_xlabel("X (linear)"); ax.set_ylabel("P(CIM norm ≤ MC median)")
    ax.set_title("Tail mass below MC median vs X"); save_png(fig, os.path.join(outdir,"tail_mass_below_MCmedian.png"))

def plot_zscore_vs_X(recs, outdir):
    xs=[r['X'] for r in recs]; zs=[]
    for r in recs:
        mc = np.sqrt(np.maximum(r['qf_mc']*r['scale'], 0.0))
        ln_mc=np.log(mc+1e-18); mu=ln_mc.mean(); sd=ln_mc.std()+1e-12
        ln_med=np.log(np.median(r['norms']))
        zs.append(float((ln_med - mu)/sd))
    fig=plt.figure(figsize=(10,6)); ax=fig.add_subplot(111)
    ax.plot(xs, zs, 'o-'); ax.set_xlabel("X"); ax.set_ylabel("Z-score of log(median vs MC)")
    ax.set_title("Median position vs uniform (log-scale z-score)"); save_png(fig, os.path.join(outdir,"zscore_vs_X.png"))

# ------------- Plot set 2: Shape & distances -------------
def plot_iqr_cv_entropy_ks_hell_wass(recs, outdir):
    xs=[r['X'] for r in recs]
    bins=common_log_bins(recs, nbins=160)
    iqr=[]; cv=[]; ent=[]; ks=[]; he=[]; wd=[]
    for r in recs:
        a=r['norms']
        q1,q3=np.percentile(a,[25,75]); iqr.append(float(q3/q1))
        ln=np.log(a); cv.append(float(ln.std()/ (abs(ln.mean())+1e-12)))
        ent.append(entropy_from_hist(a,bins))
        # distances
        mc=np.sqrt(np.maximum(r['qf_mc']*r['scale'],0.0))
        ks.append(ks_stat(a, np.sort(mc)))
        # hist for Hellinger/Wasserstein
        p,_=np.histogram(a,bins=bins,density=True); q,_=np.histogram(mc,bins=bins,density=True)
        p=(p+1e-12); p/=p.sum(); q=(q+1e-12); q/=q.sum()
        he.append(hellinger(p,q))
        wd.append(wasserstein1(np.log(a), np.log(mc)))
    # Plot 4 panes
    fig,axs=plt.subplots(2,2,figsize=(12,10))
    axs[0,0].plot(xs, iqr,'o-'); axs[0,0].set_xlabel("X"); axs[0,0].set_ylabel("Q3/Q1 (norm)"); axs[0,0].set_title("IQR ratio vs X")
    axs[0,1].plot(xs, cv,'o-'); axs[0,1].set_xlabel("X"); axs[0,1].set_ylabel("CV of log(norm)"); axs[0,1].set_title("CV(log) vs X")
    axs[1,0].plot(xs, ent,'o-'); axs[1,0].set_xlabel("X"); axs[1,0].set_ylabel("Entropy (log-binned)"); axs[1,0].set_title("Entropy vs X")
    axs[1,1].plot(xs, ks,'o-',label='KS'); axs[1,1].plot(xs, he,'o-',label='Hellinger'); axs[1,1].plot(xs, wd,'o-',label='Wasserstein( log )')
    axs[1,1].set_xlabel("X"); axs[1,1].set_ylabel("Distance"); axs[1,1].set_title("Distances CIM vs Uniform"); axs[1,1].legend()
    save_png(fig, os.path.join(outdir,"shape_distances_vs_X.png"))

# ------------- Plot set 3: Heatmaps & matrices -------------
def plot_bands_heatmap(recs, outdir):
    xs=[r['X'] for r in recs]; bands=[1.0,1.05,1.1,1.25,1.5]
    M=[]
    for r in recs:
        probs=[]
        prev=0.0
        for t in bands:
            thr=t*r['min_mc']; probs.append(float(np.mean((r['norms']>prev)&(r['norms']<=thr)))); prev=thr
        probs.append(float(np.mean(r['norms']>bands[-1]*r['min_mc'])))
        M.append(probs)
    M=np.array(M).T
    fig=plt.figure(figsize=(12,6)); ax=fig.add_subplot(111)
    im=ax.imshow(M, aspect='auto', origin='lower', extent=[xs[0], xs[-1], 0, M.shape[0]-1], cmap='viridis')
    ax.set_xlabel("X"); ax.set_ylabel("Band index (0..n)"); fig.colorbar(im, ax=ax, label="Probability")
    ax.set_title("Probability mass in relative bands vs X"); save_png(fig, os.path.join(outdir,"bands_heatmap.png"))

def plot_percentile_area(recs, outdir, bins=10):
    xs=[r['X'] for r in recs]
    edges=np.linspace(0,1,bins+1); H=[]
    for r in recs:
        h,_=np.histogram(r['pct'], bins=edges, density=True)
        h=h/np.sum(h)
        H.append(h)
    H=np.array(H).T
    fig=plt.figure(figsize=(12,6)); ax=fig.add_subplot(111)
    # Stacked area over X
    for i in range(H.shape[0]):
        bottom=np.sum(H[:i,:], axis=0)
        ax.fill_between(xs, bottom, bottom+H[i,:], alpha=0.6, label=f"{edges[i]:.1f}-{edges[i+1]:.1f}")
    ax.set_xlabel("X"); ax.set_ylabel("Fraction"); ax.set_title("Percentile bins (vs uniform) stacked over X")
    save_png(fig, os.path.join(outdir,"percentile_stacked_area.png"))

def plot_pairwise_jsd_matrix(recs, out_dir, nbins=150):
    bins = common_log_bins(recs, nbins=nbins)
    P = []
    for r in recs:
        h, _ = np.histogram(r['norms'], bins=bins, density=True)
        p = (h + 1e-12)
        p = p / p.sum()
        P.append(p)
    n = len(P)
    M = np.zeros((n, n))

    def jsd_func(p, q):
        m = 0.5 * (p + q)
        def kl(a, b):
            mask = (a > 0) & (b > 0)
            return np.sum(a[mask] * np.log(a[mask] / b[mask]))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    for i in range(n):
        for j in range(n):
            M[i, j] = jsd_func(P[i], P[j])

    xs = [r['X'] for r in recs]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, origin='lower', cmap='magma')
    fig.colorbar(im, ax=ax, label='JSD')
    ax.set_xticks(range(n)); ax.set_xticklabels(xs, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(xs)
    ax.set_title("Pairwise JSD between CIM distributions across X")
    save_png(fig, os.path.join(out_dir, "pairwise_jsd_matrix.png"))
    
def plot_pca_of_distributions(recs, outdir, nbins=150):
    bins=common_log_bins(recs, nbins=nbins)
    mats=[]; xs=[]
    for r in recs:
        h,_=np.histogram(r['norms'], bins=bins, density=True)
        p=(h+1e-12); p/=p.sum(); mats.append(p); xs.append(r['X'])
    A=np.vstack(mats)
    # PCA via SVD
    A_center=A - A.mean(axis=0, keepdims=True)
    U,S,Vt=np.linalg.svd(A_center, full_matrices=False)
    coords=U[:,:2]*S[:2]
    fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111)
    sc=ax.scatter(coords[:,0], coords[:,1], c=xs, cmap='viridis', s=40)
    for i,x in enumerate(xs):
        ax.text(coords[i,0], coords[i,1], str(x), fontsize=8, ha='center', va='center')
    fig.colorbar(sc, ax=ax, label='X')
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA of CIM distributions across X")
    save_png(fig, os.path.join(outdir,"pca_distributions.png"))

# ------------- Plot set 4: Time/throughput correlations -------------
def plot_throughput_vs_X(recs, outdir):
    xs=[r['X'] for r in recs]; y=[r['runs_per_sec'] for r in recs]
    fig=plt.figure(figsize=(10,6)); ax=fig.add_subplot(111)
    ax.plot(xs,y,'o-'); ax.set_xlabel("X"); ax.set_ylabel("Runs/sec"); ax.set_title("Throughput vs X (recomputed)")
    save_png(fig, os.path.join(outdir,"throughput_vs_X.png"))

def plot_bestpct_vs_throughput(recs, outdir):
    xs=[r['X'] for r in recs]; rp=[r['runs_per_sec'] for r in recs]; bp=[float(np.min(r['pct'])) for r in recs]
    fig=plt.figure(figsize=(10,6)); ax=fig.add_subplot(111)
    sc=ax.scatter(rp,bp,c=xs,cmap='plasma'); fig.colorbar(sc, ax=ax, label='X')
    ax.set_xlabel("Runs/sec"); ax.set_ylabel("Best percentile (lower=better)")
    ax.set_title("Best percentile vs throughput"); save_png(fig, os.path.join(outdir,"bestpct_vs_throughput.png"))

# ------------- Plot set 5: Convergence proxies -------------
def runs_to_reach_factor(norms, thr):
    idx=np.where(norms<=thr)[0]
    return int(idx[0]+1) if idx.size>0 else np.nan

def plot_runs_to_targets(recs, outdir, factors=(1.02,1.05,1.1)):
    xs=[r['X'] for r in recs]
    fig=plt.figure(figsize=(10,6)); ax=fig.add_subplot(111)
    for f in factors:
        ys=[]
        for r in recs:
            thr=f*r['min_mc']; ys.append(runs_to_reach_factor(r['norms'], thr))
        ax.plot(xs, ys, 'o-', label=f"runs to ≤ {f}×MCmin")
    ax.set_xlabel("X"); ax.set_ylabel("Runs"); ax.set_title("Runs to reach relative targets vs X"); ax.legend()
    save_png(fig, os.path.join(outdir,"runs_to_targets_vs_X.png"))

# ------------- Plot set 6: Multi-panel comparisons -------------
def plot_multi_cdf_vs_uniform(recs, outdir, sel=None):
    if sel is None:
        xs=[r['X'] for r in recs]
        sel=[xs[0], xs[len(xs)//4], xs[len(xs)//2], xs[3*len(xs)//4], xs[-1]]
        sel=sorted(set(sel))
    fig,axs=plt.subplots(len(sel),1,figsize=(8,2.5*len(sel)))
    if len(sel)==1: axs=[axs]
    for ax_i,(ax,xval) in enumerate(zip(axs, sel)):
        r=next(rr for rr in recs if rr['X']==xval)
        c=np.sort(r['norms']); y=np.arange(1,c.size+1)/float(c.size)
        mc=np.sqrt(np.maximum(r['qf_mc']*r['scale'],0.0)); u=np.sort(mc); yu=np.arange(1,u.size+1)/float(u.size)
        ax.plot(c,y,label='CIM'); ax.plot(u,yu,label='Uniform',alpha=0.7)
        set_log(ax,'x'); ax.set_xlabel("Norm (log)"); ax.set_ylabel("CDF"); ax.set_title(f"CDFs at X={xval}")
        if ax_i==0: ax.legend()
    save_png(fig, os.path.join(outdir,"cdf_cim_vs_uniform_selected.png"))

# ------------- Main -------------
def main():
    ap=argparse.ArgumentParser(description="2nd analysis: many across-X graphs (PNG only).")
    ap.add_argument("--results_dir", type=str, default=None)
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--mc_samples", type=int, default=30000)
    ap.add_argument("--mc_batch", type=int, default=4096)
    args=ap.parse_args()

    res_dir = args.results_dir or latest_results_dir()
    print(f"Using results_dir: {res_dir}")
    recs, out_root = build_records(res_dir, args.dim, args.seed, args.mc_samples, args.mc_batch)
    out_root=ensure_dir(out_root)

    # Subfolders
    folders={
        "efficiency": ensure_dir(os.path.join(out_root,"efficiency_png")),
        "shape": ensure_dir(os.path.join(out_root,"shape_png")),
        "heatmaps": ensure_dir(os.path.join(out_root,"heatmaps_png")),
        "pca_jsd": ensure_dir(os.path.join(out_root,"pca_jsd_png")),
        "throughput": ensure_dir(os.path.join(out_root,"throughput_png")),
        "convergence": ensure_dir(os.path.join(out_root,"convergence_png")),
        "multigrid": ensure_dir(os.path.join(out_root,"multigrid_png")),
    }

    # Efficiency family
    plot_best_mean_percentiles(recs, folders["efficiency"])          # 1
    plot_within_factor(recs, folders["efficiency"])                  # 2
    plot_gap_ratios(recs, folders["efficiency"])                     # 3
    plot_tail_mass_vs_X(recs, folders["efficiency"])                 # 4
    plot_zscore_vs_X(recs, folders["efficiency"])                    # 5

    # Shape & distances
    plot_iqr_cv_entropy_ks_hell_wass(recs, folders["shape"])         # 6

    # Heatmaps & matrices
    plot_bands_heatmap(recs, folders["heatmaps"])                    # 7
    plot_percentile_area(recs, folders["heatmaps"])                  # 8
    plot_pairwise_jsd_matrix(recs, folders["pca_jsd"])               # 9
    plot_pca_of_distributions(recs, folders["pca_jsd"])              # 10

    # Throughput correlations
    plot_throughput_vs_X(recs, folders["throughput"])                # 11
    plot_bestpct_vs_throughput(recs, folders["throughput"])          # 12

    # Convergence proxies
    plot_runs_to_targets(recs, folders["convergence"])               # 13

    # Multi-panel comparisons
    plot_multi_cdf_vs_uniform(recs, folders["multigrid"])            # 14

    print(f"Second analysis PNGs saved under: {out_root}")

if __name__ == "__main__":
    main()
