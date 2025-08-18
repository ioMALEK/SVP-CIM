#!/usr/bin/env python3
"""
cim_greedy_sweep.py
===================

10-hour CIM sensitivity study with restart & plotting support.

Grid
  Δt  : 12 values 0.01-0.07
  μ   : [0.1, 0.3, 0.5, 0.7]
  r   : [0.05, 0.2, 0.4, 0.6]
  σ   : [0.00, 0.05, 0.10, 0.15]

Per pivot
  – greedy search  : 600 runs   (20 / evaluation, Δt radius ±3)
  – final average  : 1 000 runs (norms stored)

Output folder structure
  results_YYYY-MM-DD_HH-MM-SS/
      manifest.yaml
      X5/
          data/
              stats.csv
              norms_pivot0000.npz
              ...
          plots/
              *.png
      X6/
      X7/
"""

# ------------------- imports & housekeeping ------------------- #
import os, sys, ast, re, math, itertools, datetime, yaml, json, hashlib, argparse
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Tuple, List, Dict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import joblib
from tqdm.auto import tqdm

from cim_optimizer.solve_Ising import Ising

getcontext().prec = 120

# ------------------- parameter grid ---------------------------- #
DT   = np.linspace(0.01, 0.07, 12)
MU   = [0.1, 0.3, 0.5, 0.7]
R    = [0.05, 0.2, 0.4, 0.6]
SIG  = [0.00, 0.05, 0.10, 0.15]
AXES = [DT, MU, R, SIG]

GREEDY_BUDGET     = 600     # runs / pivot
RUNS_PER_EVAL     = 20
DT_RADIUS         = 3       # ± index steps during greedy
FINAL_RUNS        = 1000

GLOBAL_SEED       = 2025
N_JOBS_DEFAULT    = 4

# ------------------- helpers ----------------------------------- #
def load_lattice(dim: int, seed: int, folder="svp_lattices") -> np.ndarray:
    p = Path(folder) / f"dim{dim}_seed{seed}.txt"
    txt = p.read_text().strip()
    try:
        B = np.array(ast.literal_eval(txt), dtype=object)
    except Exception:
        rows = []
        for line in txt.splitlines():
            nums = re.findall(r"-?\d+", line)
            if nums:
                rows.append([int(x) for x in nums])
        B = np.array(rows, dtype=object)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("basis not square")
    return B

def vec_sq_norm_int(v): return int(sum(int(x)*int(x) for x in v))
def gram_int(B): return B @ B.T
def build_J(B, scale=1.0):
    G = gram_int(B)
    m = max(1, max(abs(int(x)) for x in G.flat))
    J = -np.asarray(G, dtype=np.float64)/float(m)*scale
    np.fill_diagonal(J, 0.0)
    return J
def sq_from_spins(s, G):
    return int(sum(int(si)*int(sj)*int(G[i,j]) for i,si in enumerate(s) for j,sj in enumerate(s)))
def sqrt_int(n): return float(math.sqrt(n)) if n.bit_length()<52 else float(Decimal(n).sqrt())
def to_pm1(a):
    a=np.asarray(a);vals=set(np.unique(np.rint(a)))
    if vals<= {-1,1}: return a.astype(int)
    if vals<= {0,1}:  return (2*a-1).astype(int)
    return np.sign(a).astype(int)

def extract_spins(res, K:int):
    obj=getattr(res,'result',res)
    def as_mat(x): x=np.asarray(x); return x if x.ndim==2 else x[None,:]

    fields=['spin_configurations','spin_configurations_all_runs',
            'spin_config_all_runs','spins','states_all_runs']
    for n in fields:
        if isinstance(obj,dict) and n in obj: M=as_mat(obj[n])
        elif hasattr(obj,n): M=as_mat(getattr(obj,n))
        else: continue
        if M.shape[1]==K: return to_pm1(M)
    traj=['spin_trajectories','spin_trajectories_all_runs','states_trajectories']
    for n in traj:
        if isinstance(obj,dict) and n in obj: T=np.asarray(obj[n])
        elif hasattr(obj,n): T=np.asarray(getattr(obj,n))
        else: continue
        if T.ndim==3 and T.shape[-1]==K: return to_pm1(T[:,-1,:])
    single=['lowest_energy_spin_configuration','lowest_energy_spin_config',
            'lowest_energy_state','state']
    for n in single:
        if isinstance(obj,dict) and n in obj: M=as_mat(obj[n])
        elif hasattr(obj,n): M=as_mat(getattr(obj,n))
        else: continue
        if M.shape[1]==K: return to_pm1(M)
    if hasattr(obj,'__dict__'):
        for v in obj.__dict__.values():
            if isinstance(v,np.ndarray):
                M=as_mat(v)
                if M.shape[1]==K and set(np.unique(np.rint(M)))<= {-1,0,1}:
                    return to_pm1(M)
    raise RuntimeError("spins not found")

# deterministic 32-bit seed
def make_seed(*keys)->int:
    h=hashlib.blake2b(repr(keys).encode(),digest_size=4).digest()
    return int.from_bytes(h,'little')^GLOBAL_SEED

# CIM evaluation (returns list of norms)
def cim_norms(solver:Ising, idx, K, G_int, num_runs:int, seed:int):
    dt,mu,r,sig=(AXES[d][idx[d]] for d in range(4))
    st=np.random.get_state()
    np.random.seed(seed)
    try:
        res=solver.solve(num_timesteps_per_run=600,
                         cac_time_step=dt,cac_mu=mu,cac_r=r,
                         cac_noise=sig,num_runs=num_runs,
                         use_CAC=True,use_GPU=False,
                         suppress_statements=True)
    finally:
        np.random.set_state(st)
    spins=extract_spins(res,K)
    norms=[sqrt_int(sq_from_spins(s,G_int)) for s in spins]
    return norms

# neighbours generator (dt radius ±3, axis neighbours for others)
def neighbours(idx):
    i,j,k,l=idx
    for di in range(-DT_RADIUS,DT_RADIUS+1):
        ni=i+di
        if 0<=ni<len(DT) and ni!=i:
            yield (ni,j,k,l)
    for dj in (-1,1):
        nj=j+dj
        if 0<=nj<len(MU): yield (i,nj,k,l)
    for dk in (-1,1):
        nk=k+dk
        if 0<=nk<len(R):  yield (i,j,nk,l)
    for dl in (-1,1):
        nl=l+dl
        if 0<=nl<len(SIG): yield (i,j,k,nl)

# greedy search
def greedy_search(idx0,solver,K,G_int):
    cache={}
    def score(idx,tag):
        if idx in cache: return cache[idx]
        m=np.mean(cim_norms(solver,idx,K,G_int,RUNS_PER_EVAL,make_seed('g',idx,tag)))
        cache[idx]=m; return m

    best=idx0
    best_val=score(best,"start")
    budget=GREEDY_BUDGET-RUNS_PER_EVAL
    step=0
    while budget>=RUNS_PER_EVAL:
        improved=False
        for nb in neighbours(best):
            if budget<RUNS_PER_EVAL: break
            v=score(nb,f"s{step}")
            budget-=RUNS_PER_EVAL
            if v<best_val-1e-12:
                best,best_val=nb,v
                improved=True
                break
        if not improved: break
        step+=1
    return best,best_val

# progress-bar glue
class tqdm_joblib:
    def __init__(self,t): self.t=t; self._old=None
    def __enter__(self):
        self._old=joblib.parallel.BatchCompletionCallBack
        t=self.t
        class CB(self._old):
            def __call__(self,*a,**k):
                t.update(n=self.batch_size)
                return super().__call__(*a,**k)
        joblib.parallel.BatchCompletionCallBack=CB
        return self.t
    def __exit__(self,*exc):
        joblib.parallel.BatchCompletionCallBack=self._old; self.t.close()

# ------------- per-dimension run -------------------------------- #
def run_dimension(X:int, root:Path, done_resume:bool):
    dim_dir=root/f"X{X}"
    data_dir=dim_dir/"data"; plots_dir=dim_dir/"plots"
    data_dir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(exist_ok=True)

    # completed pivots
    done=set()
    stats_path=data_dir/"stats.csv"
    if stats_path.is_file():
        done_df=pd.read_csv(stats_path)
        done=set(map(tuple,done_df[["pivot_dt","pivot_mu","pivot_r","pivot_sig"]].itertuples(index=False, name=None)))
        stats_list=done_df.to_dict('records')
    else:
        stats_list=[]

    # lattice
    B_full=load_lattice(50,17)
    norms=np.array([vec_sq_norm_int(v) for v in B_full],dtype=object)
    B=B_full[np.argsort([int(x) for x in norms])][:X]
    G_int=gram_int(B); J=build_J(B,1.0)
    solver=Ising(J=J,h=np.zeros(X))
    best_sq=min(sq_from_spins(np.array([(b>>i)&1 and 1 or -1 for i in range(X)]),G_int)
                for b in range(1<<X))
    opt_norm=sqrt_int(best_sq)

    pivots=[(i,j,k,l) for i in range(len(DT))
                     for j in range(len(MU))
                     for k in range(len(R))
                     for l in range(len(SIG))]
    remaining=[idx for idx in pivots
               if (AXES[0][idx[0]],AXES[1][idx[1]],AXES[2][idx[2]],AXES[3][idx[3]]) not in done]

    if not remaining:
        print(f"[X{X}] all pivots already done.")
    else:
        print(f"[X{X}] running {len(remaining)} / {len(pivots)} pivots")

        def worker(idx,pid):
            best_idx,_=greedy_search(idx,solver,X,G_int)
            norms=cim_norms(solver,best_idx,X,G_int,FINAL_RUNS,make_seed('final',X,idx))
            mean_norm=float(np.mean(norms))
            inv_eff=opt_norm/mean_norm
            rec=dict(pivot_dt=float(AXES[0][idx[0]]),
                     pivot_mu=float(AXES[1][idx[1]]),
                     pivot_r=float(AXES[2][idx[2]]),
                     pivot_sig=float(AXES[3][idx[3]]),
                     best_dt=float(AXES[0][best_idx[0]]),
                     best_mu=float(AXES[1][best_idx[1]]),
                     best_r=float(AXES[2][best_idx[2]]),
                     best_sig=float(AXES[3][best_idx[3]]),
                     inv_eff=inv_eff,
                     mean_norm=mean_norm)
            # save norms
            np.savez_compressed(data_dir/f"norms_pivot{pid:04d}.npz",norms=np.array(norms,dtype=np.float64))
            return rec

        with tqdm_joblib(tqdm(total=len(remaining),desc=f"X{X} pivots")):
            new_records=Parallel(n_jobs=N_JOBS_DEFAULT)(
                delayed(worker)(idx, pid) for pid,idx in enumerate(remaining))

        stats_list.extend(new_records)
        pd.DataFrame(stats_list).to_csv(stats_path,index=False)

    # plots
    stats_df=pd.read_csv(stats_path)
    plt.figure(figsize=(6,4))
    sns.histplot(stats_df["inv_eff"],bins=40,color='steelblue')
    plt.xlabel("opt_norm / mean_norm"); plt.ylabel("count")
    plt.title(f"Inverse efficiency (X={X})")
    plt.tight_layout(); plt.savefig(plots_dir/"inv_eff_hist.png",dpi=160); plt.close()

    slice_df=stats_df[(stats_df.best_r==0.4)&(stats_df.best_sig==0.10)]
    if not slice_df.empty:
        piv=slice_df.pivot_table(index="best_dt",columns="best_mu",
                                 values="inv_eff",aggfunc="mean")
        plt.figure(figsize=(5,4))
        sns.heatmap(piv,vmin=0,vmax=1,cmap="viridis",annot=True,fmt=".2f")
        plt.title(f"opt_norm/mean_norm (r=0.4 σ=0.1) X={X}")
        plt.xlabel("μ"); plt.ylabel("Δt")
        plt.tight_layout(); plt.savefig(plots_dir/"heatmap_r0.4_s0.1.png",dpi=160); plt.close()
    plt.figure(figsize=(6,4))
    sc=plt.scatter(stats_df["best_dt"],stats_df["best_mu"],c=stats_df["inv_eff"],
                   cmap="viridis",vmin=0,vmax=1,s=20)
    plt.colorbar(sc,label="inv_eff")
    plt.xlabel("best Δt"); plt.ylabel("best μ"); plt.title(f"Scatter X={X}")
    plt.tight_layout(); plt.savefig(plots_dir/"scatter_dt_mu.png",dpi=160); plt.close()
    print(f"[X{X}] stats & plots done.")

# ------------------- main CLI ---------------------------------- #
def main():
    parser=argparse.ArgumentParser(description="10-h CIM greedy sweep")
    parser.add_argument("--dims",nargs="+",type=int,default=[5,6,7],
                        help="list of X values to run")
    parser.add_argument("--resume",type=str,
                        help="existing results_* directory to continue")
    parser.add_argument("--plots-only",action="store_true",
                        help="make plots only (needs --resume)")
    args=parser.parse_args()

    if args.plots_only and not args.resume:
        sys.exit("--plots-only needs --resume DIR")

    if args.resume:
        root=Path(args.resume).expanduser()
        if not root.is_dir():
            sys.exit(f"{root} not found.")
    else:
        root=Path(f"results_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}")
        root.mkdir()

    if not args.plots_only and not (root/"manifest.yaml").is_file():
        # write manifest only once
        manifest=dict(timestamp=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                      axes_sizes=dict(dt=len(DT),mu=len(MU),r=len(R),sig=len(SIG)),
                      greedy_budget=GREEDY_BUDGET,runs_per_eval=RUNS_PER_EVAL,
                      dt_radius=DT_RADIUS,final_runs=FINAL_RUNS,
                      dims=args.dims,global_seed=GLOBAL_SEED)
        with open(root/"manifest.yaml","w") as f: yaml.dump(manifest,f)

    for X in args.dims:
        run_dimension(X,root,done_resume=bool(args.resume))

if __name__=="__main__":
    main()
