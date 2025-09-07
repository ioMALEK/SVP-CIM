#!/usr/bin/env python3
"""
landscape_diagnostic_of_X20.py
Coarse (+ refined) exploration of the CIM hyper-parameter landscape for
dimension-20.  Works with either the default SVP-challenge basis or any
user-supplied lattice file passed with --basis.

Results are written block-by-block so an interruption loses at most
<chunk> samples.

Directory layout
----------------
runs/LANDSCAPE/<tag>/
    coarse.csv   dt,mu,r,noise,j_scale,steps,eff
    refined.csv  + eff_ref, incline_up, decline_down
    meta.json    opt_norm, total_samples
"""
from __future__ import annotations
import argparse, json, random, hashlib, re, contextlib, os
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from cim_svp import (load_lattice, vec_sq_norm_int, gram_int, build_J,
                     sq_from_spins, sqrt_int, CIMSolver)
from cim_svp.extras.classical_solvers import shortest_enum

# ── silence helper for CIM ---------------------------------------------------
@contextlib.contextmanager
def silence():
    with open(os.devnull, "w") as devnull, \
         contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull):
        yield

# ---------- CLI --------------------------------------------------------------
pa = argparse.ArgumentParser()
pa.add_argument("--samples", type=int, default=50000)
pa.add_argument("--jobs",    type=int, default=6)
pa.add_argument("--chunk",   type=int, default=5000,
                help="flush to disk every CHUNK samples")
pa.add_argument("--tag",     required=True)
pa.add_argument("--basis",   help="custom lattice file (plain ints)")
pa.add_argument("--resume",  action="store_true")
pa.add_argument("--blacklist")
pa.add_argument("--sanity",  action="store_true")
args = pa.parse_args()

# ---------- load basis -------------------------------------------------------
def load_plain_int_matrix(path: str) -> np.ndarray:
    rows = [list(map(int, re.findall(r"-?\d+", ln)))
            for ln in Path(path).read_text().splitlines()
            if re.search(r"\d", ln)]
    if not rows:
        raise ValueError(f"{path} contains no integers?")
    return np.array(rows, dtype=object)

DIM, SEED = 20, 17
if args.basis:
    B_full = load_plain_int_matrix(args.basis)
    print(f"[INFO] custom basis {args.basis} loaded, shape={B_full.shape}")
else:
    B_full = load_lattice(50, SEED, folder="svp_lattices")
    print("[INFO] challenge basis loaded")

idx = np.argsort([int(vec_sq_norm_int(v)) for v in B_full])[:DIM]
B   = B_full[idx]
G   = gram_int(B)
J0  = build_J(B)
OPT_NORM = shortest_enum(B)[1]                 # exact shortest vector norm

# ---------- hyper-parameter bounds ------------------------------------------
BOUNDS = dict(
    dt      =(0.01, 0.07),
    mu      =(0.05, 0.8),
    r       =(0.05, 0.6),
    noise   =(1e-5, 0.30),
    j_scale =(0.2,  5.0),
    steps   =(400, 1000)        # integer multiples of 100
)
KEYS = ("dt","mu","r","noise","j_scale","steps")

# ---------- utility functions -----------------------------------------------
def hp_hash(d: dict) -> str:
    tup = tuple(round(d[k],6) if k!="steps" else d[k] for k in KEYS)
    return hashlib.sha1(str(tup).encode()).hexdigest()

def random_point(rnd: random.Random):
    p = {k: rnd.uniform(*BOUNDS[k]) for k in BOUNDS if k!="steps"}
    p["steps"] = rnd.randrange(*BOUNDS["steps"], 100)
    return p

def efficiency(hp: dict, shots: int, rng: np.random.Generator):
    solver = CIMSolver(J0*hp["j_scale"], np.zeros(DIM))
    seeds  = rng.integers(2**31-1, size=shots)
    norms  = []
    for s in seeds:
        with silence():
            spin = solver.solve(steps=hp["steps"], dt=hp["dt"], mu=hp["mu"],
                                r=hp["r"], noise=hp["noise"], seed=int(s))
        norms.append(sqrt_int(sq_from_spins(spin, G)))
    return float(OPT_NORM / np.mean(norms))

def slopes(hp: dict, base: float, rng: np.random.Generator):
    up = down = 0.0
    for k in KEYS:
        lo, hi = BOUNDS[k]
        u, d = hp.copy(), hp.copy()
        if k=="steps":
            u[k] = min(u[k]+100, hi)
            d[k] = max(d[k]-100, lo)
        else:
            u[k] = min(hp[k]*1.05, hi)
            d[k] = max(hp[k]*0.95, lo)
        up   = max(up,   efficiency(u,10,rng) - base)
        down = max(down, base - efficiency(d,10,rng))
    return up, down

# ═══════════════════════════════════════════════════════════════════
if args.sanity:
    args.samples = 1000
    args.chunk   = 1000
    print("[SANITY] quick test run")

out = Path("runs")/"LANDSCAPE"/args.tag
out.mkdir(parents=True, exist_ok=True)

# hashes to skip
skip:set[str] = set()
if args.resume and (out/"coarse.csv").is_file():
    skip.update(pd.read_csv(out/"coarse.csv")
                   .apply(lambda r: hp_hash(r), axis=1))
if args.blacklist:
    skip.update(pd.read_csv(args.blacklist)
                   .apply(lambda r: hp_hash(r), axis=1))
print(f"[INFO] skip-set size {len(skip)}")

rng_global = np.random.default_rng(2025)
rnd_py     = random.Random(2025)

remaining = args.samples
pbar_total = tqdm(total=args.samples, desc="Coarse total", unit="pt")
while remaining:
    batch = min(remaining, args.chunk); remaining -= batch
    block=[]
    while len(block)<batch:
        hp = random_point(random.Random(rnd_py.random()))
        sig=hp_hash(hp)
        if sig in skip: continue
        skip.add(sig); block.append(hp)

    def coarse(i):
        eff = efficiency(block[i], 5, rng_global)
        return i, eff
    vals = Parallel(args.jobs)(delayed(coarse)(i) for i in range(batch))
    for i,eff in vals: block[i]["eff"] = eff
    pd.DataFrame(block).to_csv(out/"coarse.csv",
                               mode="a",
                               header=not (out/"coarse.csv").is_file(),
                               index=False)
    pbar_total.update(batch)

pbar_total.close()
df_coarse = pd.read_csv(out/"coarse.csv")
good = df_coarse.query("eff > 0.10")
print(f"[REFINE] {len(good)} promising points")

ref_path = out/"refined.csv"
done=set()
if ref_path.is_file():
    done.update(pd.read_csv(ref_path).apply(lambda r: hp_hash(r), axis=1))

with open(ref_path,"a",buffering=1) as f:
    if ref_path.stat().st_size==0:
        f.write(",".join([*KEYS,"eff_ref","incline_up","decline_down"])+"\n")
    for _,row in tqdm(good.iterrows(), total=len(good), desc="Refine", unit="pt"):
        sig=hp_hash(row)
        if sig in done: continue
        hp={k:row[k] for k in KEYS}
        rng=np.random.default_rng(int(row.name))
        eff=efficiency(hp,60,rng)
        up,dn=slopes(hp,eff,rng)
        f.write(",".join(str(hp[k]) for k in KEYS)+f",{eff},{up},{dn}\n")
        done.add(sig)

json.dump({"opt_norm": float(OPT_NORM),
           "total_samples": len(df_coarse)},
          open(out/"meta.json","w"))
print(f"[DONE] coarse rows: {len(df_coarse)}  refined rows: {len(done)}")
