#!/usr/bin/env python3
"""
SVP_search_basis.py
1. keep k shortest vectors from the source lattice
2. add every ± length-2 combination via build_extended_subspace
3. rescale each new combo so its norm equals the mean norm of the
   originals
4. save augmented basis to runs/augmented/<tag>/basis.txt
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np

from cim_svp import vec_sq_norm_int
from cim_svp.extras.encoding import build_extended_subspace   # real helper

SRC_DEFAULT = "svp_lattices/dim50_seed17.txt"

# ---------- tiny loader that handles literal & plain formats ----------
def load_matrix(path: str) -> np.ndarray:
    txt = Path(path).read_text().splitlines()
    rows = [list(map(int, re.findall(r"-?\d+", ln))) for ln in txt if re.search(r"\d", ln)]
    return np.array(rows, dtype=object)

def select_shortest(B: np.ndarray, k: int):
    idx = np.argsort([int(vec_sq_norm_int(v)) for v in B])[:k]
    return B[idx]

def rescale_to_mean(orig: np.ndarray, add: list[np.ndarray]):
    target = float(np.mean([np.linalg.norm(v.astype(float)) for v in orig]))
    scaled = [(target/np.linalg.norm(w.astype(float)))*w for w in add]
    return np.array(scaled, dtype=object)

# ----------------------------------------------------------------------
pa = argparse.ArgumentParser()
pa.add_argument("--source", default=SRC_DEFAULT)
pa.add_argument("--k", type=int, default=8)
pa.add_argument("--tag", required=True)
args = pa.parse_args()

B0    = load_matrix(args.source)
B_sel = select_shortest(B0, args.k)

# length-2 combinations with coeff ±1
stats, cand = build_extended_subspace(
                 B_sel,
                 support_min=2, support_max=2)      # correct call
combos = [vec for vec, _ in cand]
combos = rescale_to_mean(B_sel, combos)

B_out = np.vstack([B_sel, combos])

out_dir = Path("runs")/"augmented"/args.tag
out_dir.mkdir(parents=True, exist_ok=True)
np.savetxt(out_dir/"basis.txt", B_out, fmt="%d")
print(f"[DONE] {B_out.shape[0]} vectors saved → {out_dir/'basis.txt'}")