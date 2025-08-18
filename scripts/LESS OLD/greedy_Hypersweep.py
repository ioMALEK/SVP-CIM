#!/usr/bin/env python3
"""
cim_hyper_sweep_subspace.py      (progress-bar edition)

Coarse-grid study of CIM hyper-parameters for sub-bases X = 3 … 10.

Output for each X
    • sweep_X{X}.csv   (all 256 grid points)
    • heat-map PNG     (Δt × μ slice at r=0.4, σ=0.10)

CSV columns
    X, cac_dt, mu, r, noise, mean_norm, eff, p_opt
"""

# ----------------------------  housekeeping  ----------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import ast, re, math, datetime, itertools
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from cim_optimizer.solve_Ising import Ising

getcontext().prec = 100     # high-precision sqrt for huge integers

# ----------------------------  I/O helpers  -----------------------------
def load_local_lattice(dim: int, seed: int, folder="svp_lattices") -> np.ndarray:
    """
    Load a square basis (dtype=object). Accepts either a Python literal or
    plain rows of integers separated by spaces/commas.
    """
    path = Path(folder) / f"dim{dim}_seed{seed}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text().strip()
    # 1) try Python literal
    try:
        B = ast.literal_eval(text)
        B = np.array(B, dtype=object)
    except Exception:
        # 2) generic fallback
        rows = []
        for line in text.splitlines():
            nums = re.findall(r"-?\d+", line)
            if nums:
                rows.append([int(x) for x in nums])
        B = np.array(rows, dtype=object)

    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"Loaded basis has shape {B.shape}, expected square.")
    return B

# ----------------------------  math utilities  --------------------------
def vec_sq_norm_int(vec) -> int:
    return int(sum(int(x) * int(x) for x in vec))

def gram_int(B: np.ndarray) -> np.ndarray:
    """Exact Gram matrix (Python ints)."""
    return B @ B.T

def build_J(B: np.ndarray, scaling: float = 1.0) -> np.ndarray:
    """Float64 Ising matrix J = –G / max|G| ⋅ scaling."""
    G = gram_int(B)
    max_abs = max(1, max(abs(int(x)) for x in G.flat))
    J = - (np.asarray(G, dtype=np.float64) / float(max_abs)) * scaling
    np.fill_diagonal(J, 0.0)
    return J

def sq_from_spins(s: np.ndarray, G_int: np.ndarray) -> int:
    """Exact quadratic form sᵀ G s."""
    return int(sum(int(si) * int(sj) * int(G_int[i, j])
                   for i, si in enumerate(s) for j, sj in enumerate(s)))

def sqrt_int(n: int) -> float:
    """Accurate sqrt of an int, returned as float."""
    if n.bit_length() < 52:
        return math.sqrt(n)
    return float(Decimal(n).sqrt())

# ----------------------------  CIM helpers  -----------------------------
# ------------------------------------------------------------------
#  Robust ±1 conversion
# ------------------------------------------------------------------
def to_pm1(arr: np.ndarray) -> np.ndarray:
    """Map any {−1,0,1} or {0,1} array to strict ±1 integers."""
    arr = np.asarray(arr)
    vals = set(np.unique(np.rint(arr)))
    if vals <= {-1, 1}:       # already ±1
        return arr.astype(int)
    if vals <= {0, 1}:        # 0/1 → −1/+1
        return (2*arr - 1).astype(int)
    return np.sign(arr).astype(int)

# ------------------------------------------------------------------
#  Version-agnostic spin extractor
# ------------------------------------------------------------------
def extract_spins(result, K: int) -> np.ndarray:
    """
    Return an (R × K) array with entries in {−1, +1} from any
    cim_optimizer.solve_Ising result object / dict.
    """
    obj = getattr(result, 'result', result)        # some versions wrap payload
    def as_mat(x):
        x = np.asarray(x)
        return x if x.ndim == 2 else x[None, :]

    # 1) common per-run fields
    fields = ['spin_configurations', 'spin_configurations_all_runs',
              'spin_config_all_runs', 'spins', 'states_all_runs']
    for name in fields:
        if isinstance(obj, dict) and name in obj:
            M = as_mat(obj[name])
        elif hasattr(obj, name):
            M = as_mat(getattr(obj, name))
        else:
            continue
        if M.shape[1] == K:
            return to_pm1(M)

    # 2) trajectories → last time-step
    traj_fields = ['spin_trajectories', 'spin_trajectories_all_runs',
                   'states_trajectories']
    for name in traj_fields:
        if isinstance(obj, dict) and name in obj:
            T = np.asarray(obj[name])
        elif hasattr(obj, name):
            T = np.asarray(getattr(obj, name))
        else:
            continue
        if T.ndim == 3 and T.shape[-1] == K:
            return to_pm1(T[:, -1, :])

    # 3) single best configuration
    single = ['lowest_energy_spin_configuration', 'lowest_energy_spin_config',
              'lowest_energy_state', 'state', 'spins']
    for name in single:
        if isinstance(obj, dict) and name in obj:
            M = as_mat(obj[name])
        elif hasattr(obj, name):
            M = as_mat(getattr(obj, name))
        else:
            continue
        if M.shape[1] == K:
            return to_pm1(M)

    # 4) brute scan for any (R×K) ndarray with values in {−1,0,1}
    if hasattr(obj, '__dict__'):
        for v in obj.__dict__.values():
            if isinstance(v, np.ndarray):
                M = as_mat(v)
                if M.shape[1] == K and set(np.unique(np.rint(M))) <= {-1, 0, 1}:
                    return to_pm1(M)

    raise RuntimeError("Per-run spins not found in CIM result.")

# -------------------------  CIM helpers (patched) -------------------------
def cim_run_batch(solver: Ising, K: int, G_int: np.ndarray,
                  params: dict, n_runs: int, seed: int):
    """
    Run `n_runs` trajectories with a deterministic NumPy seed.
    Returns lists of squared norms and norms.
    """
    # 1) set seed locally
    old_state = np.random.get_state()
    np.random.seed(seed)

    try:
        res = solver.solve(
            num_timesteps_per_run=params["steps"],
            cac_time_step=params["cac_dt"],
            cac_r=params["r"],
            cac_mu=params["mu"],
            cac_noise=params["noise"],
            num_runs=n_runs,
            use_CAC=True,
            use_GPU=False,
            suppress_statements=True,
        )
    finally:
        # 2) restore caller’s RNG state
        np.random.set_state(old_state)

    spins = extract_spins(res, K)
    sqs   = [sq_from_spins(s, G_int) for s in spins]
    norms = [sqrt_int(sq) for sq in sqs]
    return sqs, norms
    
    
# ----------------------------  experiment grid -------------------------
GRID_DT    = [0.01, 0.03, 0.05, 0.07]
GRID_MU    = [0.1,  0.3,  0.5,  0.7]
GRID_R     = [0.05, 0.2,  0.4,  0.6]
GRID_NOISE = [0.0,  0.05, 0.10, 0.15]

RUNS_PER_SETTING = 100
STEPS            = 600
RAND_SCALING     = 1.0
BASE_SEED        = 12345

# ----------------------------  per-X routine ---------------------------
def process_one_X(X: int, B_sorted: np.ndarray, out_dir: Path):
    subB   = B_sorted[:X]
    G_int  = gram_int(subB)
    J      = build_J(subB, RAND_SCALING)
    solver = Ising(J=J, h=np.zeros(X, dtype=np.float64))

    # true optimum via enumeration
    best_sq = min(
        sq_from_spins(np.array([(bits >> i) & 1 and 1 or -1 for i in range(X)]),
                      G_int)
        for bits in range(1 << X)
    )
    opt_norm = sqrt_int(best_sq)

    records = []
    grid = list(itertools.product(GRID_DT, GRID_MU, GRID_R, GRID_NOISE))
    bar = tqdm(grid, desc=f"X={X}", leave=False)

    for idx, (dt, mu, r, noise) in enumerate(bar):
        params = dict(cac_dt=dt, mu=mu, r=r, noise=noise, steps=STEPS)
        seed   = BASE_SEED + X*1000 + idx
        sqs, norms = cim_run_batch(solver, X, G_int,
                                   params, RUNS_PER_SETTING, seed)
        mean_norm = float(np.mean(norms))
        p_opt     = float(np.mean([sq == best_sq for sq in sqs]))
        eff       = mean_norm / opt_norm

        records.append(dict(
            X=X, cac_dt=dt, mu=mu, r=r, noise=noise,
            mean_norm=mean_norm, eff=eff, p_opt=p_opt
        ))

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_dir / f"sweep_X{X}.csv", index=False)

    # quick heat-map (Δt × μ slice  at  r=0.4, σ=0.10)
    slice_df = df[(df.r == 0.4) & (df.noise == 0.10)]
    if not slice_df.empty:
        pivot = slice_df.pivot(index="cac_dt", columns="mu", values="eff")
        plt.figure(figsize=(4, 3))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis_r")
        plt.title(f"efficiency  X={X}  (r=0.4, σ=0.10)")
        plt.xlabel("mu"); plt.ylabel("Δt")
        plt.tight_layout()
        plt.savefig(out_dir / f"eff_heatmap_X{X}.png", dpi=160)
        plt.close()

    return df

# ----------------------------  main ------------------------------------
def main():
    DIM, SEED = 50, 17
    B = load_local_lattice(DIM, SEED)

    # order basis rows by actual norm
    row_norms = np.array([vec_sq_norm_int(row) for row in B], dtype=object)
    perm      = np.argsort([int(x) for x in row_norms])
    B_sorted  = B[perm]

    ts   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = Path(f"CIM_sensitivity_results_{ts}")
    root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing results to  {root.resolve()}\n")

    dims = list(range(3, 11))
    dfs  = []
    for X in tqdm(dims, desc="Dimensions"):
        df = process_one_X(X, B_sorted, root)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(root / "sweep_all_dims.csv", index=False)
    print("\n[DONE] CSVs and heat-maps saved.")

if __name__ == "__main__":
    main()
