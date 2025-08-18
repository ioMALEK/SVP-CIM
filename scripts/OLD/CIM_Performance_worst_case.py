#!/usr/bin/env python3
"""
CIM_Performance_worst_case.py
Run a Coherent Ising Machine (CIM) experiment for SVP-like subspace search with ±1 spins.

Overnight settings + immediate data storage + initial ETA:
- Correct Ising mapping: minimize s^T G s with J = -G (scaled).
- Safe arithmetic: Gram in Python ints; normalized float64 for solver.
- Exact squared norms via quadratic form s^T G_int s (fast, no vector build).
- HPO budget = 3% with early stop; Evaluation = 97%.
- Batched runs (batch_runs=256); N_JOBS=4 by default.
- All per-X raw norms are written immediately (open_memmap .npy) to results/.../data.
- At start: short calibration to print back-of-envelope ETAs.
"""

import os
# Avoid oversubscription (joblib + possible OpenMP in solver)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
import ast
import re
import time
import contextlib
from decimal import Decimal, getcontext
from joblib import Parallel, delayed
import joblib
from tqdm.auto import tqdm
from numpy.lib.format import open_memmap
from cim_optimizer.solve_Ising import Ising

getcontext().prec = 100

RESULTS_FOLDER = "CIM_performance_worst_case_results"
DATA_SUBFOLDER = "data"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ----------------- IO -----------------
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

def load_shortest_norm(seed, csv_file="svp50_shortest_norms.csv"):
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['seed']) == seed:
                return float(row['lambda1'])
    raise ValueError(f"Seed {seed} not found in {csv_file}")

# ----------------- Helpers -----------------
def to_pm1(arr):
    arr = np.asarray(arr)
    vals = set(np.unique(np.rint(arr)))
    if vals <= {-1, 1}:
        return arr.astype(int)
    if vals <= {0, 1}:
        return (2*arr - 1).astype(int)
    return np.sign(arr).astype(int)

def get_spins_matrix_from_result(result, K):
    """
    Robustly extract an (R, K) spins matrix (±1) from various cim-optimizer versions.
    Includes trajectory fallback (last timestep) if per-run configs are not provided directly.
    """
    obj = getattr(result, 'result', result)  # some versions wrap the payload in .result

    def as_mat(x):
        x = np.asarray(x)
        return x if x.ndim == 2 else x[None, :]

    # 1) Common per-run fields
    names = [
        'spin_configurations_all_runs',
        'spin_config_all_runs',
        'spin_configurations',
        'spins_all_runs',
        'states_all_runs'
    ]
    for name in names:
        if isinstance(obj, dict) and name in obj:
            M = as_mat(obj[name])
            if M.shape[1] == K:
                return to_pm1(M)
        if hasattr(obj, name):
            M = as_mat(getattr(obj, name))
            if isinstance(M, np.ndarray) and M.shape[1] == K:
                return to_pm1(M)

    # 2) Trajectory fallback: (runs, timesteps, K) -> last step
    traj_names = ['spin_trajectories', 'spin_trajectories_all_runs', 'states_trajectories']
    for name in traj_names:
        if isinstance(obj, dict) and name in obj:
            T = np.asarray(obj[name])
            if T.ndim == 3 and T.shape[-1] == K:
                return to_pm1(T[:, -1, :])
        if hasattr(obj, name):
            T = np.asarray(getattr(obj, name))
            if T.ndim == 3 and T.shape[-1] == K:
                return to_pm1(T[:, -1, :])

    # 3) Single best config (replicate to shape (1,K))
    names_single = ['lowest_energy_spin_config', 'lowest_energy_spin_configuration', 'spins', 'state']
    for name in names_single:
        if isinstance(obj, dict) and name in obj:
            M = as_mat(obj[name])
            if M.shape[1] == K:
                return to_pm1(M)
        if hasattr(obj, name):
            M = as_mat(getattr(obj, name))
            if isinstance(M, np.ndarray) and M.shape[1] == K:
                return to_pm1(M)

    # 4) Scan attributes for a (R,K) array with values in {-1,0,1}
    if hasattr(obj, '__dict__'):
        for _, val in obj.__dict__.items():
            if isinstance(val, np.ndarray):
                M = as_mat(val)
                if M.shape[1] == K and set(np.unique(np.rint(M))) <= {-1, 0, 1}:
                    return to_pm1(M)

    raise RuntimeError("Per-run spins not found")

def vec_sq_norm_int(vec):
    return int(sum(int(x) * int(x) for x in vec))

def gram_from_basis_rows_float(B):
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
    max_abs = max(1, max(abs(int(G[i, j])) for i in range(K) for j in range(K)))
    Gf = np.empty((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            Gf[i, j] = float(int(G[i, j]) / max_abs)
    return Gf, max_abs

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

def build_J_from_basis(B, rand_scaling=1.0):
    Gf, _ = gram_from_basis_rows_float(B)
    J = -Gf * float(rand_scaling)
    np.fill_diagonal(J, 0.0)
    return J

def sq_from_spins_with_Gram(spins_row, G_int):
    s = spins_row
    K = s.shape[0]
    total = 0
    for i in range(K):
        total += int(G_int[i, i])  # s_i^2 = 1
        si = int(s[i])
        for j in range(i + 1, K):
            total += 2 * si * int(s[j]) * int(G_int[i, j])
    return int(total)

# ----------------- Progress bar glue for joblib -----------------
class tqdm_joblib:
    def __init__(self, tqdm_obj):
        self.tqdm_obj = tqdm_obj
        self._old_cb = None
    def __enter__(self):
        self._old_cb = joblib.parallel.BatchCompletionCallBack
        tqdm_obj = self.tqdm_obj
        class TqdmBatchCompletionCallback(self._old_cb):
            def __call__(self, *args, **kwargs):
                tqdm_obj.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        return self.tqdm_obj
    def __exit__(self, exc_type, exc_val, exc_tb):
        joblib.parallel.BatchCompletionCallBack = self._old_cb
        self.tqdm_obj.close()

# ----------------- HPO -----------------
def robust_random_hpo_for_X(search_basis, hpo_total_runs=1_200, batch_runs=256, seed=17,
                            show_pbar=False, early_stop_noimp=10, steps_lo=300, steps_hi=800):
    K = search_basis.shape[0]
    h = np.zeros(K, dtype=np.float64)
    J_base = build_J_from_basis(search_basis, rand_scaling=1.0)
    G_int = gram_from_basis_rows_int(search_basis)

    rng = np.random.RandomState(seed)
    n_trials = max(8, int(hpo_total_runs // max(1, batch_runs)))
    best_params, best_score = {}, float('inf')
    hyperparam_history = []
    start = time.time()
    noimp = 0

    iterator = range(n_trials)
    if show_pbar:
        iterator = tqdm(iterator, desc=f"HPO X={K}", leave=False)

    for _ in iterator:
        params = {
            "cac_time_step": float(rng.uniform(0.01, 0.08)),
            "cac_r": float(rng.uniform(0.05, 0.6)),
            "cac_mu": float(rng.uniform(0.05, 0.8)),
            "cac_noise": float(rng.uniform(0.0, 0.15)),
            "steps": int(rng.randint(steps_lo, steps_hi + 1)),
            "rand_scaling": float(rng.choice([0.5, 0.8, 1.0, 1.3])),
            "batch_runs": int(batch_runs),
        }
        try:
            J = J_base * params["rand_scaling"]
            solver = Ising(J=J, h=h)
            with open(os.devnull, 'w') as devnull, \
                 contextlib.redirect_stdout(devnull), \
                 contextlib.redirect_stderr(devnull):
                result = solver.solve(
                    num_timesteps_per_run=params["steps"],
                    cac_time_step=params["cac_time_step"],
                    cac_r=params["cac_r"],
                    cac_mu=params["cac_mu"],
                    cac_noise=params["cac_noise"],
                    num_runs=params["batch_runs"],
                    suppress_statements=True,
                    use_CAC=True,
                    use_GPU=False,
                )
            spins_mat = get_spins_matrix_from_result(result, K)
            sq_best = None
            for i in range(spins_mat.shape[0]):
                srow = spins_mat[i]
                sq = sq_from_spins_with_Gram(srow, G_int)
                if (sq_best is None) or (sq < sq_best):
                    sq_best = sq
            if sq_best is None:
                continue
            score = float(sq_best)
            hyperparam_history.append((params.copy(), score))
            if score + 1e-12 < best_score:
                best_score, best_params, noimp = score, params.copy(), 0
            else:
                noimp += 1
                if noimp >= early_stop_noimp:
                    break
        except Exception:
            continue

    hpo_time = time.time() - start
    return best_params, hyperparam_history, hpo_time

# ----------------- Measure per X -----------------
def cim_hist_for_X(search_basis, n_runs_eval, shortest_norm, results_dir, X, hpo_params,
                   show_pbar=False, batch_runs=256):
    os.makedirs(results_dir, exist_ok=True)
    K, _ = search_basis.shape

    rand_scaling = hpo_params.get('rand_scaling', 1.0)
    J = build_J_from_basis(search_basis, rand_scaling=rand_scaling)
    G_int = gram_from_basis_rows_int(search_basis)
    h = np.zeros(K, dtype=np.float64)
    solver = Ising(J=J, h=h)

    count_shortest = 0
    shortest_sq = int(round(float(shortest_norm) ** 2))

    steps = int(hpo_params.get("steps", 600))
    cac_time_step = float(hpo_params.get("cac_time_step", 0.03))
    cac_r = float(hpo_params.get("cac_r", 0.3))
    cac_mu = float(hpo_params.get("cac_mu", 0.3))
    cac_noise = float(hpo_params.get("cac_noise", 0.05))

    # Create memmap with NPY header so np.load works
    norms_path = os.path.join(results_dir, f"norms_X{X}_runs{n_runs_eval}.npy")
    norms_mm = open_memmap(norms_path, mode='w+', dtype=np.float64, shape=(n_runs_eval,))
    write_idx = 0

    remaining = n_runs_eval
    sim_start = time.time()
    pbar = tqdm(total=n_runs_eval, desc=f"Eval X={X}", leave=False) if show_pbar else None

    while remaining > 0:
        r = min(batch_runs, remaining)
        with open(os.devnull, 'w') as devnull, \
             contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):
            result = solver.solve(
                num_timesteps_per_run=steps,
                cac_time_step=cac_time_step,
                cac_r=cac_r,
                cac_mu=cac_mu,
                cac_noise=cac_noise,
                num_runs=r,
                suppress_statements=True,
                use_CAC=True,
                use_GPU=False,
            )
        try:
            spins_mat = get_spins_matrix_from_result(result, K)
            for i in range(spins_mat.shape[0]):
                if write_idx >= n_runs_eval:
                    break
                srow = spins_mat[i]
                sqn = sq_from_spins_with_Gram(srow, G_int)
                norms_mm[write_idx] = float(Decimal(sqn).sqrt())
                if sqn == shortest_sq:
                    count_shortest += 1
                write_idx += 1
        except Exception:
            # skip failed batch without writing NaNs
            pass

        remaining -= r
        if pbar:
            pbar.update(r)
        norms_mm.flush()

    if pbar:
        pbar.close()

    sim_time = time.time() - sim_start

    arr = np.load(norms_path, mmap_mode='r')
    arr_valid = arr[np.isfinite(arr) & (arr > 0)]
    valid_outputs = int(arr_valid.size)
    fraction_shortest = (count_shortest / valid_outputs) if valid_outputs > 0 else 0.0
    norm_std = float(np.nanstd(arr_valid)) if valid_outputs > 1 else 0.0
    median_norm = float(np.nanmedian(arr_valid)) if valid_outputs > 0 else 0.0

    if valid_outputs > 0:
        min_pos = float(np.min(arr_valid))
        max_v = float(np.max(arr_valid))
        bins = np.logspace(np.log10(min_pos), np.log10(max_v), 200)
        counts, edges = np.histogram(arr_valid, bins=bins)
        probs = counts / float(valid_outputs)
    else:
        bins = np.logspace(0, 1, 10)
        counts = np.zeros(len(bins)-1, dtype=int)
        edges = bins
        probs = counts.astype(float)

    hist_txt = os.path.join(results_dir, f"histogram_X{X}_runs{n_runs_eval}.txt")
    with open(hist_txt, "w", encoding="utf-8") as f:
        f.write("bin_left,bin_right,count,probability\n")
        for i in range(len(counts)):
            f.write(f"{edges[i]},{edges[i+1]},{int(counts[i])},{probs[i]:.12g}\n")
        f.flush()
        os.fsync(f.fileno())

    average_basis_norm = float(np.mean([
        float(Decimal(vec_sq_norm_int(search_basis[i])).sqrt()) for i in range(K)
    ])) if K > 0 else 0.0

    plt.figure(figsize=(10, 6))
    if valid_outputs > 0:
        plt.hist(arr_valid, bins=bins, alpha=0.85, color='royalblue')
        plt.xscale('log')
    else:
        plt.hist([], bins=np.logspace(0, 1, 10))
        plt.xscale('log')
    plt.axvline(float(shortest_norm), color='r', linestyle='--',
                label=f"Shortest norm: {shortest_norm:.2g}", linewidth=2.0)
    plt.xlabel("CIM output norm (log scale)")
    plt.ylabel("Count")
    plt.title(f"CIM Output Norms (X={X}, runs={n_runs_eval:,})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"histogram_X{X}_runs{n_runs_eval}.pdf"))
    plt.savefig(os.path.join(results_dir, f"histogram_X{X}_runs{n_runs_eval}.png"), dpi=160)
    plt.close()

    return {
        'X': X,
        'count_shortest': count_shortest,
        'fraction_shortest': fraction_shortest,
        'average_basis_norm': average_basis_norm,
        'std_norm': norm_std,
        'median_norm': median_norm,
        'hist_file': hist_txt,
        'norms_file': norms_path,
        'sim_time': sim_time,
        'valid_outputs': valid_outputs
    }

# ----------------- Calibration -----------------
def calibrate_runs_per_second(B_sorted, steps=600, batch_runs=256, rand_scaling=1.0):
    Xc = min(10, B_sorted.shape[0])
    subB = B_sorted[:Xc]
    h = np.zeros(Xc, dtype=np.float64)
    J = build_J_from_basis(subB, rand_scaling=rand_scaling)
    solver = Ising(J=J, h=h)
    # warmup
    with open(os.devnull, 'w') as devnull, \
         contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull):
        solver.solve(
            num_timesteps_per_run=steps,
            cac_time_step=0.03, cac_r=0.3, cac_mu=0.3, cac_noise=0.05,
            num_runs=batch_runs, suppress_statements=True, use_CAC=True, use_GPU=False,
        )
    # timed
    t0 = time.time()
    with open(os.devnull, 'w') as devnull, \
         contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull):
        solver.solve(
            num_timesteps_per_run=steps,
            cac_time_step=0.03, cac_r=0.3, cac_mu=0.3, cac_noise=0.05,
            num_runs=batch_runs, suppress_statements=True, use_CAC=True, use_GPU=False,
        )
    t1 = time.time()
    elapsed = max(1e-9, t1 - t0)
    rps = batch_runs / elapsed
    return rps

# ----------------- Main -----------------
def main():
    # ---------- User-configurable ----------
    dim = 50
    seed = 17
    total_runs_per_X = 40_000
    hpo_fraction = 0.03
    hpo_batch_runs = 256
    N_JOBS = 4
    steps_lo, steps_hi = 300, 800
    # --------------------------------------

    now = datetime.datetime.now()
    results_dir = os.path.join(RESULTS_FOLDER, now.strftime("%Y-%m-%d_%H-%M-%S"))
    data_dir = os.path.join(results_dir, DATA_SUBFOLDER)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    print(f"Data will be written immediately to: {data_dir}")

    B = load_local_lattice(dim, seed)
    shortest_norm = load_shortest_norm(seed)
    print(f"Shortest vector norm for seed {seed}: {shortest_norm:.2f}")

    basis_sq_norms = np.array([vec_sq_norm_int(B[i]) for i in range(B.shape[0])], dtype=object)
    perm = np.argsort(np.array([int(x) for x in basis_sq_norms]))
    B_sorted = B[perm]
    X_vals = list(range(2, dim + 1))

    print("Calibrating runs/sec (this takes ~a few seconds)...")
    calib_rps = calibrate_runs_per_second(B_sorted, steps=600, batch_runs=256)
    total_runs_allX = int(total_runs_per_X * len(X_vals))
    est_seconds_total = total_runs_allX / max(1e-9, calib_rps * max(1, N_JOBS))
    est_hours_total = est_seconds_total / 3600.0
    est_hours_per_X = (total_runs_per_X / max(1e-9, calib_rps)) / 3600.0
    print(f"[ETA] Calibration: ~{calib_rps:.1f} runs/sec/worker. With N_JOBS={N_JOBS}: "
          f"~{calib_rps*N_JOBS:.1f} runs/sec total.")
    print(f"[ETA] Back-of-envelope time per X (incl. HPO fraction ~{hpo_fraction*100:.0f}%): "
          f"~{est_hours_per_X:.2f} h/X")
    print(f"[ETA] Back-of-envelope total for {len(X_vals)} X's: ~{est_hours_total:.2f} h "
          f"(target ~12 h). This will refine as we run.\n")

    def run_one_X(idx, X, show_worker_pbars):
        hpo_total_runs = int(hpo_fraction * total_runs_per_X)
        n_runs_eval = total_runs_per_X - hpo_total_runs

        best_params, hpo_trace, hpo_time = robust_random_hpo_for_X(
            B_sorted[:X], hpo_total_runs=hpo_total_runs, batch_runs=hpo_batch_runs,
            seed=seed + X, show_pbar=show_worker_pbars, early_stop_noimp=10,
            steps_lo=steps_lo, steps_hi=steps_hi
        )
        hpo_evo_path = os.path.join(data_dir, f"hyperparams_X{X}.txt")
        with open(hpo_evo_path, "w", encoding="utf-8") as f:
            f.write("trial,cac_time_step,cac_r,cac_mu,cac_noise,steps,rand_scaling,batch_runs,best_sq_norm\n")
            for i, (params, score) in enumerate(hpo_trace):
                f.write(f"{i},{params['cac_time_step']},{params['cac_r']},{params['cac_mu']},"
                        f"{params['cac_noise']},{params['steps']},{params['rand_scaling']},"
                        f"{params['batch_runs']},{score}\n")
            f.flush()
            os.fsync(f.fileno())

        rec = cim_hist_for_X(B_sorted[:X], n_runs_eval, shortest_norm, data_dir, X, best_params,
                             show_pbar=show_worker_pbars, batch_runs=hpo_batch_runs)
        rec['hpo_time'] = hpo_time
        rec['n_runs_eval'] = n_runs_eval
        rec['hpo_runs'] = hpo_total_runs
        return rec

    records = []
    if N_JOBS == 1:
        for idx, X in enumerate(tqdm(X_vals, desc="X jobs")):
            rec = run_one_X(idx, X, show_worker_pbars=True)
            records.append(rec)
    else:
        with tqdm_joblib(tqdm(total=len(X_vals), desc="X jobs")):
            records = Parallel(n_jobs=N_JOBS)(
                delayed(run_one_X)(idx, X, False) for idx, X in enumerate(X_vals)
            )

    out_txt = os.path.join(data_dir, "CIM_X_scan_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("X,valid_outputs,hpo_runs,n_runs_eval,count_shortest,fraction_shortest,average_basis_norm,std_norm,median_norm,sim_time_sec\n")
        for rec in records:
            f.write(f"{rec['X']},{rec['valid_outputs']},{rec['hpo_runs']},{rec['n_runs_eval']},"
                    f"{rec['count_shortest']},{rec['fraction_shortest']},"
                    f"{rec['average_basis_norm']},{rec['std_norm']},{rec['median_norm']},"
                    f"{rec['sim_time']}\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"Summary saved to {out_txt}")

    X_arr = np.array([rec['X'] for rec in records])

    def save_plot(x, y, xlabel, ylabel, title, fname, color='C0'):
        plt.figure()
        plt.plot(x, y, 'o-', color=color)
        plt.xscale('log')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, fname + ".pdf"))
        plt.savefig(os.path.join(results_dir, fname + ".png"), dpi=160)
        plt.close()

    save_plot(X_arr, [rec['valid_outputs'] for rec in records],
              "X (log scale)", "Valid outputs used",
              "Valid outputs per X", "valid_outputs_vs_X")

    save_plot(X_arr, [rec['fraction_shortest'] for rec in records],
              "X (log scale)", "P(v has shortest norm)",
              "Fraction of shortest outputs vs. X", "fraction_shortest_vs_X")

    save_plot(X_arr, [rec['median_norm'] for rec in records],
              "X (log scale)", "Median output norm",
              "Median output norm vs. X", "median_norm_vs_X", color='purple')

    save_plot(X_arr, [rec['std_norm'] for rec in records],
              "X (log scale)", "Std of output norms",
              "Std of output norms vs. X", "std_norm_vs_X")

    save_plot(X_arr, [rec['sim_time'] for rec in records],
              "X (log scale)", "CIM time per X (s)",
              "Time for CIM runs per X", "sim_time_vs_X")
    save_plot(X_arr, [rec['hpo_time'] for rec in records],
              "X (log scale)", "HPO time per X (s)",
              "Time for HPO per X", "hpo_time_vs_X")

    global_min, global_max = None, None
    for rec in records:
        arr = np.load(rec['norms_file'], mmap_mode='r')
        if arr.size == 0:
            continue
        arr_pos = arr[np.isfinite(arr) & (arr > 0)]
        if arr_pos.size == 0:
            continue
        mpos = float(np.min(arr_pos))
        mx = float(np.max(arr_pos))
        global_min = mpos if (global_min is None) else min(global_min, mpos)
        global_max = mx if (global_max is None) else max(global_max, mx)

    if (global_min is not None) and (global_max is not None) and (global_max > 0):
        bins = np.logspace(np.log10(global_min), np.log10(global_max), 200)
        counts_all = np.zeros(len(bins)-1, dtype=np.int64)
        for rec in records:
            arr = np.load(rec['norms_file'], mmap_mode='r')
            arr_pos = arr[np.isfinite(arr) & (arr > 0)]
            if arr_pos.size == 0:
                continue
            c, _ = np.histogram(arr_pos, bins=bins)
            counts_all += c
        plt.figure()
        plt.bar((bins[:-1]*bins[1:])**0.5, counts_all, width=np.diff(bins),
                align='center', alpha=0.8, color='steelblue', edgecolor='none')
        plt.xscale('log')
        plt.xlabel("CIM output norm (log scale)")
        plt.ylabel("Count across all X")
        plt.title("Distribution of CIM output norms across all X")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "unique_norms_histogram_all_X.pdf"))
        plt.savefig(os.path.join(results_dir, "unique_norms_histogram_all_X.png"), dpi=160)
        plt.close()

    print("All plots and data saved.")

if __name__ == "__main__":
    main()
