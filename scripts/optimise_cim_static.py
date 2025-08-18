#!/usr/bin/env python3
"""
optimise_cim_static.py
======================

Run one Optuna study (CAC only) for a sub-dimension X.

Called by sweep_cim_static_dims.py, which passes:
    --dim   X
    --jobs  <cores>
    --root  <ATTEMPT-folder>

Key points
• Reference optimum comes from cim_svp.extras.classical_solvers
  – exact enumeration up to enum_thresh
  – Monte-Carlo afterwards.
• Works regardless of current working directory:
  lattice folder resolved relative to project root.
• Accepts both dim50_seed17.txt and dim50_17.txt.
"""
# ───── stdlib ─────────────────────────────────────────────────────────
import os, sys, math, argparse, datetime, itertools, json, re
from pathlib import Path
from decimal import getcontext

# ───── third-party ───────────────────────────────────────────────────

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from joblib import Parallel, delayed

# ───── project helpers ───────────────────────────────────────────────
from cim_optimizer.solve_Ising import Ising
from cim_svp import (
    load_lattice, vec_sq_norm_int, gram_int, build_J,
    sq_from_spins, sqrt_int, extract_spins, make_seed
)
from cim_svp.extras.classical_solvers import (
    shortest_enum, shortest_monte_carlo          # ← reference solvers
)

# ───── constants ─────────────────────────────────────────────────────
getcontext().prec = 120
GLOBAL_SEED  = 2025
EVAL_RUNS    = 1_000
DIM_FULL     = 50
LATTICE_SEED = 17

# ---------- absolute lattice folder ----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATTICE_DIR  = PROJECT_ROOT / "svp_lattices"

# ═════════════ flexible lattice loader ═══════════════════════════════
def load_lattice_flex(dim: int, seed: int) -> np.ndarray:
    """Try dimXX_seedYY.txt, else dimXX_YY.txt (both under LATTICE_DIR)."""
    try:
        return load_lattice(dim, seed, folder=str(LATTICE_DIR))
    except FileNotFoundError:
        alt = LATTICE_DIR / f"dim{dim}_{seed}.txt"
        if not alt.is_file():
            raise
        rows = [[int(x) for x in re.findall(r"-?\\d+", ln)]
                for ln in alt.read_text().splitlines()
                if re.search(r"\\d", ln)]
        M = np.array(rows, dtype=object)
        if M.shape != (dim, dim):
            raise ValueError(f"{alt} shape {M.shape}, expected ({dim},{dim})")
        return M

# ═════════════ reference optimum via classical_solvers ═══════════════
def reference_optimum(B: np.ndarray,
                      enum_thresh: int, mc_samples: int,
                      seed: int | None) -> float:
    K = B.shape[0]
    if K <= enum_thresh:
        _vec, norm = shortest_enum(B, max_dim=enum_thresh)
    else:
        _vec, norm = shortest_monte_carlo(B, samples=mc_samples, seed=seed)
    return norm

# ═════════════ helper I/O (cast first) ═══════════════════════════════
def save_hist(norms, opt_norm, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    if norms.size == 0:
        return
    uniq = np.unique(norms[~np.isinf(norms)])
    bins = min(40, max(1, uniq.size - 1))
    plt.figure(figsize=(6, 4))
    if bins > 1:
        sns.histplot(norms, bins=bins, color="steelblue")
    else:
        plt.axvline(norms[0], lw=6, color="steelblue")
    plt.axvline(opt_norm, color="red", ls="--", label="shortest")
    plt.xlabel("norm"); plt.ylabel("count"); plt.legend(); plt.tight_layout()
    plt.savefig(folder / "norm_hist.png", dpi=150)
    plt.close()

def safe_save(root: Path, study, opt_norm, norms):
    try:
        (root/"data").mkdir(parents=True, exist_ok=True)
        norms = np.asarray(norms, dtype=float)          # cast first
        save_hist(norms, opt_norm, root/"plots")
        pd.DataFrame({"norm": norms}).to_csv(root/"data"/"spin_norms.csv",
                                            index=False)
        study.trials_dataframe().to_csv(root/"data"/"trials.csv", index=False)
        json.dump({"best_parameters": study.best_trial.params,
                   "cmd": " ".join(sys.argv),
                   "finished": datetime.datetime.now().isoformat()},
                  open(root/"README.json","w"), indent=2)
    except Exception as e:
        print("[WARN] saving failed:", e, flush=True)

# ───── Optuna callbacks ──────────────────────────────────────────────
class StopOnPlateau:
    def __init__(self, patience): self.best=math.inf; self.cnt=0; self.p=patience
    def __call__(self, st, tr):
        if tr.value < self.best - 1e-12: self.best, self.cnt = tr.value, 0
        else:
            self.cnt += 1
            if self.cnt >= self.p: print("[EARLY-STOP]"); st.stop()
def StopIfEfficiency(t=0.99):
    return lambda st,tr: st.stop() if tr.value >= t else None

# ═════════════════════════════ main ══════════════════════════════════
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dim",  type=int, required=True)
    pa.add_argument("--root", type=str,  required=True)
    pa.add_argument("--jobs", type=int,  default=1)
    pa.add_argument("--n-trials",   type=int, default=400)
    pa.add_argument("--patience",   type=int, default=50)
    pa.add_argument("--enum-thresh",type=int, default=12)
    pa.add_argument("--mc-samples", type=int, default=200_000)
    args = pa.parse_args()

    for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(args.jobs)

    root = Path(args.root) / f"X{args.dim:02d}"
    (root/"data").mkdir(parents=True, exist_ok=True)
    print(f"[INFO] results → {root}", flush=True)

    # ---------- load lattice ----------------------------------------
    B50 = load_lattice_flex(DIM_FULL, LATTICE_SEED)
    B   = B50[np.argsort([int(vec_sq_norm_int(v)) for v in B50])
             ][:args.dim]

    G = gram_int(B);  J = build_J(B)
    solver = Ising(J=J, h=np.zeros(args.dim))

    print("[INFO] computing reference optimum …", flush=True)
    opt_norm = reference_optimum(B,
                                 enum_thresh=args.enum_thresh,
                                 mc_samples=args.mc_samples,
                                 seed=make_seed("ref", args.dim))
    print(f"[INFO] |b*| = {opt_norm:.4g}", flush=True)

    study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True,
                                           seed=GLOBAL_SEED + args.dim,
                                           n_startup_trials=30),
        pruner=optuna.pruners.HyperbandPruner(min_resource=400,
                                              max_resource=1000,
                                              reduction_factor=3))

    def objective(trial):
        hp = dict(
            dt    = trial.suggest_float("dt",    0.01, 0.07),
            mu    = trial.suggest_float("mu",    0.05, 0.8),
            r     = trial.suggest_float("r",     0.05, 0.6),
            noise = trial.suggest_float("noise", 1e-4, 0.30, log=True),
            steps = trial.suggest_int  ("steps", 400,  1000, step=100),
        )
        runs  = 20 if trial.number < 50 else 100
        seeds = np.random.randint(2**31 - 1, size=runs)

        def one(sd):
            np.random.seed(sd)
            res = solver.solve(num_timesteps_per_run=hp["steps"],
                               cac_time_step=hp["dt"],
                               cac_mu=hp["mu"], cac_r=hp["r"],
                               cac_noise=hp["noise"],
                               num_runs=1, use_CAC=True, use_GPU=False,
                               suppress_statements=True)
            s = extract_spins(res, args.dim)[0]
            return sqrt_int(sq_from_spins(s, G))

        norms = Parallel(n_jobs=args.jobs)(delayed(one)(sd) for sd in seeds)
        return float(opt_norm / np.mean(norms))

    study.optimize(objective, n_trials=args.n_trials, n_jobs=1,
                   show_progress_bar=(args.jobs == 1),
                   callbacks=[StopOnPlateau(args.patience),
                              StopIfEfficiency(0.99)])

    # ---------- final evaluation ------------------------------------
    best  = study.best_trial.params
    seeds = np.random.randint(2**31 - 1, size=EVAL_RUNS)

    def run_eval(sd):
        np.random.seed(sd)
        res = solver.solve(num_timesteps_per_run=best["steps"],
                           cac_time_step=best["dt"],
                           cac_mu=best["mu"], cac_r=best["r"],
                           cac_noise=best["noise"],
                           num_runs=1, use_CAC=True, use_GPU=False,
                           suppress_statements=True)
        s = extract_spins(res, args.dim)[0]
        return sqrt_int(sq_from_spins(s, G))

    norms = Parallel(n_jobs=args.jobs)(delayed(run_eval)(sd) for sd in seeds)
    safe_save(root, study, opt_norm, norms)
    print(f"[DONE] {root}", flush=True)


if __name__ == "__main__":
    main()
