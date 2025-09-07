#!/usr/bin/env python3
# ======================================================================
#  Batch_Solve_CPM.py      (Benchmark-6 · Basis-Reduction Iteration)
#  – ΣCPM for a BRI = largest ΣCPM among its five batches
#  – ηCPM printed / logged is the one belonging to that ΣCPM
#  – γ printed with five decimals so slow drift is visible
#  – quick-test = 5 BRIs (25 batches)  →  export QUICK_TEST=1
# ======================================================================

from __future__ import annotations
import argparse, json, math, os, shutil, sys, time, warnings
from collections import deque
from pathlib  import Path
from typing    import List, Dict

import numpy as np, optuna, torch
from joblib import Parallel, delayed

from cim_svp.io      import load_lattice
from cim_svp.maths    import vec_sq_norm_int, sqrt_int
from cim_svp.cpmwrap  import build_J, spins_to_vector
from cim_svp.extras.classical_solvers import shortest_enum
from cim_optimizer.CAC_Potts          import CIM_CAC_Potts_GPU

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"numpy\.core\._methods")

# ───────────────────────── CONFIG ────────────────────────────────────
CONFIG: Dict = dict(
    RNG_SEED          = 42,
    MAX_OUTER         = 10000,      # 24 BRIs (5 batches each)
    N_JOBS            = 5,

    T_TIME            = 400,
    DEVICE            = "cpu",
    BATCH_SIZE        = 1,

    BASE_TRIALS_Q     = {2: 100, 3: 100, 4: 50, 5: 50, 6: 15, 7: 15, 8: 15},
    BASE_EVAL_PER_TR  = 30,
    BASE_M_EVAL_INC   = 150,

    SMALL_STEP        = 1.0003,
    BIG_STEP          = 1.02,
    NO_IMPROV_BIG     = 20,
 
    GAMMA_START       = {2: 0.061, 3: 0.10, 4: 0.16, 5: 0.24, 6: 0.32, 7: 0.40,  8: 0.50},
    
    S_REPL= 5,
    S_NORM= 5,
        
    Q_CYCLE_MAX       = 8,

    DT_RANGE          = (0.02, 0.10),
    PUMP_RANGE        = (0.60, 0.83),
    BETA_RANGE        = (0.10, 0.40),
    NOISE_RANGE       = (0.0,  0.02),
)

# quick-test  (≈ 5 BRIs)   ──   export QUICK_TEST=1
if os.getenv("QUICK_TEST") == "1":
    CONFIG.update(
        MAX_OUTER        = 25,         # 5 BRIs
        T_TIME           = 150,
        BASE_TRIALS_Q    = {q: 5 for q in range(2,7)},
        BASE_EVAL_PER_TR = 1,
        BASE_M_EVAL_INC  = 10,
        N_JOBS           = 2,
    )

# ────────────────────── helpers ──────────────────────────────────────
def is_degenerate(n1:int, n2:int) -> bool:
    return n1 == n2 or (n1 > n2 and n1 % n2 == 0) or (n2 > n1 and n2 % n1 == 0)

def vec_hash(v) -> int:          # works for list or ndarray[int]
    return hash(tuple(int(x) for x in v))

def prepare_attempt_dir(root:Path, attempt:int|None) -> Path:
    root.mkdir(exist_ok=True)
    if attempt is None:
        nums = [int(p.name.split("_")[1]) for p in root.glob("ATTEMPT_*")]
        attempt = max(nums)+1 if nums else 1
    dest = root/f"ATTEMPT_{attempt}"; dest.mkdir()
    (dest/"PER_BATCH").mkdir();  (dest/"PER_BRI").mkdir()
    shutil.copy(__file__, dest/f"Batch_Solve_CPM_ATTEMPT_{attempt}.py")
    with (dest/"run_config.json").open("w") as fp:
        json.dump(CONFIG, fp, indent=2)
    return dest

# ═════════════════════ worker ════════════════════════════════════════
def optimise_batch(bid:int, W:np.ndarray, Q:int, γ:float,
                   Nt:int, Ne:int, Me:int) -> dict:
    optuna.logging.disable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    dev = torch.device(CONFIG["DEVICE"])

    # enumeration (diagnostics, not timed)
    _, λ_Q = shortest_enum(W, Q=Q)
    _, λ_1 = shortest_enum(W, Q=1)
    minW   = min(sqrt_int(vec_sq_norm_int(v)) for v in W)

    # timed section
    t0 = time.time()
    J_W = build_J(W)

    θ_list: List[dict] = []
    def objective(tr):
        p = dict(dt   = tr.suggest_float("dt",   *CONFIG["DT_RANGE"]),
                 pump = tr.suggest_float("pump", *CONFIG["PUMP_RANGE"]),
                 beta = tr.suggest_float("beta", *CONFIG["BETA_RANGE"]),
                 noise= tr.suggest_float("noise",*CONFIG["NOISE_RANGE"]))
        best = math.inf
        for _ in range(Ne):
            s,_,_ = CIM_CAC_Potts_GPU(
                CONFIG["T_TIME"], J_W, Q=Q, batch_size=CONFIG["BATCH_SIZE"],
                time_step=p["dt"], r=p["pump"], beta=p["beta"],
                phase_lock=γ, noise=p["noise"], device=dev)
            best = min(best,
                       sqrt_int(vec_sq_norm_int(spins_to_vector(s[0], W))))
        if best < getattr(objective,"_best",math.inf):
            θ_list.append(p.copy()); objective._best = best
        return best

    optuna.create_study(direction="minimize",
                        sampler=optuna.samplers.TPESampler(seed=CONFIG["RNG_SEED"])
                       ).optimize(objective, n_trials=Nt, show_progress_bar=False)

    cand: List[np.ndarray] = []
    for p in θ_list:
        for _ in range(Me):
            s,_,_ = CIM_CAC_Potts_GPU(
                CONFIG["T_TIME"], J_W, Q=Q, batch_size=CONFIG["BATCH_SIZE"],
                time_step=p["dt"], r=p["pump"], beta=p["beta"],
                phase_lock=γ, noise=p["noise"], device=dev)
            v = spins_to_vector(s[0], W)
            if vec_sq_norm_int(v): cand.append(v)

    surv, sqs = [], []
    for v in cand:
        n2 = vec_sq_norm_int(v)
        if any(is_degenerate(n2,m2) for m2 in sqs): continue
        surv.append(v); sqs.append(n2)

    merged = list(zip(sqs,surv)) + [(vec_sq_norm_int(v),v) for v in W]
    merged.sort(key=lambda t:t[0])
    best_norm = sqrt_int(merged[0][0])
    top10     = [v for _,v in merged[:10]]

    η_CIM = λ_1 / best_norm
    η_CPM = λ_Q / best_norm
    Σ_CPM = abs(minW - best_norm)
    best_norm= min(minW,best_norm)

    return dict(batch_id=bid, cand=top10, R=len(surv),
                deg_removed=len(cand)-len(surv),
                best_norm=best_norm,
                eta_CIM=float(η_CIM),
                eta_CPM=float(η_CPM),
                Sigma_CPM=float(Σ_CPM),
                wall_time=round(time.time()-t0,2))

# ═════════════════════ master ═══════════════════════════════════════
def main(argv):
    optuna.logging.disable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    ap = argparse.ArgumentParser(); ap.add_argument("--attempt", type=int)
    ATT = prepare_attempt_dir(Path("FINAL_CPM"), ap.parse_args(argv).attempt)
    (ATT/"Deg_count.csv").write_text("BRI,deg_removed\n")

    PB_DIR, PR_DIR = ATT/"PER_BATCH", ATT/"PER_BRI"
    batch_hdr=("outer,batch,Q,gamma,R,best_norm,eta_CIM,eta_CPM,"
               "Sigma_CPM,wall_time\n")
    bri_hdr  =("BRI,Q,gamma,global_best,mean_pool,replacements,"
               "eta_CIM_max,eta_CPM_max,Sigma_CPM_max,BRI_time\n")

    rng = np.random.default_rng(CONFIG["RNG_SEED"])
    B   = load_lattice(50, seed=17).astype(object).tolist();  Bp=[]
    outer, blk_idx = 0, -1
    γ, Q = CONFIG["GAMMA_START"][2], 2
    global_best = math.inf
    repl_hist, norm_hist = deque(maxlen=CONFIG["S_REPL"]), deque(maxlen=CONFIG["S_NORM"])

    while outer < CONFIG["MAX_OUTER"]:
    
        Nt = CONFIG["BASE_TRIALS_Q"][Q] * Q
        Ne = CONFIG["BASE_EVAL_PER_TR"] * Q
        Me = CONFIG["BASE_M_EVAL_INC"] * Q
            
        cur_bri, tot_bri = outer//5, CONFIG["MAX_OUTER"]//5
        print(f"BRI {cur_bri} – sim {cur_bri}/{tot_bri}  ({100*cur_bri/tot_bri:4.1f}% complete)")

        orig_hashes = {vec_hash(v) for v in B}

        idx_perm = rng.permutation(len(B))
        groups   = [idx_perm[i*10:(i+1)*10].tolist() for i in range(5)]
        batches  = [[B[i] for i in g] for g in groups]
        for idx in sorted(idx_perm.tolist(),reverse=True): B.pop(idx)

        t_bri = time.time()
        res = Parallel(n_jobs=CONFIG["N_JOBS"], backend="loky")(
            delayed(optimise_batch)(
                bid, np.stack(batches[bid],dtype=object),
                Q,γ,int(Nt),int(Ne),int(Me)) for bid in range(5)
        )

        for r in res:
            global_best = min(global_best, r["best_norm"])
            (PB_DIR/f"batch_{outer:05d}_{r['batch_id']}.csv").write_text(
                batch_hdr+ ",".join(map(str,[
                    outer,r['batch_id'],Q,f"{γ:.5f}",r['R'],r['best_norm'],
                    r['eta_CIM'],r['eta_CPM'],r['Sigma_CPM'],r['wall_time']]))+"\n")
            Bp.extend(r["cand"])

        new_hashes   = {vec_hash(v) for v in Bp}
        replacements = len(new_hashes - orig_hashes)

        total_R   = sum(r["R"] for r in res)
        deg_this  = sum(r["deg_removed"] for r in res)
        with (ATT/"Deg_count.csv").open("a") as fp:
            fp.write(f"{cur_bri},{deg_this}\n")

        # γ feedback
        if replacements > 45 :
            γ*= 1.002
        
        if replacements < 41 :
            γ*= 0.95

        repl_hist.append(total_R); norm_hist.append(global_best)
        stalled = replacements < 10
        print(f"replacements check = {replacements} -> STALLED = {stalled}")
        if stalled:
            Q = Q+1 if (Q<8) else 2
            γ = CONFIG["GAMMA_START"][Q]; repl_hist.clear(); norm_hist.clear()
            print(f"[STALL] → Q={Q}, gamma reset to {γ}")
        
        mean_pool = np.mean([sqrt_int(vec_sq_norm_int(v)) for v in Bp])
        if not B: B,Bp = Bp,[]

        # choose batch with largest ΣCPM
        pairs = [(r["Sigma_CPM"], r["eta_CPM"], r["eta_CIM"])
                 for r in res if math.isfinite(r["Sigma_CPM"])]
        if pairs:
            Sigma_CPM_max, eta_CPM_max, eta_CIM_of_max = max(pairs, key=lambda t: t[0])
            eta_CIM_max = eta_CIM_of_max
        else:
            Sigma_CPM_max = eta_CPM_max = eta_CIM_max = float("nan")

        BRI_time = round(time.time()-t_bri,2)
        (PR_DIR/f"bri_{cur_bri:05d}.csv").write_text(
            bri_hdr+ ",".join(map(str,[
                cur_bri,Q,f"{γ:.5f}",global_best,mean_pool,replacements,
                eta_CIM_max,eta_CPM_max,Sigma_CPM_max,BRI_time]))+"\n")

        # three-line banner
        print(f"best norm so far = {global_best:.2e} | Q = {Q} | gamma = {γ:.5f}")
        print(f"replacements = {replacements} | BRI time = {BRI_time}s")
        print(f"eta_CPM = {eta_CPM_max:.3g} | eta_CIM = {eta_CIM_max:.3g} | "
              f"Sigma_CPM = {Sigma_CPM_max:.3g}\n")

        outer += 5

    # average degeneracy
    deg_vals=[int(l.split(",")[1]) for l in (ATT/"Deg_count.csv").read_text().splitlines()[1:]]
    with (ATT/"Deg_count.csv").open("a") as fp:
        fp.write(f"average,{sum(deg_vals)/len(deg_vals):.2f}\n")
    print("Completed.")

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(sys.argv[1:])
