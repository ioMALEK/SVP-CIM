#!/usr/bin/env python
"""
cim_physics_study.py
====================

Exploratory script to capture *physical* observables of the CIM while it
solves SVP instances.

Why this file?
--------------
The Optuna-based optimiser only cares about the *final* Ising energy.
Here we keep the entire time trace x_i(t) so we can study:

1) Kibble–Zurek freeze-out vs pump-ramp speed             (Sec. 3.1, McMahon ↓)
2) CAC chaotic hops vs plain linear ramp                  (Fig. 2, Hamerly ↓)
3) Stochastic resonance vs noise amplitude                (see Yamamoto 2021)
4) κ-gain algorithmic phase transition                    (Stern 2022)

References
----------
• McMahon et al., *Science* 354, 614 (2016)  
• Hamerly et al., *PRX* 9, 021032 (2019)  
• cim-optimizer README & source:  
  https://github.com/mcmahon-lab/cim-optimizer/blob/main/README.md

Usage
-----
$ python scripts/cim_physics_study.py \
      --lattice svp_lattices/challenge/50d/instance_01.npy \
      --pump 0.8 1.0 \
      --ramp  500 1500 3000 \
      --noise 0.00 0.02 0.05 \
      --out   results.csv
"""

from __future__ import annotations
import argparse, itertools, json, time, csv, sys, pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cim_svp.cimwrap import gram_to_ising, solve_with_cim


# ---------------------------------------------------------------------------
# Helper: load a lattice basis and build the Gram matrix
# ---------------------------------------------------------------------------
def load_gram(path: str) -> np.ndarray:
    """Expect a *.npy* file holding integer basis B (shape d×d)."""
    B = np.load(path)
    return B @ B.T


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------
def run_cim_once(
    G: np.ndarray,
    *,
    pump_strength: float,
    ramp_steps: int,
    noise: float,
    n_steps: int = 4000,
) -> dict:
    """
    Run the CIM once and return a dict with physics observables.

    Notes on kwargs → CIM physics
    -----------------------------
    * pump_strength : Crystal-pump power as a fraction of threshold.  
    * gain_schedule : Here we build a *linear* ramp → “annealing” mode.
    * noise_std     : Vacuum noise floor; non-zero simulates technical noise.
    """

    # Build a linear ramp array; length == n_steps
    gain_schedule = np.minimum(
        1.0, np.arange(n_steps, dtype=float) / ramp_steps
    )

    # Ask cimwrap for full ‘history’ by forwarding CIM kwarg
    spin, E = solve_with_cim(
        G,
        num_runs=1,                       # ⇐ we keep *one* trace (physics!)
        n_steps=n_steps,
        pump_strength=pump_strength,
        gain_schedule=gain_schedule,
        noise_std=noise,
        record_history=True,              # ← Ising.solve() stores x_i(t)
    )
    # The extra data live in Ising.history; cimwrap forwards the dict
    history = E.history if hasattr(E, "history") else None  # TODO API check

    # Physics metrics
    freeze_step = None
    if history is not None:
        # Freeze-out: first step where max|x_i| > 0.5
        amps = np.abs(history["x"])
        freeze_step = int(np.argmax(amps.max(axis=1) > 0.5))

    return dict(
        best_energy=float(E),
        freeze_step=freeze_step,
        pump=pump_strength,
        ramp=ramp_steps,
        noise=noise,
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", required=True)
    parser.add_argument("--pump", nargs="+", type=float, default=[0.8, 1.0])
    parser.add_argument("--ramp", nargs="+", type=int,   default=[1000, 3000])
    parser.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.02])
    parser.add_argument("--out", default="cim_physics_results.csv")
    args = parser.parse_args(argv)

    G = load_gram(args.lattice)

    results = []
    for p, r, n in itertools.product(args.pump, args.ramp, args.noise):
        print(f"[INFO] pump={p:.2f}  ramp={r}  noise={n:.3f}")
        tic = time.time()
        res = run_cim_once(G, pump_strength=p, ramp_steps=r, noise=n)
        res["elapsed"] = time.time() - tic
        results.append(res)

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"Saved → {args.out}")

    # Quick plot: success prob vs noise for each pump
    fig, ax = plt.subplots()
    for p in args.pump:
        subset = df[df.pump == p]
        grp = subset.groupby("noise")["best_energy"].mean()
        ax.plot(grp.index, grp.values, marker="o", label=f"pump={p:.2f}")
    ax.set(xlabel="noise_std", ylabel="mean best_energy")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()