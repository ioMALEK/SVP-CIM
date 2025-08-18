#!/usr/bin/env python3
"""
Optuna-based hyper-parameter search for a *static* Coherent Ising Machine
on a single SVP lattice dimension.

After the run finishes you can either
    • pass --auto-plot to generate figures automatically, or
    • answer y/yes to the prompt if running interactively.
"""

from __future__ import annotations
import argparse
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import optuna
from optuna.trial import TrialState

from cim_svp.utils.timeouts import time_limit
from cim_svp.result_io import RUN_ROOT, save_dimension_result

LOG = logging.getLogger(__name__)

# ─────────────────── fallback CIM stub (remove when real) ────────────
try:
    from cim_svp.cim_runner import run_cim_trial  # type: ignore
except ModuleNotFoundError:                       # pragma: no cover
    LOG.warning("cim_svp.cim_runner not found → using random stub")

    def run_cim_trial(dim: int, **params) -> Tuple[list[int], float]:  # type: ignore
        pump_rate = params.get("pump_rate", 1.0)
        vec = (np.sign(np.random.randn(dim)) + pump_rate * 0.01 *
               np.random.randn(dim)).astype(int).tolist()
        norm = float(np.linalg.norm(vec))
        return vec, norm


# ═════════════════════════════════ objective / optimiser ═════════════
def objective_factory(dim: int):
    def objective(trial: optuna.trial.Trial) -> float:
        pump_rate = 10.0 ** trial.suggest_float("pump_rate_log10", -1.0, 1.0)
        feedback_rate = trial.suggest_float("feedback_rate", 0.0, 1.0)
        num_roundtrips = trial.suggest_int("num_roundtrips", 50, 500, log=True)
        vec, norm = run_cim_trial(
            dim=dim,
            pump_rate=pump_rate,
            feedback_rate=feedback_rate,
            num_roundtrips=num_roundtrips,
        )
        trial.set_user_attr("vector", vec)
        return norm
    return objective


def optimise_dim(
    dim: int,
    *,
    max_trials: int = 1_000,
    patience: int = 100,
    timeout_s: float | None = None,
    theoretical_bound: float | None = None,
) -> dict[str, Any]:
    study = optuna.create_study(direction="minimize")
    objective = objective_factory(dim)

    def early_stop(st: optuna.study.Study, tr: optuna.trial.FrozenTrial) -> None:
        if not st.best_trials:
            return
        if theoretical_bound is not None and st.best_value <= theoretical_bound:
            st.set_user_attr("stop_reason", "theoretical_bound")
            st.stop()
            return
        last_improve = st.best_trial.number
        if tr.number - last_improve >= patience:
            st.set_user_attr("stop_reason", "patience")
            st.stop()

    t0 = time.time()
    stop_reason = "finished"
    try:
        with time_limit(timeout_s):
            study.optimize(
                objective,
                n_trials=max_trials,
                callbacks=[early_stop],
                show_progress_bar=False,
            )
    except TimeoutError:
        stop_reason = "timeout"
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        raise

    elapsed = time.time() - t0
    best_val = study.best_value if study.best_trials else float("inf")
    best_par = study.best_trial.params if study.best_trials else {}

    return dict(
        dimension=dim,
        best_norm=best_val,
        best_params=best_par,
        trials_run=len(study.trials),
        pruned_trials=len([t for t in study.trials if t.state == TrialState.PRUNED]),
        elapsed_s=elapsed,
        stop_reason=study.user_attrs.get("stop_reason", stop_reason),
    )


# ═══════════════════════ CLI / entry-point helpers ═══════════════════
def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING - 10 * min(3, verbosity)
    logging.basicConfig(
        level=level,
        format="%(asctime)s optimise [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--max-trials", type=int, default=1_000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--timeout-s", type=float)
    p.add_argument("--bound", type=float)
    p.add_argument("--auto-plot", action="store_true",
                   help="Run cim_svp.plots --run-dir <this> --all after finishing")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args(argv)


def _maybe_run_plotter(auto: bool) -> None:
    """
    Decide whether to call the plotting module.
    • if --auto-plot      → always run
    • else if stdin TTY   → ask the user
    • else                → skip
    """
    if auto:
        answer = "y"
    elif sys.stdin.isatty():
        try:
            answer = input("Generate plots for this run? [y/N]: ").strip().lower()
        except EOFError:      # piped stdin
            return
    else:
        return

    if answer not in {"y", "yes"}:
        return

    try:
        subprocess.run(
            [sys.executable, "-m", "cim_svp.plots",
             "--run-dir", str(RUN_ROOT), "--all"],
            check=True,
        )
    except Exception as exc:                           # noqa: BLE001
        LOG.error("Plotting failed: %s", exc)


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    _setup_logging(args.verbose)

    try:
        res = optimise_dim(
            dim=args.dim,
            max_trials=args.max_trials,
            patience=args.patience,
            timeout_s=args.timeout_s,
            theoretical_bound=args.bound,
        )
    except KeyboardInterrupt:
        LOG.warning("Interrupted; saving partial data.")
        res = dict(dimension=args.dim, stop_reason="keyboard_interrupt")

    save_dimension_result(args.dim, res)
    LOG.info("Saved results for dim %d → %s",
             args.dim, RUN_ROOT.relative_to(Path.cwd()))
    json.dump(res, sys.stdout, indent=2, default=str)
    print()

    _maybe_run_plotter(args.auto_plot)


if __name__ == "__main__":
    main()
