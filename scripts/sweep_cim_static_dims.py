#!/usr/bin/env python3
"""
Sweep over several lattice dimensions, running optimise_dim() for each.
Optionally auto-plot at the end or prompt the user.
"""

from __future__ import annotations
import argparse
import importlib
import logging
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cim_svp.result_io import RUN_ROOT, save_dimension_result

optimiser = importlib.import_module("scripts.optimise_cim_static")
optimise_dim = optimiser.optimise_dim           # type: ignore[attr-defined]

LOG = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
def _run_one(dim: int, kw: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        return dim, optimise_dim(dim=dim, **kw)
    except Exception as exc:                                   # noqa: BLE001
        return dim, dict(dimension=dim,
                         best_norm=float("inf"),
                         stop_reason=f"exception:{type(exc).__name__}",
                         exception=str(exc))


# ─────────────────────────── CLI helpers ────────────────────────────
def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dims", required=True,
                   help="Comma-separated list, e.g. 8,16,24")
    p.add_argument("--max-trials", type=int, default=1_000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--timeout-s", type=float)
    p.add_argument("--bound", type=float)
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--auto-plot", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args(argv)


def _setup_logging(verbosity: int) -> None:
    lvl = logging.WARNING - 10 * min(3, verbosity)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s sweep [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _maybe_run_plotter(auto: bool) -> None:
    if auto:
        answer = "y"
    elif sys.stdin.isatty():
        try:
            answer = input("Generate plots for this run? [y/N]: ").strip().lower()
        except EOFError:
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

    dims = [int(x.strip()) for x in args.dims.split(",") if x.strip()]
    kw_common = dict(max_trials=args.max_trials,
                     patience=args.patience,
                     timeout_s=args.timeout_s,
                     theoretical_bound=args.bound)

    LOG.info("Sweep over %s → output %s",
             dims, RUN_ROOT.relative_to(Path.cwd()))

    try:
        if args.processes > 1:
            with ProcessPoolExecutor(max_workers=args.processes) as ex:
                futs = {ex.submit(_run_one, d, kw_common): d for d in dims}
                for fut in as_completed(futs):
                    dim, res = fut.result()
                    save_dimension_result(dim, res)
                    LOG.info("Saved dim %d (%s)", dim, res.get("stop_reason"))
        else:
            for dim in dims:
                dim, res = _run_one(dim, kw_common)
                save_dimension_result(dim, res)
                LOG.info("Saved dim %d (%s)", dim, res.get("stop_reason"))
    except KeyboardInterrupt:
        LOG.warning("User interrupted sweep – partial results kept.")
        sys.exit(130)

    _maybe_run_plotter(args.auto_plot)


if __name__ == "__main__":
    main()
