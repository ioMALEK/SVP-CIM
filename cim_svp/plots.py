"""
Quick-look plotting utilities for SVP-CIM experiments.
See module docstring in earlier version for usage examples.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

LOG = logging.getLogger(__name__)
FIG_DPI = 150


# ───────────────────────── helpers ───────────────────────── #
def _ensure_fig_dir(run_dir: Path) -> Path:
    d = run_dir / "figures"
    d.mkdir(exist_ok=True)
    return d


def _load_summary(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "summary.csv")
    return df.sort_values("dim")


def _auto_scale_ax(ax: plt.Axes) -> None:
    if "norm" in (ax.get_ylabel() or "").lower():
        ax.set_yscale("log")


def _log_saved(path: Path) -> None:
    try:
        LOG.info("Saved %s", path.relative_to(Path.cwd()))
    except ValueError:
        LOG.info("Saved %s", path)


# ───────────────────────── plots ─────────────────────────── #
def plot_best_norm(run_dir: str | Path, *, show: bool = True) -> Path:
    run_dir = Path(run_dir)
    df = _load_summary(run_dir)

    fig, ax = plt.subplots()
    ax.plot(df["dim"], df["best_norm"], marker="o")
    ax.set_xlabel("dimension")
    ax.set_ylabel("best_norm")
    _auto_scale_ax(ax)
    ax.grid(True, which="both", ls="--", lw=0.5)

    fn = _ensure_fig_dir(run_dir) / "best_norm_vs_dim.png"
    fig.savefig(fn, dpi=FIG_DPI, bbox_inches="tight")
    _log_saved(fn)
    if show:
        plt.show()
    plt.close(fig)
    return fn


def plot_trials_vs_dim(run_dir: str | Path, *, show: bool = True) -> Path:
    run_dir = Path(run_dir)
    df = _load_summary(run_dir)

    fig, ax = plt.subplots()
    ax.bar(df["dim"], df["trials_run"])
    ax.set_xlabel("dimension")
    ax.set_ylabel("trials_run")
    ax.set_title("Trials per dimension")
    ax.grid(axis="y", ls="--", lw=0.5)

    fn = _ensure_fig_dir(run_dir) / "trials_vs_dim.png"
    fig.savefig(fn, dpi=FIG_DPI, bbox_inches="tight")
    _log_saved(fn)
    if show:
        plt.show()
    plt.close(fig)
    return fn


def plot_params_scatter(run_dir: str | Path,
                        *,
                        param: str = "pump_rate_log10",
                        show: bool = True) -> Path:
    run_dir = Path(run_dir)
    rows = []
    for jf in run_dir.glob("dim*.json"):
        with jf.open() as f:
            d = json.load(f)
        val = d.get("best_params", {}).get(param)
        if val is not None:
            rows.append((d["dimension"], val, d["best_norm"]))
    if not rows:
        LOG.warning("Parameter %s not found in any JSON", param)
        return run_dir / "figures" / f"scatter_{param}.png"

    df = pd.DataFrame(rows, columns=["dim", param, "best_norm"])
    fig, ax = plt.subplots()
    sc = ax.scatter(df[param], df["best_norm"],
                    c=df["dim"], cmap="viridis", s=40, edgecolor="k")
    ax.set_xlabel(param)
    ax.set_ylabel("best_norm")
    _auto_scale_ax(ax)
    fig.colorbar(sc, ax=ax, label="dimension")

    fn = _ensure_fig_dir(run_dir) / f"scatter_{param}.png"
    fig.savefig(fn, dpi=FIG_DPI, bbox_inches="tight")
    _log_saved(fn)
    if show:
        plt.show()
    plt.close(fig)
    return fn


# ───────────────────────── CLI omitted (unchanged) ────────────────── #
# Keep rest of file same as previous version (parse args, main, etc.)

# ───────────────────────────── CLI ────────────────────────────────── #
def _parse_cli(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot SVP-CIM run results")
    p.add_argument("--run-dir", type=str,
                   help="Path to results/<DAY>_<attemptN>")
    p.add_argument("--latest", action="store_true",
                   help="Pick the most recent results/*/ directory")
    p.add_argument("--what", type=str, default="norm-vs-dim",
                   choices=_PLOT_FUNCS.keys(),
                   help="Which plot to make")
    p.add_argument("--param", type=str, default="pump_rate_log10",
                   help="Parameter name for param-scatter")
    p.add_argument("--all", action="store_true",
                   help="Make all default plots")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args(argv)


def _setup_logging(verbosity: int) -> None:
    lvl = logging.WARNING - 10 * min(3, verbosity)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s plots [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _latest_results_dir(base: Path = Path("results")) -> Path:
    runs = sorted(base.glob("*_attempt*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError("No results/*_attempt*/ directories found.")
    return runs[-1]


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_cli(argv)
    _setup_logging(args.verbose)

    if args.latest:
        run_dir = _latest_results_dir()
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        print("Either --run-dir or --latest must be specified.", file=sys.stderr)
        sys.exit(2)

    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    if args.all:
        plot_best_norm(run_dir)
        plot_trials_vs_dim(run_dir)
        plot_params_scatter(run_dir, param=args.param)
    else:
        func = _PLOT_FUNCS[args.what]
        if args.what == "param-scatter":
            func(run_dir, param=args.param)
        else:
            func(run_dir)


if __name__ == "__main__":
    main()
