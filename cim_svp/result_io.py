"""
Persistence helper.

On first import it creates a per-run directory under results/ using the
pattern   DD_MM_attemptN/  (N auto-increments, race-safe).

Public objects
--------------
RUN_ROOT  : pathlib.Path          # folder for *this* program run
save_dimension_result(dim, dict)  # write JSON + append summary.csv
"""
from __future__ import annotations
import csv
import datetime as _dt
import json
import pathlib
from typing import Any

# ────────────────────────── initialise run folder ─────────────────── #
_BASE = pathlib.Path(__file__).resolve().parents[1] / "results"
_BASE.mkdir(exist_ok=True)
_DATE = _dt.date.today().strftime("%d_%m")      # e.g. "18_08"


def _next_attempt(dir_date: str) -> pathlib.Path:
    """
    Create results/<dir_date>_attemptN atomically.
    Retries N = 1,2,3… until mkdir succeeds (handles concurrent runs).
    """
    n = 1
    while True:
        run_root = _BASE / f"{dir_date}_attempt{n}"
        try:
            run_root.mkdir(parents=True)
            (run_root / "figures").mkdir()
            return run_root
        except FileExistsError:
            n += 1   # another process grabbed that name – try next


RUN_ROOT: pathlib.Path = _next_attempt(_DATE)

# ───────────────────────── public helper API ──────────────────────── #
def _json_filename(dim: int) -> pathlib.Path:
    today = _dt.date.today().isoformat()
    return RUN_ROOT / f"dim{dim:02d}_{today}.json"


def save_dimension_result(dim: int, payload: dict[str, Any]) -> None:
    """Idempotent save: overwrites JSON if same dim re-run in this session."""
    # JSON ------------------------------------------------------------------
    fn = _json_filename(dim)
    with fn.open("w") as f:
        json.dump(payload, f, indent=2, default=str)

    # CSV summary -----------------------------------------------------------
    csv_fn = RUN_ROOT / "summary.csv"
    first = not csv_fn.exists()
    flat = {k: v for k, v in payload.items()
            if not isinstance(v, (dict, list, tuple))}
    flat["dim"] = dim
    with csv_fn.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat.keys()))
        if first:
            writer.writeheader()
        writer.writerow(flat)
