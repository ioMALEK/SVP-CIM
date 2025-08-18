"""
Validate that save_dimension_result() writes JSON and CSV under a
monkey-patched results root, and that re-saving overwrites the JSON.
"""
from pathlib import Path
import importlib
import json

import cim_svp.result_io as rio


def test_save_and_overwrite(tmp_path: Path):
    # Monkey-patch base and run_root to tmpdir
    rio._BASE = tmp_path                        # type: ignore[attr-defined]
    rio.RUN_ROOT = tmp_path                     # type: ignore[attr-defined]
    importlib.reload(rio)

    # Create first attempt folder
    attempt_dir = rio._next_attempt("test")     # type: ignore[attr-defined]
    rio.RUN_ROOT = attempt_dir                  # type: ignore[attr-defined]
    importlib.reload(rio)

    rio.save_dimension_result(4, {"dimension": 4, "best_norm": 2.0})

    json_path = rio._json_filename(4)           # type: ignore[attr-defined]
    assert json_path.exists()

    csv_path = rio.RUN_ROOT / "summary.csv"     # type: ignore[attr-defined]
    assert csv_path.exists()

    # Overwrite and confirm JSON updated
    rio.save_dimension_result(4, {"dimension": 4, "best_norm": 1.5})
    with json_path.open() as f:
        assert json.load(f)["best_norm"] == 1.5