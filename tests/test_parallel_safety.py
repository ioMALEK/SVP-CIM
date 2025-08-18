"""
Ensure that two independent Python processes, started almost
simultaneously, each get a unique results/<date>_attemptN/ folder.
"""
from pathlib import Path
import subprocess
import sys
import os
import json


SCRIPT = r"""
import json, sys
from cim_svp.result_io import RUN_ROOT, save_dimension_result
save_dimension_result(1, {"dimension": 1, "best_norm": 1.0})
# Print the run folder so parent can read it on stdout
print(json.dumps(str(RUN_ROOT)))
"""


def _start_proc(results_root: Path):
    env = os.environ.copy()
    env["CIM_SVP_RESULTS_DIR"] = str(results_root)  # optional override
    return subprocess.Popen(
        [sys.executable, "-c", SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_two_process_attempt_dirs(tmp_path: Path):
    """
    Launch two short Python processes in parallel and ensure they
    create separate attempt directories.
    """
    p1 = _start_proc(tmp_path)
    p2 = _start_proc(tmp_path)

    out1, err1 = p1.communicate(timeout=10)
    out2, err2 = p2.communicate(timeout=10)

    assert p1.returncode == 0, err1
    assert p2.returncode == 0, err2

    run1 = Path(json.loads(out1.strip()))
    run2 = Path(json.loads(out2.strip()))
    # Paths must differ (attempt1 vs attempt2)
    assert run1 != run2

    for rd in (run1, run2):
        assert rd.is_dir()
        assert (rd / "summary.csv").exists()