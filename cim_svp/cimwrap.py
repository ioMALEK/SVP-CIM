"""
cim_svp.cimwrap
===============

• silence_stdio  – context-manager that mutes C / Fortran chatter.
• extract_spins  – robust, version-agnostic ±1 spin extractor.
• safe_solve     – thin wrapper around solver.solve (stdout muted).

The extractor recognises every field name seen in cim-optimizer
≤ 1.0.3, 1.0.4, 1.0.5-dev and falls back to a brute ndarray scan, so it
should keep working when upstream changes names again.
"""
from __future__ import annotations
import os, contextlib
import numpy as np
from .maths import to_pm1   # maps {-1,0,1} / {0,1} → strict ±1


# ---------- silence -----------------------------------------------------
@contextlib.contextmanager
def silence_stdio():
    """Temporarily redirect both stdout and stderr to /dev/null."""
    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):
            yield
    finally:
        devnull.close()


# ---------- spin extractor ---------------------------------------------
def _as_mat(x: np.ndarray) -> np.ndarray:
    """Return a 2-D view (R × K) whatever the input dimensionality."""
    x = np.asarray(x)
    return x if x.ndim == 2 else x[None, :]


def extract_spins(result, K: int) -> np.ndarray:
    """
    Return an (R × K) ±1 int array from *any* cim-optimizer result object
    or dict.  Accepted sources (in that order):

        1. Direct per-run fields
           ['spin_configurations', 'spin_configurations_all_runs',
            'spins', 'states_all_runs']

        2. Trajectories  → last time-step of every run
           ['spin_trajectories_all_runs', 'spin_trajectories',
            'trajectories']

        3. Single best configuration
           ['lowest_energy_state', 'state']

        4. Brute scan – any ndarray whose last axis length == K and whose
           values lie in {-1, 0, 1}.
    """
    obj = getattr(result, "result", result)   # some versions nest payload

    # ---- 1) direct ±1 configurations ----------------------------------
    direct_fields = [
        "spin_configurations",
        "spin_configurations_all_runs",
        "spins",
        "states_all_runs",
    ]
    for name in direct_fields:
        if isinstance(obj, dict) and name in obj:
            M = _as_mat(obj[name])
        elif hasattr(obj, name):
            M = _as_mat(getattr(obj, name))
        else:
            continue
        if M.shape[1] == K:
            return to_pm1(M)

    # ---- 2) trajectories → final step ---------------------------------
    traj_fields = [
        "spin_trajectories_all_runs",
        "spin_trajectories",
        "trajectories",                 # seen in 1.0.5-dev
    ]
    for name in traj_fields:
        if isinstance(obj, dict) and name in obj:
            T = np.asarray(obj[name])
        elif hasattr(obj, name):
            T = np.asarray(getattr(obj, name))
        else:
            continue
        if T.shape[-1] != K:
            continue
        if T.ndim == 3:                 # (runs, time, K)
            return to_pm1(T[:, -1, :])
        if T.ndim == 2:                 # (time, K) – single run
            return to_pm1(T[None, -1, :])

    # ---- 3) single best configuration ---------------------------------
    for name in ["lowest_energy_state", "state"]:
        if isinstance(obj, dict) and name in obj:
            M = _as_mat(obj[name])
        elif hasattr(obj, name):
            M = _as_mat(getattr(obj, name))
        else:
            continue
        if M.shape[1] == K:
            return to_pm1(M)

    # ---- 4) last-chance brute scan ------------------------------------
    for val in (list(obj.values()) if isinstance(obj, dict)
                else obj.__dict__.values() if hasattr(obj, "__dict__")
                else []):
        if isinstance(val, np.ndarray) and val.shape[-1] == K:
            u = np.unique(np.rint(val))
            if set(u) <= {-1, 0, 1}:
                return to_pm1(val.reshape(-1, K))

    raise RuntimeError("spin matrix not found in result.")


# ---------- safe solve --------------------------------------------------
def safe_solve(solver, **kwargs):
    """
    Call `solver.solve(**kwargs)` while silencing its console output.
    Extend with time-outs or automatic retries if necessary.
    """
    with silence_stdio():
        return solver.solve(**kwargs)
