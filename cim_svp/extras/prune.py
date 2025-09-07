"""
cim_svp.extras.prune
====================

Filter a search basis to make it more CIM-friendly.

Two heuristics:
1. Norm filter
   • mode="std"        : keep ‖v‖ within ± k·σ  of the mean
   • mode="log_factor" : keep log10‖v‖ within ± factor of log10(mean‖v‖)

2. Direction filter
   • discard any vector whose angle to a *kept* vector is smaller than
     `dir_threshold_deg` (default 10°).

Public function
---------------
prune_basis(vectors, *, norm_mode="std", norm_param=1.0,
            dir_threshold_deg=10.0)

    vectors : list[np.ndarray]  or (m×d) NumPy array (dtype=object allowed)

Returns
-------
pruned_vectors : list[np.ndarray]
stats          : dict
"""

from __future__ import annotations
import math
from typing import Iterable, List, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
#  helpers                                                               #
# ---------------------------------------------------------------------- #
def _norm(v: np.ndarray) -> float:
    return float(math.sqrt(sum(int(x) * int(x) for x in v)))


def _unit_f64(v: np.ndarray) -> np.ndarray:
    v_f = np.asarray(v, dtype=float)
    n = np.linalg.norm(v_f)
    return v_f / n if n else v_f


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cos = np.clip(np.dot(u, v), -1.0, 1.0)
    return math.degrees(math.acos(cos))


# ---------------------------------------------------------------------- #
#  main routine                                                          #
# ---------------------------------------------------------------------- #
def prune_basis(
    vectors: Iterable[np.ndarray] | np.ndarray,
    *,
    norm_mode: str = "std",
    norm_param: float = 1.0,
    dir_threshold_deg: float = 10.0,
):
    """
    Parameters
    ----------
    vectors : iterable of 1-D integer arrays  (basis or candidate set)
    norm_mode :
        "std"         – keep norms within ± norm_param · σ
        "log_factor"  – keep log10 norms within ± norm_param of mean log10 norm
        None          – skip norm filter
    norm_param : float
        k  in ±kσ   or   factor in log10 filter
    dir_threshold_deg : float
        angular threshold in degrees for direction deduplication

    Returns
    -------
    pruned_vectors : list[np.ndarray]
    stats : dict
    """
    vecs = [np.asarray(v, dtype=object) for v in vectors]
    if not vecs:
        raise ValueError("Empty vector list.")

    norms = np.array([_norm(v) for v in vecs], dtype=float)

    # ---------- norm filter ----------
    keep_mask = np.ones(len(vecs), bool)
    if norm_mode == "std":
        mu, sigma = norms.mean(), norms.std()
        lower, upper = mu - norm_param * sigma, mu + norm_param * sigma
        keep_mask &= (norms >= lower) & (norms <= upper)
    elif norm_mode == "log_factor":
        logn = np.log10(norms)
        mu = logn.mean()
        keep_mask &= np.abs(logn - mu) <= norm_param
    elif norm_mode is None:
        pass
    else:
        raise ValueError("norm_mode must be 'std', 'log_factor', or None")

    vecs_norm_filtered = [v for v, keep in zip(vecs, keep_mask) if keep]

    # ---------- direction filter ----------
    kept: List[np.ndarray] = []
    dir_removed = 0
    threshold = dir_threshold_deg
    for v in vecs_norm_filtered:
        un = _unit_f64(v)
        if not kept:                 # first vector always kept
            kept.append(v)
            continue
        if all(_angle_deg(un, _unit_f64(w)) > threshold for w in kept):
            kept.append(v)
        else:
            dir_removed += 1

    stats = dict(
        n_initial=len(vecs),
        after_norm=len(vecs_norm_filtered),
        final=len(kept),
        removed_norm=int(np.sum(~keep_mask)),
        removed_dir=dir_removed,
        mean_norm=float(norms.mean()),
        var_norm=float(norms.var()),
    )
    return kept, stats