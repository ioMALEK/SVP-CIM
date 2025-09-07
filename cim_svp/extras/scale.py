"""
cim_svp.extras.scale
====================

Scale a set of *new* candidate vectors so that their Euclidean norms sit in the
same order of magnitude.

Algorithm
---------
1. Compute N_max = max ‖v‖.
2. For each vector compute
       k = floor(N_max / ‖v‖)           (or ceil if `rounding="ceil"`)
   with k ≥ 1.  An optional `k_cap` limits runaway magnitudes.
3. Replace v by k · v.
4. Remove exact duplicates produced by the scaling.

No filtering against the original basis is done here; call
`encoding.build_extended_subspace` first to obtain candidates without basis
rows.

Public function
---------------
scale_candidates(candidates, *, rounding="floor", k_cap=None)
    candidates : list[np.ndarray]
    rounding   : "floor" | "ceil"
    k_cap      : None or int (scaling factors are clipped to this max)

Returns
-------
scaled  : list[np.ndarray]          (duplicates removed)
stats   : dict
"""

from __future__ import annotations
import math
from typing import List, Tuple, Sequence, Set

import numpy as np


# ---------------------------------------------------------------------- #
#  Helpers                                                               #
# ---------------------------------------------------------------------- #
def _norm_float(v: np.ndarray) -> float:
    return float(math.sqrt(sum(int(x) * int(x) for x in v)))


# ---------------------------------------------------------------------- #
#  Main routine                                                          #
# ---------------------------------------------------------------------- #
def scale_candidates(
    vectors: Sequence[np.ndarray],
    *,
    rounding: str = "floor",
    k_cap: int | None = None,
) -> Tuple[List[np.ndarray], dict]:
    """
    Parameters
    ----------
    vectors  : candidate vectors (dtype=object)
    rounding : "floor" (default) or "ceil"
    k_cap    : optional int, clip k to this value

    Returns
    -------
    scaled_vectors : list with duplicates removed
    stats : dict  (n_initial, n_final, duplicates_removed,
                   max_scale, min_scale)
    """
    if rounding not in ("floor", "ceil"):
        raise ValueError("rounding must be 'floor' or 'ceil'.")

    if not vectors:
        raise ValueError("Empty candidate list.")

    norms = np.array([_norm_float(v) for v in vectors], dtype=float)
    n_max = float(norms.max())

    scale_func = math.floor if rounding == "floor" else math.ceil

    scaled: List[np.ndarray] = []
    seen: Set[Tuple[int, ...]] = set()
    scales: List[int] = []
    duplicates = 0

    for v, n in zip(vectors, norms):
        k = max(1, scale_func(n_max / n))
        if k_cap is not None:
            k = min(k, int(k_cap))
        vec_scaled = k * v
        key = tuple(int(x) for x in vec_scaled)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        scaled.append(vec_scaled)
        scales.append(k)

    stats = dict(
        n_initial=len(vectors),
        n_final=len(scaled),
        duplicates_removed=duplicates,
        max_scale=max(scales),
        min_scale=min(scales),
    )
    return scaled, stats