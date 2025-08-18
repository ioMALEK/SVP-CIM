"""
cim_svp.extras.encoding
=======================

Brute enumeration of lattice vectors built from a *small* subset of basis rows
with bounded integer coefficients.  Extended version:

  • Flexible coefficient grid:
        linear     (default)
        log10
        explicit list
  • Flexible support size:
        support_min ≤ |support| ≤ support_max
        `"all"`     → any non-empty subset of rows
  • Filters out every vector that *already* belongs to the basis.
  • Returns both the unique candidate list and summary statistics.

API
---
build_extended_subspace(
        B,
        coeff_min=1, coeff_max=1, coeff_interval=1,
        coeff_mode="linear", coeff_values=None,
        support_min=2, support_max=2,     # or support="all"
    ) -> stats_dict, candidates_list
"""

from __future__ import annotations
import math
from decimal import Decimal, getcontext
from itertools import combinations, product
from typing import Iterable, List, Tuple, Set

import numpy as np

getcontext().prec = 120


# ---------------------------------------------------------------------- #
#  inner helpers                                                         #
# ---------------------------------------------------------------------- #
def _sq_norm(v: np.ndarray) -> int:
    return int(sum(int(x) * int(x) for x in v))


def _canonical_dir(vec: Tuple[int, ...]) -> Tuple[int, ...]:
    """Normalise by sign & gcd to detect colinearity."""
    nz = next((x for x in vec if x), 0)
    if nz == 0:
        return vec
    sign = 1 if nz > 0 else -1
    g = math.gcd(*[abs(x) for x in vec if x])
    return tuple(sign * x // g for x in vec)


def _build_coeff_set(
    coeff_min: int,
    coeff_max: int,
    coeff_interval: int,
    coeff_mode: str,
    coeff_values: Iterable[int] | None,
) -> List[int]:
    if coeff_values is not None:
        pos = sorted({abs(int(c)) for c in coeff_values if c})
    else:
        if coeff_mode == "linear":
            pos = list(range(coeff_min, coeff_max + 1, coeff_interval))
        elif coeff_mode == "log":
            pos = [10 ** p for p in range(coeff_min, coeff_max + 1, coeff_interval)]
        else:
            raise ValueError("coeff_mode must be 'linear' or 'log'")
    if not pos:
        raise ValueError("Coefficient set is empty.")
    return [c for p in pos for c in (-p, p)]


# ---------------------------------------------------------------------- #
#  main routine                                                          #
# ---------------------------------------------------------------------- #
def build_extended_subspace(
    B: np.ndarray,
    *,
    coeff_min: int = 1,
    coeff_max: int = 1,
    coeff_interval: int = 1,
    coeff_mode: str = "linear",
    coeff_values: Iterable[int] | None = None,
    support_min: int | str = 2,
    support_max: int | None = 2,
):
    """
    Parameters
    ----------
    B : (n×m) basis, dtype=object
    coeff_* : define the *positive* magnitudes (see docstring)
    support_min, support_max :
        • ints        → enumerate subsets with size in [min, max]
        • support_min == "all" → any non-empty subset, ignore support_max

    Returns
    -------
    stats : dict
    candidates : list[(vector, norm²)]   (basis vectors removed)
    """

    n = B.shape[0]

    # ---- support size handling ----
    if support_min == "all":
        k_values = range(1, n + 1)
    else:
        if not (isinstance(support_min, int) and isinstance(support_max, int)):
            raise ValueError("support_min/max must be int or 'all'.")
        if not (1 <= support_min <= support_max <= n):
            raise ValueError("support_min/max out of range.")
        k_values = range(support_min, support_max + 1)

    # ---- coefficient magnitudes ----
    coeffs = _build_coeff_set(
        coeff_min, coeff_max, coeff_interval, coeff_mode, coeff_values
    )

    # ---- basis tuples for filtering ----
    basis_rows = {tuple(int(x) for x in row) for row in B}

    seen_vectors: Set[Tuple[int, ...]] = set()
    direction_set: Set[Tuple[int, ...]] = set()
    duplicates_removed = 0
    candidates: List[Tuple[np.ndarray, int]] = []

    for k in k_values:
        for idxs in combinations(range(n), k):
            for ks in product(coeffs, repeat=k):
                if all(c == 0 for c in ks):
                    continue
                vec = sum(c * B[i] for i, c in zip(idxs, ks))
                key = tuple(int(x) for x in vec)
                if key in basis_rows:
                    continue                        # skip original basis vector
                if key in seen_vectors:
                    duplicates_removed += 1
                    continue
                seen_vectors.add(key)
                dir_key = _canonical_dir(key)
                direction_set.add(dir_key)
                candidates.append((vec, _sq_norm(vec)))

    if not candidates:
        raise ValueError("No new candidate vectors were generated.")

    # ---- statistics ----
    norms = np.array([math.sqrt(n2) for _, n2 in candidates], dtype=float)
    ln    = np.log(norms)
    dirs  = np.array([np.asarray(v, dtype=float)/n for v, n in zip(
                      (c[0] for c in candidates), norms)], dtype=float)

    stats = dict(
        n_vectors         = len(candidates),
        duplicates_removed= duplicates_removed,
        colinear_duplicates = len(candidates) - len(direction_set),
        mean_norm         = float(norms.mean()),
        var_norm          = float(norms.var()),
        mean_log_norm     = float(ln.mean()),
        var_log_norm      = float(ln.var()),
        mean_direction    = dirs.mean(axis=0),
    )
    return stats, candidates