"""
classical_solvers.py  – reference (non-CIM) lattice-SVP algorithms.

WARNING
-------
• For benchmarking and unit–testing *only*.
• Do NOT import this module inside the optimisation / CIM pipeline.

Functions
---------
shortest_enum(B, max_dim=20)
    Find the shortest vector that can be expressed as a linear combination
    of the rows of B with coefficients in  {-Q,…,-1, 1,…,Q}.

shortest_monte_carlo(B, samples=200_000, seed=None)
    Random ±1 sampling, returns best vector and norm.

lll_reduce(B, delta=0.99)
    Float LLL; uses fpylll if available, else a minimal Numpy version.

gs_greedy(B)
    Greedy length-reduction using Gram–Schmidt coefficients.

bkz_reduce(B, beta=20, max_loops=8)
    Thin wrapper around fpylll.BKZ; raises if fpylll not installed.
"""

from __future__ import annotations
import math, itertools, random
import numpy as np
from decimal import Decimal, getcontext

getcontext().prec = 120


# ---------- helpers ----------------------------------------------------
def _vec_sq_norm_int(v):
    return int(sum(int(x) * int(x) for x in v))


def _gram_int(B):
    return B @ B.T


def _sqrt_int(n: int) -> float:
    return float(Decimal(n).sqrt()) if n.bit_length() > 52 else math.sqrt(n)


# ---------- 1. exact enumeration --------------------------------------
def shortest_enum(B: np.ndarray, Q: int = 1, max_dim: int = 20):
    """
    Find the shortest vector that can be expressed as a linear combination
    of the rows of B with coefficients in  {-Q,…,-1, 1,…,Q}.

    Parameters
    ----------
    B        : (K, N) ndarray  – lattice basis (dtype=object, big ints OK)
    Q        : int ≥ 1         – Potts parameter (Q=1 ⇒ ±1 enumeration)
    max_dim  : int             – safety cut-off on K to keep runtime finite

    Returns
    -------
    coeffs   : length-K ndarray of ints
    norm     : int   (Euclidean length of the corresponding lattice vector)
    """
    K = B.shape[0]
    if K > max_dim:
        raise ValueError(f"Enumeration capped at {max_dim} dims (got {K})")

    # pre-compute Gram matrix once
    G = _gram_int(B)

    # coefficient alphabet  {-Q…-1, 1…Q}
    alph = list(range(-Q, 0)) + list(range(1, Q + 1))

    best_sq  = math.inf
    best_vec = None
    for coeffs in itertools.product(alph, repeat=K):
        s  = np.array(coeffs, dtype=int)
        sq = int(np.sum(s[:, None] * s[None, :] * G))   # ‖Σ s_i b_i‖²
        if sq < best_sq:
            best_sq, best_vec = sq, s
    return best_vec, _sqrt_int(best_sq)


# ---------- 2. Monte-Carlo sampling -----------------------------------
def shortest_monte_carlo(B: np.ndarray,
                         Q: int = 2,
                         samples: int = 200_000,
                         seed=None):
    """
    Heuristic shortest-vector search by random sampling of integer
    coefficient vectors.

    Parameters
    ----------
    B : (k, n) ndarray[int]
        Lattice basis (one vector per row, *object* dtype allowed).
    Q : int, default 2
        “Potts order” ⇒ coefficients are sampled from
            {-⌊Q/2⌋, …, 0, …, ⌊Q/2⌋}.
        Q = 2 reproduces the original ±1 (Ising) search.
        The value is clipped at 20 by the caller’s policy, but the
        algorithm accepts any Q ≥ 1.
    samples : int, default 200_000
        Number of random coefficient vectors to test.
    seed : hashable, optional
        Seed for Python's `random` module, for reproducibility.

    Returns
    -------
    best_s : ndarray[int]   (shape = (k,))
        Coefficient vector that produced the current best length.
    best_len : int
        Exact Euclidean length (integer square-root) of that vector.
    """
    if Q < 1:
        raise ValueError("Q must be >= 1")

    rng = random.Random(seed)
    k   = B.shape[0]
    G   = _gram_int(B)                     # integer Gram matrix

    # --- coefficient alphabet ---------------------------------------
    if Q == 1:
        alphabet = [0]                     # almost useless, but legal
    elif Q == 2:
        alphabet = [-1, 1]                 # original Ising case
    else:
        m = Q // 2                         # max absolute value
        alphabet = list(range(-m, m + 1))  # symmetric incl. 0

    best_sq  = math.inf
    best_vec = None

    for _ in range(samples):
        s = np.fromiter((rng.choice(alphabet) for _ in range(k)), int)
        if not s.any():                    # skip the zero vector
            continue
        sq = int(np.sum(s[:, None] * s[None, :] * G))
        if sq < best_sq:
            best_sq, best_vec = sq, s

    return best_vec, _sqrt_int(best_sq)


# ---------- 3. LLL reduction ------------------------------------------
def lll_reduce(B: np.ndarray, delta: float = 0.99):
    try:
        from fpylll import IntegerMatrix, LLL
        A = IntegerMatrix.from_matrix([[int(x) for x in row] for row in B])
        LLL.reduction(A, delta=delta)
        return np.array(A, dtype=object)
    except ImportError:
        # minimal float-LLL (Lenstra, Lenstra, Lovasz)
        Bf = np.asarray(B, float).copy()
        n = Bf.shape[0]
        GSO = np.zeros_like(Bf)
        mu   = np.zeros((n, n))
        def gso():
            for i in range(n):
                v = Bf[i].copy()
                for j in range(i):
                    mu[i, j] = np.dot(v, GSO[j]) / np.dot(GSO[j], GSO[j])
                    v -= mu[i, j] * GSO[j]
                GSO[i] = v
        gso()
        k = 1
        while k < n:
            for j in range(k-1, -1, -1):
                q = round(mu[k, j])
                if q:
                    Bf[k] -= q * Bf[j]
            gso()
            if np.dot(GSO[k], GSO[k]) >= (delta - mu[k, k-1]**2) * np.dot(GSO[k-1], GSO[k-1]):
                k += 1
            else:
                Bf[[k, k-1]] = Bf[[k-1, k]]
                gso()
                k = max(k-1, 1)
        return Bf.astype(object)


# ---------- 4. Gram–Schmidt greedy ------------------------------------
def gs_greedy(B: np.ndarray, loops: int = 2):
    Bf = B.astype(float).copy()
    for _ in range(loops):
        for i in range(Bf.shape[0] - 1):
            if _vec_sq_norm_int(Bf[i]) > _vec_sq_norm_int(Bf[i+1]):
                Bf[[i, i+1]] = Bf[[i+1, i]]
    return Bf.astype(object)


# ---------- 5. BKZ wrapper --------------------------------------------
def bkz_reduce(B: np.ndarray, beta: int = 20, max_loops: int = 8):
    try:
        from fpylll import IntegerMatrix, BKZ
        A = IntegerMatrix.from_matrix([[int(x) for x in row] for row in B])
        par = BKZ.Param(block_size=beta, max_loops=max_loops)
        BKZ.reduction(A, par)
        return np.array(A, dtype=object)
    except ImportError as e:
        raise ImportError("fpylll required for BKZ") from e
