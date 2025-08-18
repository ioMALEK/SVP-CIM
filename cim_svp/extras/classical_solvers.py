"""
classical_solvers.py  – reference (non-CIM) lattice-SVP algorithms.

WARNING
-------
• For benchmarking and unit–testing *only*.
• Do NOT import this module inside the optimisation / CIM pipeline.

Functions
---------
shortest_enum(B, max_dim=20)
    Exact enumeration (Kannan’s algorithm) when dim ≤ max_dim.

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
def shortest_enum(B: np.ndarray, max_dim: int = 20):
    K = B.shape[0]
    if K > max_dim:
        raise ValueError(f"Enumeration capped at {max_dim} dims (got {K})")
    G = _gram_int(B)
    best_sq = math.inf
    best_vec = None
    for bits in itertools.product([-1, 1], repeat=K):
        s = np.array(bits)
        sq = int(np.sum(s[:, None] * s[None, :] * G))
        if sq < best_sq:
            best_sq, best_vec = sq, s
    return best_vec, _sqrt_int(best_sq)


# ---------- 2. Monte-Carlo sampling -----------------------------------
def shortest_monte_carlo(B: np.ndarray, samples: int = 200_000, seed=None):
    rng = random.Random(seed)
    K = B.shape[0];  G = _gram_int(B)
    best_sq = math.inf; best_vec = None
    for _ in range(samples):
        s = np.fromiter((rng.choice([-1, 1]) for _ in range(K)), int)
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