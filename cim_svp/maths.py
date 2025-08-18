import math
from decimal import Decimal, getcontext
import numpy as np
getcontext().prec = 120

def vec_sq_norm_int(v) -> int:
    return int(sum(int(x) * int(x) for x in v))

def gram_int(B: np.ndarray) -> np.ndarray:
    return B @ B.T          # dtype=object → exact

def build_J(B: np.ndarray, scale: float = 1.0) -> np.ndarray:
    G = gram_int(B)
    m = max(1, max(abs(int(x)) for x in G.flat))
    J = -np.asarray(G, dtype=np.float64) / float(m) * scale
    np.fill_diagonal(J, 0.0)
    return J

def sq_from_spins(s: np.ndarray, G_int: np.ndarray) -> int:
    return int(sum(int(si)*int(sj)*int(G_int[i,j])
                   for i,si in enumerate(s)
                   for j,sj in enumerate(s)))

def sqrt_int(n: int) -> float:
    return math.sqrt(n) if n.bit_length() < 52 else float(Decimal(n).sqrt())

def to_pm1(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    vals = set(np.unique(np.rint(a)))
    if vals <= {-1, 1}:
        return a.astype(int)
    if vals <= {0, 1}:
        return (2 * a - 1).astype(int)
    return np.sign(a).astype(int)
