import numpy as np
from fpylll import IntegerMatrix, LLL
from cim_optimizer.solve_Ising import Ising

def lll_reduce_basis(B):
    """
    Perform LLL reduction on an integer basis B.
    B should be an (n×n) integer numpy array.
    Returns a new (n×n) integer numpy array S.
    """
    n = B.shape[0]
    M = IntegerMatrix(n, n)
    for i in range(n):
        for j in range(n):
            M[i, j] = int(B[i, j])
    LLL.reduction(M)
    S = np.array([[M[i, j] for j in range(n)] for i in range(n)], dtype=int)
    return S

def svp_cim(B, num_iter=None, cim_params=None):
    """
    Approximate-SVP via CIM heuristic.

    Args:
      B            : (n×n) integer basis array
      num_iter     : how many CIM iterations to run (default = n)
      cim_params   : dict of extra kwargs to pass to Ising(...)

    Returns:
      best_vec     : (n,) integer vector found
      best_norm    : its Euclidean norm
    """
    # 1) LLL-reduce
    S = lll_reduce_basis(B)
    n = S.shape[0]
    if num_iter is None:
        num_iter = n

    # Track the current "worst" basis vector to replace
    def basis_norms(S):
        return np.linalg.norm(S, axis=1)
    best_vec = None
    best_norm = np.inf

    # Pre-pack any CIM hyperparams
    cim_kwargs = dict(
        pump_rate    = 1.5,
        feedback_strength = 0.8,
        noise_amplitude   = 0.05,
        time_step         = 0.01,
        num_steps         = 2000,
    )
    if cim_params:
        cim_kwargs.update(cim_params)

    for it in range(num_iter):
        # 2) Build Gram matrix G = S S^T
        G = S @ S.T

        # 3) Solve the Ising model H(n) = n^T G n
        solver = Ising(J=G, h=np.zeros(n), **cim_kwargs)
        result = solver.solve()
        n_vec = result.spin  # array of +1 / -1

        # 4) Compute the candidate vector Y = sum_i n_i * S[i]
        Y = n_vec @ S       # shape (n,)
        Y_norm = np.linalg.norm(Y)

        # 5) If it's nonzero and better than our best, keep it and replace the worst
        if Y_norm > 0 and Y_norm < best_norm:
            best_norm = Y_norm
            best_vec = Y.copy()

        # 6) Replace the basis vector with maximum norm if Y is better
        norms = basis_norms(S)
        j_max = np.argmax(norms)
        if Y_norm > 0 and Y_norm < norms[j_max]:
            S[j_max, :] = Y

        print(f"Iter {it+1:2d}:  ‖Y‖ = {Y_norm:.4f},  best = {best_norm:.4f}")

    return best_vec, best_norm

if __name__ == "__main__":
    # --- Example use ---
    # Create a random 10×10 integer basis (for testing only)
    np.random.seed(0)
    B0 = np.random.randint(-10, 11, size=(10,10))

    v, norm_v = svp_cim(B0)
    print("\nResult:")
    print("Best vector:", v)
    print("Norm       :", norm_v)
