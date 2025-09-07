from __future__ import annotations
import numpy as np, torch, optuna, math
from typing import Dict, Tuple
from cim_optimizer.simcpm import CPM
from cim_svp.maths       import build_J as build_J_exact
from cim_svp.maths       import vec_sq_norm_int, sqrt_int

# -------- CPM defaults ------------------------------------------------
Q_STATE      = 3
T_TIME       = 400
DEVICE       = torch.device("cpu")

# =====================================================================
def spins_to_vector(spins: np.ndarray,
                    sub_basis: np.ndarray) -> np.ndarray:
    """Integer linear combination (exact)."""
    coeffs = spins.astype(object)
    return np.sum(coeffs[:, None] * sub_basis, axis=0)


# =====================================================================
def build_J(sub_basis: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Wrapper around maths.build_J  → returns float64 coupling matrix scaled
    for the CPM.
    """
    return build_J_exact(sub_basis, scale=scale)


# =====================================================================
def _cpm_once(J: np.ndarray, params: Dict) -> np.ndarray:
    """Run CPM once and return Potts spin vector (len X)."""
    sim = CPM(J,
              Q          = Q_STATE,
              T_time     = T_TIME,
              phase_lock = params["phase_lock"],
              beta       = params["beta"],
              r          = params["pump"],
              noise      = params["noise"],
              batch_size = 1,
              device     = DEVICE)
    sim.run()
    return sim.spins()[0]


# =====================================================================
def optimize_params(J: np.ndarray,
                    sub_basis: np.ndarray,
                    *,
                    n_trials: int = 30,
                    m_runs: int   = 10,
                    rng: np.random.Generator | None = None
                   ) -> Dict:
    """Optuna tuning, objective = shortest vector length."""
    if rng is None:
        rng = np.random.default_rng()

    def objective(trial):
        params = dict(
            phase_lock = trial.suggest_float("phase_lock", 0.05, 0.5),
            beta       = trial.suggest_float("beta",       1.5, 4.0),
            pump       = trial.suggest_float("pump",       0.7, 0.95),
            noise      = trial.suggest_float("noise",      0.0, 0.05),
        )
        best_sq = math.inf
        for _ in range(m_runs):
            spins  = _cpm_once(J, params)
            vec    = spins_to_vector(spins, sub_basis)
            best_sq = min(best_sq, vec_sq_norm_int(vec))
        return sqrt_int(best_sq)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# =====================================================================
def evaluate_params(J: np.ndarray,
                    sub_basis: np.ndarray,
                    params: Dict,
                    *,
                    n_runs: int,
                    rng: np.random.Generator | None = None
                   ) -> Tuple[np.ndarray, float]:
    """Run CPM many times; return best vector and its length (float)."""
    if rng is None:
        rng = np.random.default_rng()

    best_vec: np.ndarray | None = None
    best_sq:  int               = math.inf

    for _ in range(n_runs):
        spins  = _cpm_once(J, params)
        vec    = spins_to_vector(spins, sub_basis)
        sq     = vec_sq_norm_int(vec)
        if sq < best_sq:
            best_sq, best_vec = sq, vec

    return best_vec, sqrt_int(best_sq)