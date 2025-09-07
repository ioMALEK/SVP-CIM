# CAC_Potts.py
# =========================================================================
# Chaotic-Amplitude-Control solver with multi-phase (Potts) dynamics.
# Created 2025-08-24 by adapting the binary CIM_CAC_GPU kernel.

from __future__ import annotations
import math
import torch
import numpy as np

torch.backends.cudnn.benchmark = True    # keep GPU fast


# -------------------------------------------------------------------------
def CIM_CAC_Potts_GPU(
        T_time: int,
        J: np.ndarray,
        *,
        Q: int = 3,
        batch_size: int = 1,
        time_step: float = 0.05,
        r: float | None = None,
        beta: float = 0.25,
        phase_lock: float = 0.03,
        noise: float = 0.0,
        device: torch.device = torch.device("cpu"),
        stop_when_solved: bool = False,
        H_target: float | None = None):
    """
    Chaotic-Amplitude-Control solver with 2·Q phase wells.

    Parameters
    ----------
    T_time        : int      – number of round trips
    J             : ndarray  – symmetric coupling matrix  (float32/64)
    Q             : int      – half the number of Potts states (≥1)
    batch_size    : int
    time_step     : float
    r             : float or None  – pump value; default `0.8-(N/220)**2`
    beta          : float          – amplitude-feedback strength
    phase_lock    : float          – γ in   φ ← φ − γ sin(2Qφ)
    noise         : float          – additive white noise on amplitude
    device        : torch.device
    stop_when_solved : bool        – early exit if energy ≤ H_target
    H_target         : float       – ground-state energy (if known)

    Returns
    -------
    potts_spins        : (batch, N) int tensor  in {−Q…−1, 1…Q}
    spin_trajectory    : (batch, N, T_time) float32 – real amplitude |a|
    """
    # ------------------------------------------------------------------
    # Tensor setup
    # ------------------------------------------------------------------
    J = torch.as_tensor(J, dtype=torch.float32, device=device)
    N = J.shape[0]
    if r is None:
        r = 0.8 - (N / 220.0) ** 2

    r_t  = torch.full((T_time,), r,     device=device, dtype=torch.float32)
    bet  = torch.full((T_time,), beta,  device=device, dtype=torch.float32)

    # complex amplitudes  a = x_re + i x_im
    x_re = 0.001 * torch.randn(batch_size, N, device=device)
    x_im = 0.001 * torch.randn(batch_size, N, device=device)

    spin_trajectory = torch.empty(batch_size, N, T_time, device=device)

    # ------------------------------------------------------------------
    # Helper: quantise phase → Potts spin  (vectorised)
    # ------------------------------------------------------------------
    def phase_to_spin(phi: torch.Tensor) -> torch.Tensor:
        k = torch.round(Q * phi / math.pi).to(torch.int8)     # (-Q … Q)
        zeros = (k == 0)
        k[zeros & (phi >= 0)] = 1
        k[zeros & (phi <  0)] = -1
        k.clamp_(-Q, Q)
        return k

    # ------------------------------------------------------------------
    # Main time evolution
    # ------------------------------------------------------------------
    for t in range(T_time):
        # store amplitude magnitude for optional plotting
        spin_trajectory[:, :, t] = torch.sqrt(x_re**2 + x_im**2)

        # ---------------- linear CAC update --------------------------
        MVM_re = x_re @ J
        MVM_im = x_im @ J
        x_re = x_re + time_step * ( -x_re + r_t[t] * MVM_re )
        x_im = x_im + time_step * ( -x_im + r_t[t] * MVM_im )

        # ---------------- non-linear clip ----------------------------
        r_amp = torch.sqrt(x_re**2 + x_im**2)
        r_amp = torch.tanh(bet[t] * r_amp)            # amplitude saturation

        phi = torch.atan2(x_im, x_re)
        phi = phi - phase_lock * torch.sin(2 * Q * phi)   # phase locking

        x_re = r_amp * torch.cos(phi)
        x_im = r_amp * torch.sin(phi)

        # ---------------- optional noise -----------------------------
        if noise > 0:
            x_re.add_(noise * torch.randn_like(x_re))
            x_im.add_(noise * torch.randn_like(x_im))

        # ---------------- early stopping ----------------------------
        if stop_when_solved and H_target is not None:
            spins_now = phase_to_spin(phi)
            same = spins_now.unsqueeze(2).eq(spins_now.unsqueeze(1))
            H = -0.5 * torch.sum(J * same, dim=(1, 2))
            if torch.any(H <= H_target):
                break

    # ------------------------------------------------------------------
    # Final spin read-out
    # ------------------------------------------------------------------
    final_phi   = torch.atan2(x_im, x_re)
    potts_spins = phase_to_spin(final_phi).cpu()

    return (potts_spins.numpy(),
            spin_trajectory.cpu().numpy(),
            t + 1)        # actual steps executed