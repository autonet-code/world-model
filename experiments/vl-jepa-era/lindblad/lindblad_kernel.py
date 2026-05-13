"""Lindblad master-equation evolver for the substrate.

A standalone numerical kernel that integrates

    dρ/dt = -i[H, ρ] + Σⱼ γⱼ · D[Lⱼ](ρ)

where D[L](ρ) = LρL† - ½(L†Lρ + ρL†L).

No substrate dependencies. Plain numpy. Tested against textbook two-level
problems with closed-form solutions before being wired up to anything.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# Constants — Pauli matrices in the {|PRO⟩=|0⟩, |CON⟩=|1⟩} basis
# ---------------------------------------------------------------------------

I2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

# Transition operators driving population.
# L_PRO drives population from CON to PRO (|0⟩⟨1|).
# L_CON drives population from PRO to CON (|1⟩⟨0|).
RAISE_TO_PRO = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
LOWER_TO_CON = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)


# ---------------------------------------------------------------------------
# Core Lindblad pieces
# ---------------------------------------------------------------------------


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B + B @ A


def dissipator(L: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """The Lindblad dissipator D[L](ρ) = LρL† - ½(L†Lρ + ρL†L)."""
    Ldag = L.conj().T
    LdagL = Ldag @ L
    return L @ rho @ Ldag - 0.5 * anticommutator(LdagL, rho)


def lindblad_rhs(
    rho: np.ndarray,
    H: np.ndarray,
    jump_ops: Sequence[tuple[np.ndarray, float]],
) -> np.ndarray:
    """Right-hand side of the master equation.

    jump_ops is a list of (L_j, gamma_j). The Lindblad rate appears as a
    multiplier on the dissipator, equivalent to absorbing √γ into L
    (we keep them separate for clarity in tests).
    """
    drho = -1j * commutator(H, rho)
    for L, gamma in jump_ops:
        if gamma == 0.0:
            continue
        drho = drho + gamma * dissipator(L, rho)
    return drho


# ---------------------------------------------------------------------------
# Time evolution
# ---------------------------------------------------------------------------


def evolve(
    rho_0: np.ndarray,
    H: np.ndarray,
    jump_ops: Sequence[tuple[np.ndarray, float]],
    t_total: float,
    dt: float = 1e-3,
    record_every: int = 0,
) -> tuple[np.ndarray, list[tuple[float, np.ndarray]]]:
    """Integrate the master equation from t=0 to t=t_total via RK4.

    Args:
        rho_0: initial density matrix (d × d, complex).
        H: Hamiltonian (d × d, Hermitian).
        jump_ops: list of (L_j, gamma_j) pairs.
        t_total: total integration time.
        dt: time step.
        record_every: if > 0, store a snapshot every N steps.

    Returns:
        (rho_final, trajectory) where trajectory is a list of
        (t, rho_at_t) tuples (empty if record_every == 0).
    """
    rho = rho_0.astype(complex).copy()
    n_steps = int(round(t_total / dt))
    trajectory: list[tuple[float, np.ndarray]] = []
    if record_every > 0:
        trajectory.append((0.0, rho.copy()))
    for step in range(1, n_steps + 1):
        # RK4
        k1 = lindblad_rhs(rho, H, jump_ops)
        k2 = lindblad_rhs(rho + 0.5 * dt * k1, H, jump_ops)
        k3 = lindblad_rhs(rho + 0.5 * dt * k2, H, jump_ops)
        k4 = lindblad_rhs(rho + dt * k3, H, jump_ops)
        rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Numerical-noise hygiene: re-Hermitize and renormalize trace.
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho).real
        if tr > 0:
            rho = rho / tr
        if record_every > 0 and step % record_every == 0:
            trajectory.append((step * dt, rho.copy()))
    return rho, trajectory


# ---------------------------------------------------------------------------
# Substrate-flavored helpers
# ---------------------------------------------------------------------------


def root_hamiltonian(omega: float, zeta: float) -> np.ndarray:
    """Single-root Hamiltonian H = ω σ_z + ζ σ_x.

    omega encodes net stake bias (PRO+ vs CON-).
    zeta encodes unresolved cross-pole stake (drives oscillation).
    """
    return omega * SIGMA_Z + zeta * SIGMA_X


def maximally_mixed(d: int = 2) -> np.ndarray:
    return np.eye(d, dtype=complex) / d


def pure_state(coeffs: Sequence[complex]) -> np.ndarray:
    """Density matrix for a pure state ρ = |ψ⟩⟨ψ| with given amplitudes."""
    psi = np.array(coeffs, dtype=complex).reshape(-1, 1)
    psi = psi / np.linalg.norm(psi)
    return psi @ psi.conj().T


def population_pro(rho: np.ndarray) -> float:
    """⟨PRO|ρ|PRO⟩ for a 2-level state. Generalizes to multi-qubit by
    convention if you pass the marginal."""
    return rho[0, 0].real


def coherence_off_diagonal(rho: np.ndarray) -> complex:
    """Off-diagonal coherence c = ρ_01."""
    return rho[0, 1]


def kron(*matrices: np.ndarray) -> np.ndarray:
    """Iterated Kronecker product."""
    out = matrices[0]
    for m in matrices[1:]:
        out = np.kron(out, m)
    return out


def partial_trace_2qubit(rho: np.ndarray, keep: int) -> np.ndarray:
    """Partial trace of a 4×4 two-qubit state, returning the 2×2 marginal
    of qubit `keep` (0 or 1)."""
    if rho.shape != (4, 4):
        raise ValueError("partial_trace_2qubit expects a 4×4 matrix")
    if keep == 0:
        # Trace out qubit 1 (right factor).
        out = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                out[i, j] = rho[2 * i, 2 * j] + rho[2 * i + 1, 2 * j + 1]
        return out
    elif keep == 1:
        # Trace out qubit 0 (left factor).
        out = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                out[i, j] = rho[i, j] + rho[i + 2, j + 2]
        return out
    else:
        raise ValueError("keep must be 0 or 1")
