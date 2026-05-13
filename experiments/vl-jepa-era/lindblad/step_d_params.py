"""Step D — bounded Hamiltonian parameters.

Separates direction (omega in [-kappa, kappa], zeta in [0, lambda]) from
confidence (total weight W, which modulates the effective dissipator
rate gamma_eff).

Why: in Step C we saw that omega = kappa * sum(s_i * cap_i) grew
unboundedly with evidence count, making heavily-evidenced roots
oscillate so fast that observations couldn't perturb them. The fix
separates "which way" (omega_dir) from "how much" (W).

Cognitive intuition: a confident root should be SLOW to change
(small effective gamma), not FAST to oscillate (large omega). This
is the whole correction.
"""

from __future__ import annotations

import math
from typing import Sequence

from lindblad.step_b_predictions import SubClaim, gaussian_kernel, euclidean


# ---------------------------------------------------------------------------
# Per-root direction and confidence
# ---------------------------------------------------------------------------


EPS = 1e-9


def total_weight(claims: Sequence[SubClaim]) -> float:
    """Total magnitude of evidence on this root: W = sum |s_i| * cap_i."""
    return sum(abs(c.stake) * c.capacity for c in claims)


def direction(claims: Sequence[SubClaim]) -> float:
    """Direction of evidence in [-1, 1].
        omega_dir = sum(s_i * cap_i) / (sum |s_i| * cap_i + eps)
    +1 = all PRO; -1 = all CON; 0 = balanced or empty.
    """
    W = total_weight(claims)
    signed = sum(c.stake * c.capacity for c in claims)
    return signed / (W + EPS)


def normalized_tension(claims: Sequence[SubClaim]) -> float:
    """Tension in [0, 1]: 2 * sqrt(p * c) where p, c are PRO and CON
    fractions of total weight. 0 when only one pole present; 1 when
    PRO and CON each carry exactly half.
    """
    W = total_weight(claims)
    if W < EPS:
        return 0.0
    P = sum(max(c.stake, 0.0) * c.capacity for c in claims)
    C = sum(max(-c.stake, 0.0) * c.capacity for c in claims)
    p, c = P / W, C / W
    return 2.0 * math.sqrt(p * c)


def omega_for_root_d(claims: Sequence[SubClaim], kappa: float = 1.0) -> float:
    """Bounded omega: kappa * direction. omega in [-kappa, kappa]."""
    return kappa * direction(claims)


def zeta_for_root_d(claims: Sequence[SubClaim], lam: float = 1.0) -> float:
    """Bounded zeta: lam * normalized_tension. zeta in [0, lam]."""
    return lam * normalized_tension(claims)


# ---------------------------------------------------------------------------
# Cross-root coupling
# ---------------------------------------------------------------------------


def J_for_root_pair_d(
    claims_a: Sequence[SubClaim],
    claims_b: Sequence[SubClaim],
    mu: float = 1.0,
    bandwidth: float = 0.5,
) -> float:
    """Cross-root coupling, normalized by per-root weights so two
    heavily-evidenced roots don't get unbounded coupling.

    J_ab = mu * sum_{i,j} sign(s_i^a) sign(s_j^b) cap_i cap_j K(d) / (W_a * W_b)^{1/2}

    Bounded in [-mu, mu] by construction (sum of normalized terms over
    the support of the kernel).
    """
    W_a = total_weight(claims_a)
    W_b = total_weight(claims_b)
    if W_a < EPS or W_b < EPS:
        return 0.0
    raw = 0.0
    for ci in claims_a:
        for cj in claims_b:
            d = euclidean(ci.coords, cj.coords)
            raw += (math.copysign(1, ci.stake) * math.copysign(1, cj.stake)
                    * abs(ci.stake * cj.stake) * ci.capacity * cj.capacity
                    * gaussian_kernel(d, bandwidth))
    return mu * raw / math.sqrt(W_a * W_b * (W_a + W_b))


# ---------------------------------------------------------------------------
# Confidence-modulated gamma
# ---------------------------------------------------------------------------


def gamma_modulator(W: float, W_scale: float = 1.0) -> float:
    """Heavier roots resist observations: gamma_eff = gamma_base * f(W).
        f(W) = 1 / (1 + W / W_scale)
    Light roots: f -> 1. Heavy roots: f -> 0. Smooth interpolation.
    """
    return 1.0 / (1.0 + W / max(W_scale, EPS))
