"""Step C — equilibrate_continuous: a continuous-time replacement for
the substrate's discrete equilibrate, using the Step A Hamiltonian.

This is a SIDE-BY-SIDE candidate, not a swap-in replacement. It reads
substrate state, derives (omega, zeta, J) per the Step A formulas,
evolves under the Lindblad master equation, and writes the resulting
alpha values back as scalar scores. The substrate's tree topology,
ledger, and stake mechanics are not touched.

Architecture:

  substrate state              continuous-time evolution
  -----------------            -------------------------
  per-root sub-claims  ----->  omega, zeta (per root)
  cross-root pairs     ----->  J_ab
  pending observations ----->  jump operators L_j with rates gamma_j
  current scores       ----->  rho_0 (initial density matrix)

  evolve(rho_0, H, jump_ops, t)
                       ----->  rho_t
                       ----->  alpha_a per root
                       ----->  written back as substrate score

The mapping isn't lossless: the substrate stores stake-history per
node, while we collapse to (omega, zeta) per root. That's intentional:
the Hamiltonian summarizes the *current* state of the graph; history
lives in the ledger.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Substrate
sys.path.insert(0, r"C:\code\autonet")
sys.path.insert(0, r"C:\code\world-model")

from world_model.generalized import World, GeneralizedTendency  # type: ignore

# Lindblad kernel
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lindblad.lindblad_kernel import (  # type: ignore
    I2, SIGMA_X, SIGMA_Z,
    RAISE_TO_PRO, LOWER_TO_CON,
    evolve, kron, pure_state, maximally_mixed,
    population_pro,
)
from lindblad.step_b_predictions import (  # type: ignore
    SubClaim, omega_for_root, zeta_for_root, J_for_root_pair,
    single_qubit_op, two_qubit_zz, marginal,
)
from lindblad.step_d_params import (  # type: ignore
    omega_for_root_d, zeta_for_root_d, J_for_root_pair_d,
    total_weight, gamma_modulator,
)


# ---------------------------------------------------------------------------
# Substrate -> sub-claim view
# ---------------------------------------------------------------------------


def extract_subclaims(world: World) -> Dict[str, List[SubClaim]]:
    """For each tendency in the world, return the list of sub-claims
    (direct children of the root). Stakes are signed by position;
    capacity comes from tendency.node_capacity. Coords come from the
    tendency's claim anchors.
    """
    out: Dict[str, List[SubClaim]] = {}
    for tid, tendency in world.tendencies.items():
        claims: List[SubClaim] = []
        root = tendency.tree.root_node
        for child in root.all_children:
            # Signed stake: positive for PRO position, negative for CON.
            # Magnitude = absolute net_score on the subtree (this captures
            # the full weight of evidence below the sub-claim, not just
            # the direct stake).
            sign = +1.0 if child.position.value == "pro" else -1.0
            stake_magnitude = abs(child.net_score)
            stake = sign * stake_magnitude
            # Capacity from the tendency's smooth-promotion machinery.
            cap = tendency.node_capacity.get(child.id, 0.0)
            # Coordinates from the claim anchor.
            claim_obj = tendency._node_to_claim.get(child.id)
            if claim_obj is None or not claim_obj.anchor:
                # Fall back to the root anchor (sub-claim sits at root coord)
                coords = tendency.anchor
            else:
                coords = claim_obj.anchor
            claims.append(SubClaim(stake=stake, capacity=cap, coords=tuple(coords)))
        out[tid] = claims
    return out


# ---------------------------------------------------------------------------
# Sub-claim view -> Hamiltonian parameters
# ---------------------------------------------------------------------------


def hamiltonian_params(
    subclaims_per_root: Dict[str, List[SubClaim]],
    bandwidth: float = 0.5,
    kappa: float = 1.0,
    lam: float = 1.0,
    mu: float = 1.0,
    mode: str = "stepA",
) -> Tuple[List[str], List[float], List[float], Dict[Tuple[int, int], float]]:
    """Compute (root_ids, omegas, zetas, Js) from sub-claim configurations.

    mode = "stepA": original unbounded formulas (kept for comparison).
    mode = "stepD": bounded direction/tension formulas with confidence
                    separation. Recommended.
    """
    root_ids = sorted(subclaims_per_root.keys())
    if mode == "stepD":
        omegas = [omega_for_root_d(subclaims_per_root[r], kappa=kappa) for r in root_ids]
        zetas = [zeta_for_root_d(subclaims_per_root[r], lam=lam) for r in root_ids]
    else:
        omegas = [omega_for_root(subclaims_per_root[r], kappa=kappa) for r in root_ids]
        zetas = [zeta_for_root(subclaims_per_root[r], lam=lam) for r in root_ids]
    Js: Dict[Tuple[int, int], float] = {}
    for i in range(len(root_ids)):
        for j in range(i + 1, len(root_ids)):
            if mode == "stepD":
                J = J_for_root_pair_d(
                    subclaims_per_root[root_ids[i]],
                    subclaims_per_root[root_ids[j]],
                    mu=mu, bandwidth=bandwidth,
                )
            else:
                J = J_for_root_pair(
                    subclaims_per_root[root_ids[i]],
                    subclaims_per_root[root_ids[j]],
                    mu=mu, bandwidth=bandwidth,
                )
            if abs(J) > 1e-9:
                Js[(i, j)] = J
    return root_ids, omegas, zetas, Js


def assemble_full_hamiltonian(
    omegas: List[float],
    zetas: List[float],
    Js: Dict[Tuple[int, int], float],
) -> np.ndarray:
    """H = sum_a (-omega_a sigma_z^a + zeta_a sigma_x^a)
        + sum_{a<b} (-J_ab) sigma_z^a tensor sigma_z^b
    """
    N = len(omegas)
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    for a in range(N):
        H = H + (-omegas[a]) * single_qubit_op(N, a, SIGMA_Z)
        H = H + zetas[a] * single_qubit_op(N, a, SIGMA_X)
    for (a, b), J in Js.items():
        if a > b:
            a, b = b, a
        H = H + (-J) * two_qubit_zz(N, a, b)
    return H


# ---------------------------------------------------------------------------
# Substrate score <-> alpha mapping
# ---------------------------------------------------------------------------


def alpha_from_substrate_score(score: float) -> float:
    """Map substrate net_score (unbounded) -> alpha in [0,1] via sigmoid."""
    if score > 30:
        return 1.0
    if score < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-score))


def substrate_score_from_alpha(alpha: float) -> float:
    """Inverse: alpha in [0,1] -> substrate net_score via logit."""
    eps = 1e-6
    a = max(eps, min(1.0 - eps, alpha))
    return math.log(a / (1.0 - a))


def _root_coherence_from_n(world: World, tid: str) -> float:
    """Average n across the root's direct sub-claims. Returns 0 if no
    sub-claims (no coherence). This is the per-root scalar that we
    use to initialize the off-diagonal of rho.

    Average chosen because we want a "typical surprise level" for the
    root region. Max would over-weight one disturbed sub-claim;
    average reflects the region's overall confidence.
    """
    tendency = world.tendencies[tid]
    root = tendency.tree.root_node
    children = root.all_children
    if not children:
        return 0.0
    return sum(c.n for c in children) / len(children)


def initial_density_matrix(
    world: World,
    root_ids: List[str],
    use_novelty: bool = True,
) -> np.ndarray:
    """Build the initial joint density matrix from per-root marginals
    that include both population (from score) AND coherence (from n).

    Per-root rho_i:
      alpha_i  = sigmoid(net_score_i)              (population in PRO)
      c_i      = n_root_i * sqrt(alpha_i * (1-alpha_i))  (coherence)

    where n_root_i is the average n of the root's direct sub-claims.

    The coherence formula keeps rho positive semi-definite by
    construction: |c|^2 <= alpha(1-alpha) since 0 <= n <= 1. n=0 gives
    a fully decohered classical mixture; n=1 gives maximum coherence
    for the given populations.

    The joint state is the tensor product of per-root marginals -- it
    assumes initial independence between roots, which loses cross-root
    correlations but is the natural starting point. Hamiltonian
    evolution introduces correlations as we go.

    use_novelty=False reverts to the diagonal-only initial state
    (the Step C/D behavior) for comparison.
    """
    N = len(root_ids)
    rho = None
    for tid in root_ids:
        a = alpha_from_substrate_score(world.tendencies[tid].tree.score)
        if use_novelty:
            n_root = _root_coherence_from_n(world, tid)
            c = n_root * math.sqrt(max(0.0, a * (1.0 - a)))
            rho_i = np.array([[a, c], [c, 1 - a]], dtype=complex)
        else:
            rho_i = np.diag([a, 1 - a]).astype(complex)
        rho = rho_i if rho is None else np.kron(rho, rho_i)
    return rho


# ---------------------------------------------------------------------------
# Observation -> jump operator
# ---------------------------------------------------------------------------


def observation_jump_ops(
    world: World,
    root_ids: List[str],
    bandwidth: float = 0.5,
    base_gamma: float = 0.5,
    weight_modulation: bool = False,
    subclaims_per_root: Dict[str, List[SubClaim]] | None = None,
    W_scale: float = 1.0,
) -> List[Tuple[np.ndarray, float]]:
    """Translate pending observations into Lindblad jump operators.

    With weight_modulation=True (Step D semantics), gamma is reduced for
    heavily-evidenced roots: gamma_eff = gamma_base * 1/(1 + W/W_scale).
    Confident roots resist new observations.
    """
    N = len(root_ids)
    ops: List[Tuple[np.ndarray, float]] = []
    weights: Dict[str, float] = {}
    if weight_modulation and subclaims_per_root is not None:
        for r in root_ids:
            weights[r] = total_weight(subclaims_per_root[r])
    for obs in world.observations.values():
        for idx, tid in enumerate(root_ids):
            tendency = world.tendencies[tid]
            anchor = tendency.anchor
            polarity = tendency.polarity_axis
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(obs.coords, anchor)))
            kernel = math.exp(-d * d / (2.0 * bandwidth * bandwidth))
            gamma = base_gamma * kernel
            if weight_modulation:
                gamma = gamma * gamma_modulator(weights.get(tid, 0.0), W_scale)
            if gamma < 1e-4:
                continue
            dot = sum(o * p for o, p in zip(obs.coords, polarity))
            if dot > 0:
                L = single_qubit_op(N, idx, RAISE_TO_PRO)
            elif dot < 0:
                L = single_qubit_op(N, idx, LOWER_TO_CON)
            else:
                continue
            ops.append((L, gamma))
    return ops


# ---------------------------------------------------------------------------
# Top-level continuous equilibrate
# ---------------------------------------------------------------------------


def equilibrate_continuous(
    world: World,
    t_total: float = 1.0,
    dt: float = 1e-3,
    bandwidth: float = 0.5,
    kappa: float = 1.0,
    lam: float = 1.0,
    mu: float = 1.0,
    base_gamma: float = 0.5,
    write_back: bool = True,
    mode: str = "stepA",
    W_scale: float = 1.0,
    use_novelty_in_rho0: bool = True,
) -> Dict[str, Any]:
    """Evolve the world's per-root scores under a Lindblad master equation
    derived from the current sub-claim configuration.

    mode = "stepA" or "stepD". Step D recommended -- direction-only omega
    plus confidence-modulated gamma.
    W_scale only matters in stepD mode: half-rate observations occur at
    W = W_scale.
    """
    subs = extract_subclaims(world)
    root_ids, omegas, zetas, Js = hamiltonian_params(
        subs, bandwidth=bandwidth, kappa=kappa, lam=lam, mu=mu, mode=mode,
    )
    H = assemble_full_hamiltonian(omegas, zetas, Js)
    jump_ops = observation_jump_ops(
        world, root_ids, bandwidth=bandwidth, base_gamma=base_gamma,
        weight_modulation=(mode == "stepD"),
        subclaims_per_root=subs if mode == "stepD" else None,
        W_scale=W_scale,
    )
    rho_0 = initial_density_matrix(world, root_ids, use_novelty=use_novelty_in_rho0)
    initial_alpha = [alpha_from_substrate_score(world.tendencies[r].tree.score)
                     for r in root_ids]
    rho_t, _ = evolve(rho_0, H, jump_ops, t_total=t_total, dt=dt)
    N = len(root_ids)
    final_alpha: List[float] = []
    for idx in range(N):
        rho_i = marginal(rho_t, N=N, keep=idx)
        final_alpha.append(population_pro(rho_i))

    if write_back:
        # Map alpha back to net_score and apply as a single bulk stake
        # against the root node from a synthetic agent. Done via direct
        # score override using a stake adjustment.
        for idx, tid in enumerate(root_ids):
            target_score = substrate_score_from_alpha(final_alpha[idx])
            # Direct override: reset stakes on the root and add a single
            # net_score-equivalent stake. This replaces the substrate's
            # stake-accumulation result with the Lindblad-derived score.
            tendency = world.tendencies[tid]
            root = tendency.tree.root_node
            # Compute current root.net_score; adjust by delta to reach target.
            current = root.net_score
            delta = target_score - current
            if abs(delta) > 1e-6:
                root.add_stake(agent_id="_lindblad_continuous", weight=delta)

    return {
        "root_ids": root_ids,
        "omegas": omegas,
        "zetas": zetas,
        "Js": {f"{a}-{b}": v for (a, b), v in Js.items()},
        "weights": [total_weight(subs[r]) for r in root_ids],
        "n_observations": len(world.observations),
        "n_jump_ops": len(jump_ops),
        "dim": 2 ** N,
        "initial_alpha": initial_alpha,
        "final_alpha": final_alpha,
        "t_total": t_total,
        "dt": dt,
        "mode": mode,
    }
