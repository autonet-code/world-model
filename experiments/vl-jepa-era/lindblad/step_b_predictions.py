"""Step B — falsifiability tests for the Step A Hamiltonian.

For each prediction in STEP_A_HAMILTONIAN.md we build a hand-crafted
substrate configuration, derive (omega, zeta, J) per the formulas,
evolve under the Lindblad master equation, and check whether the
predicted phenomenon shows up.

Predictions tested:
  1. Damped quantum beats during settling (high-tension regime)
  2. Tilted steady states alpha = (1 + omega/h)/2 with h = sqrt(w^2 + z^2)
  3. Cross-root entanglement: observation on root a shifts root b
  4. Reversibility in the unitary regime (gamma = 0)

We don't touch the substrate code — just the math and the kernel.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lindblad.lindblad_kernel import (  # type: ignore
    I2, SIGMA_X, SIGMA_Z,
    RAISE_TO_PRO, LOWER_TO_CON,
    evolve, lindblad_rhs,
    pure_state, maximally_mixed, population_pro, coherence_off_diagonal,
    kron, partial_trace_2qubit,
)


# ---------------------------------------------------------------------------
# Sub-claim configuration → Hamiltonian parameters
# ---------------------------------------------------------------------------


class SubClaim:
    """A sub-claim attached to a root: signed stake + capacity + coords."""
    def __init__(self, stake: float, capacity: float, coords: tuple[float, ...]):
        self.stake = float(stake)         # positive = PRO, negative = CON
        self.capacity = float(capacity)    # in [0, 1]; how settled this sub-claim is
        self.coords = tuple(float(c) for c in coords)


def omega_for_root(claims: Sequence[SubClaim], kappa: float = 1.0) -> float:
    """Net stake bias: omega = kappa * sum(s_i * cap_i)."""
    return kappa * sum(c.stake * c.capacity for c in claims)


def zeta_for_root(claims: Sequence[SubClaim], lam: float = 1.0) -> float:
    """Cross-pole tension: zeta = lambda * sqrt(P * C) where P, C are the
    settled-weighted PRO and CON totals.
    """
    P = sum(max(c.stake, 0.0) * c.capacity for c in claims)
    C = sum(max(-c.stake, 0.0) * c.capacity for c in claims)
    return lam * math.sqrt(P * C)


def gaussian_kernel(d: float, bandwidth: float) -> float:
    return math.exp(-d * d / (2.0 * bandwidth * bandwidth))


def euclidean(p: tuple[float, ...], q: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def J_for_root_pair(
    claims_a: Sequence[SubClaim],
    claims_b: Sequence[SubClaim],
    mu: float = 1.0,
    bandwidth: float = 0.5,
) -> float:
    """Cross-root coupling: J = mu * sum_{i,j} s_i^a s_j^b cap_i cap_j K(d)."""
    total = 0.0
    for ci in claims_a:
        for cj in claims_b:
            d = euclidean(ci.coords, cj.coords)
            total += ci.stake * cj.stake * ci.capacity * cj.capacity * gaussian_kernel(d, bandwidth)
    return mu * total


def single_root_hamiltonian(omega: float, zeta: float) -> np.ndarray:
    """H = -omega sigma_z + zeta sigma_x in the {PRO, CON} basis."""
    return -omega * SIGMA_Z + zeta * SIGMA_X


# ---------------------------------------------------------------------------
# Multi-root tensor builders
# ---------------------------------------------------------------------------


def single_qubit_op(N: int, idx: int, op: np.ndarray) -> np.ndarray:
    """Embed a 2x2 op `op` at qubit `idx` in an N-qubit register."""
    factors = [I2] * N
    factors[idx] = op
    return kron(*factors)


def two_qubit_zz(N: int, a: int, b: int) -> np.ndarray:
    """sigma_z^a tensor sigma_z^b in an N-qubit register."""
    factors = [I2] * N
    factors[a] = SIGMA_Z
    factors[b] = SIGMA_Z
    return kron(*factors)


def assemble_full_hamiltonian(
    omegas: Sequence[float],
    zetas: Sequence[float],
    Js: dict[tuple[int, int], float],
) -> np.ndarray:
    """Build the full N-qubit Hamiltonian.

    H = sum_a (-omega_a sigma_z^a + zeta_a sigma_x^a)
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
# Marginal extraction for arbitrary N
# ---------------------------------------------------------------------------


def marginal(rho: np.ndarray, N: int, keep: int) -> np.ndarray:
    """Partial-trace out all qubits except `keep`. Returns 2x2.

    Uses index reshape: rho is (2,)*N x (2,)*N when reshaped.
    """
    if rho.shape != (2 ** N, 2 ** N):
        raise ValueError(f"rho must be {2**N}x{2**N}, got {rho.shape}")
    shape = (2,) * N + (2,) * N
    rho_t = rho.reshape(shape)
    # Sum over all index pairs except (keep, keep + N).
    out = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            # Build slice: keep=i in row index, keep=j in col index
            # Sum over all other indices being equal.
            idx_row = [slice(None)] * N
            idx_col = [slice(None)] * N
            idx_row[keep] = i
            idx_col[keep] = j
            sub = rho_t[tuple(idx_row + idx_col)]
            # sub now has shape (2,)*(N-1) x (2,)*(N-1); take its trace.
            sub_flat = sub.reshape((2 ** (N - 1), 2 ** (N - 1)))
            out[i, j] = np.trace(sub_flat)
    return out


# ---------------------------------------------------------------------------
# Prediction 1: damped quantum beats
# ---------------------------------------------------------------------------


def test_damped_beats():
    """High-tension single root: omega small, zeta large.
    Add observation channel toward PRO with rate gamma.
    Prediction: alpha(t) shows oscillations on top of decay.
    """
    print("\n=== Prediction 1: damped quantum beats ===")

    # Hand-built sub-claim config with strong tension.
    # 3 PRO sub-claims at stake 1.0, settled (cap=1).
    # 3 CON sub-claims at stake -1.0, settled.
    # Net: omega = 0. P = C = 3. zeta = sqrt(9) = 3.
    claims = [
        SubClaim(1.0, 1.0, (0.0,)),
        SubClaim(1.0, 1.0, (0.0,)),
        SubClaim(1.0, 1.0, (0.0,)),
        SubClaim(-1.0, 1.0, (0.0,)),
        SubClaim(-1.0, 1.0, (0.0,)),
        SubClaim(-1.0, 1.0, (0.0,)),
    ]
    omega = omega_for_root(claims)
    zeta = zeta_for_root(claims)
    print(f"  omega = {omega:.3f}  zeta = {zeta:.3f}")

    H = single_root_hamiltonian(omega, zeta)
    gamma = 0.3   # weak observation rate so we see beats
    jump_ops = [(RAISE_TO_PRO, gamma)]

    rho_0 = pure_state([0.0, 1.0])  # start at CON
    t_total = 8.0
    dt = 1e-3
    rho, traj = evolve(rho_0, H, jump_ops, t_total=t_total, dt=dt, record_every=20)

    times = np.array([t for t, _ in traj])
    alphas = np.array([population_pro(r) for _, r in traj])

    # Detect oscillations: count zero-crossings of (alpha - smoothed_alpha).
    # Smoothing window for "trend"
    window = 100
    if len(alphas) > window:
        kernel = np.ones(window) / window
        alpha_trend = np.convolve(alphas, kernel, mode="same")
        residual = alphas - alpha_trend
        # Zero crossings in residual indicate oscillation
        signs = np.sign(residual[10:-10])  # trim edges
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        print(f"  zero-crossings in residual: {crossings} (>=4 = oscillation present)")

    print(f"  alpha(0) = {alphas[0]:.4f}")
    print(f"  alpha(end) = {alphas[-1]:.4f}")
    print(f"  alpha range during evolution: [{alphas.min():.4f}, {alphas.max():.4f}]")
    # Oscillation signature: range > some threshold proportional to zeta/gamma ratio.
    # Without zeta, alpha would monotonically increase to gamma_+/(gamma_+ + 0) = 1.
    # With zeta, we expect alpha to overshoot/undershoot during transient.
    monotonic = all(alphas[i] <= alphas[i+1] + 1e-9 for i in range(len(alphas) - 1))
    print(f"  monotonic: {monotonic}")

    return {
        "test": "damped_beats",
        "omega": omega,
        "zeta": zeta,
        "monotonic": monotonic,
        "alpha_range": [float(alphas.min()), float(alphas.max())],
        "alpha_trajectory": (times.tolist(), alphas.tolist()),
    }


# ---------------------------------------------------------------------------
# Prediction 2: tilted steady states
# ---------------------------------------------------------------------------


def test_tilted_steady_state():
    """Single root with non-zero omega AND zeta. Predict steady-state alpha
    = (1 + omega/h)/2 where h = sqrt(omega^2 + zeta^2).

    With observation channels gamma_+ and gamma_-, the actual steady state
    is shifted from the pure-classical gamma_+/(gamma_+ + gamma_-).
    """
    print("\n=== Prediction 2: tilted steady states ===")

    # IMPORTANT: any open system with symmetric jump channels (gamma_+ == gamma_-)
    # has steady state alpha = 0.5 regardless of H, because the dissipator's fixed
    # point is the maximally mixed state. The tilted ground-state alpha is what
    # emerges in the gamma -> 0 strict limit, NOT in the steady-state-with-obs
    # limit. To see the tilt empirically, we use ASYMMETRIC obs rates.
    # The prediction with asymmetric obs: alpha_steady should be a balance
    # between gamma_+/(gamma_+ + gamma_-) (classical pull) and the H-induced
    # ground-state tilt (1 + omega/h)/2.
    test_cases = [
        # (omega, zeta, gamma_plus, gamma_minus, label)
        (1.0, 0.0, 0.0, 0.0, "omega only, no obs (pure unitary, no steady state)"),
        (1.0, 0.5, 0.1, 0.4, "omega > zeta, asymmetric CON-favoring obs"),
        (0.5, 1.0, 0.1, 0.4, "zeta > omega, asymmetric CON-favoring obs"),
        (1.0, 1.0, 0.1, 0.4, "omega = zeta, asymmetric CON-favoring obs"),
        (2.0, 0.5, 0.1, 0.4, "strong PRO bias, CON-favoring obs"),
        (0.0, 1.0, 0.1, 0.4, "no bias, only tension, asymmetric obs"),
        # Control: same asymmetric obs but no H at all - should give pure classical alpha
        (0.0, 0.0, 0.1, 0.4, "no H (pure classical), asymmetric obs"),
    ]

    out = []
    for omega, zeta, gp, gm, label in test_cases:
        h = math.sqrt(omega ** 2 + zeta ** 2)
        alpha_ground = (1.0 + omega / h) / 2.0 if h > 0 else 0.5

        H = single_root_hamiltonian(omega, zeta)
        jump_ops = []
        if gp > 0: jump_ops.append((RAISE_TO_PRO, gp))
        if gm > 0: jump_ops.append((LOWER_TO_CON, gm))
        rho_0 = maximally_mixed()
        t_total = 80.0  # long enough for steady state under weak obs
        rho_ss, _ = evolve(rho_0, H, jump_ops, t_total=t_total, dt=1e-3)
        alpha_ss = population_pro(rho_ss)

        alpha_classical = gp / (gp + gm) if (gp + gm) > 0 else float("nan")
        print(f"  {label}")
        print(f"    omega={omega} zeta={zeta} gp={gp} gm={gm} h={h:.3f}")
        print(f"    ground-state alpha (no obs): {alpha_ground:.4f}")
        print(f"    classical alpha (gp/(gp+gm)): {alpha_classical:.4f}")
        print(f"    actual steady-state alpha:   {alpha_ss:.4f}")
        out.append({
            "label": label,
            "omega": omega, "zeta": zeta, "gamma_plus": gp, "gamma_minus": gm,
            "alpha_ground": alpha_ground,
            "alpha_classical": alpha_classical,
            "alpha_steady": alpha_ss,
        })

    return out


# ---------------------------------------------------------------------------
# Prediction 3: cross-root entanglement
# ---------------------------------------------------------------------------


def test_cross_root_entanglement():
    """Two roots with cross-coupling J. Apply observation to root a only.
    Predict: root b's alpha should also shift, with magnitude proportional
    to |J|.
    """
    print("\n=== Prediction 3: cross-root entanglement ===")

    # Configuration: two roots, each with mixed PRO/CON sub-claims at
    # nearby coordinates. The mixed stake gives non-zero zeta (transverse
    # field), so each root has dynamics on its own. The cross-coupling
    # J should then propagate dynamics from one to the other.
    claims_a = [
        SubClaim(1.0, 1.0, (0.0,)),    # PRO sub-claim
        SubClaim(-0.5, 1.0, (0.0,)),   # weaker CON sub-claim
    ]
    claims_b = [
        SubClaim(1.0, 1.0, (0.1,)),    # PRO sub-claim near a's
        SubClaim(-0.5, 1.0, (0.1,)),   # weaker CON sub-claim near a's
    ]
    omega_a = omega_for_root(claims_a)
    omega_b = omega_for_root(claims_b)
    zeta_a = zeta_for_root(claims_a)
    zeta_b = zeta_for_root(claims_b)
    J = J_for_root_pair(claims_a, claims_b, mu=1.0, bandwidth=0.5)
    print(f"  omega_a = {omega_a:.3f}, omega_b = {omega_b:.3f}")
    print(f"  zeta_a  = {zeta_a:.3f}, zeta_b  = {zeta_b:.3f}")
    print(f"  J_ab    = {J:.3f}  (ferromagnetic since J > 0)")

    # Strategy: NO observations on b. Only on a. Compare b's TRAJECTORY
    # under J=0 vs J=high. With J=0, b is decoupled from a's dynamics
    # and just oscillates around its initial state under H_b alone.
    # With J>0, the time-dependent <sigma_z^a(t)> creates an effective
    # field on b that should drive observable oscillations in alpha_b.
    # Entanglement signature = max excursion of alpha_b from its initial value.
    gamma_a_con = 0.5
    L_a_con = single_qubit_op(2, 0, LOWER_TO_CON)

    out_cases = []
    for J_value, label in [(0.0, "J=0 baseline"), (J, f"J={J:.3f} coupling")]:
        H = assemble_full_hamiltonian(
            omegas=[omega_a, omega_b],
            zetas=[zeta_a, zeta_b],
            Js={(0, 1): J_value},
        )
        jump_ops = [(L_a_con, gamma_a_con)]
        rho_0 = pure_state([1.0, 0.0, 0.0, 0.0])  # |PRO, PRO>
        t_total = 8.0
        # Record trajectory so we can see b's response over time.
        rho_t, traj = evolve(rho_0, H, jump_ops, t_total=t_total, dt=1e-3,
                              record_every=20)
        alpha_a_traj = []
        alpha_b_traj = []
        for t, rho in traj:
            rho_a_t = marginal(rho, N=2, keep=0)
            rho_b_t = marginal(rho, N=2, keep=1)
            alpha_a_traj.append(population_pro(rho_a_t))
            alpha_b_traj.append(population_pro(rho_b_t))
        b_min = min(alpha_b_traj)
        b_max = max(alpha_b_traj)
        b_excursion = b_max - b_min  # max minus min of b over the run
        print(f"  {label}")
        print(f"    alpha_a final: {alpha_a_traj[-1]:.4f}")
        print(f"    alpha_b final: {alpha_b_traj[-1]:.4f}")
        print(f"    alpha_b range: [{b_min:.4f}, {b_max:.4f}]  excursion = {b_excursion:.4f}")
        out_cases.append({
            "label": label, "J": float(J_value),
            "alpha_a_final": alpha_a_traj[-1],
            "alpha_b_final": alpha_b_traj[-1],
            "alpha_b_excursion": b_excursion,
        })

    # The signature: with J > 0, root b's excursion should be larger
    # than with J = 0, because it's dynamically coupled to a.
    excursion_no_J = out_cases[0]["alpha_b_excursion"]
    excursion_with_J = out_cases[1]["alpha_b_excursion"]
    delta = excursion_with_J - excursion_no_J
    final_no_J = out_cases[0]["alpha_b_final"]
    final_with_J = out_cases[1]["alpha_b_final"]
    delta_final = final_no_J - final_with_J
    print(f"  delta_alpha_b excursion: {delta:+.4f}  (positive = J amplifies b's response)")
    print(f"  delta_alpha_b final state: {delta_final:+.4f}")
    print(f"  prediction holds: {delta > 0.01 or abs(delta_final) > 0.01}")

    return out_cases


# ---------------------------------------------------------------------------
# Prediction 4: reversibility in the unitary regime
# ---------------------------------------------------------------------------


def test_reversibility():
    """With gamma = 0 (no observations), evolution is purely unitary.
    Evolve forward then backward. State should return to initial.

    This isn't a 'prediction' in the sense of testing the substrate vs.
    classical. It's a sanity check that the Hamiltonian-only regime
    is well-formed.
    """
    print("\n=== Prediction 4: reversibility under unitary evolution ===")

    # Hand-built two-root config with non-trivial cross-coupling.
    omega_a, omega_b = 0.5, -0.3
    zeta_a, zeta_b = 0.7, 0.4
    J = 0.6

    H = assemble_full_hamiltonian(
        omegas=[omega_a, omega_b],
        zetas=[zeta_a, zeta_b],
        Js={(0, 1): J},
    )
    rho_0 = pure_state([0.5, 0.5, 0.5, 0.5])  # uniform superposition

    # Evolve forward
    t = 5.0
    rho_fwd, _ = evolve(rho_0, H, jump_ops=[], t_total=t, dt=1e-3)
    # Evolve backward by reversing the sign of H (equivalent to running -t).
    rho_bwd, _ = evolve(rho_fwd, -H, jump_ops=[], t_total=t, dt=1e-3)

    # rho_bwd should equal rho_0 within numerical error.
    err = np.linalg.norm(rho_bwd - rho_0)
    print(f"  forward+backward error: {err:.6e}")
    print(f"  reversibility holds: {err < 1e-3}")
    return {"reversibility_error": float(err)}


# ---------------------------------------------------------------------------
# Plotting (Prediction 1's beats)
# ---------------------------------------------------------------------------


def plot_beats(result: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable; skipping plot)")
        return
    times, alphas = result["alpha_trajectory"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, alphas, "-", linewidth=1.5, label="alpha(t)")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel("alpha = <PRO|rho|PRO>")
    ax.set_title(f"Prediction 1: damped beats (omega={result['omega']:.2f}, zeta={result['zeta']:.2f})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path = HERE / "step_b_beats.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("Step B — falsifiability tests for the Step A Hamiltonian")
    print("=" * 72)

    results = {}
    results["pred1_beats"] = test_damped_beats()
    plot_beats(results["pred1_beats"])
    results["pred2_steady"] = test_tilted_steady_state()
    results["pred3_entanglement"] = test_cross_root_entanglement()
    results["pred4_reversibility"] = test_reversibility()

    import json
    out_path = HERE / "step_b_results.json"
    # Strip the heavy alpha_trajectory before persisting summary
    p1 = dict(results["pred1_beats"])
    p1.pop("alpha_trajectory", None)
    summary = {
        "pred1_beats": p1,
        "pred2_steady": results["pred2_steady"],
        "pred3_entanglement": results["pred3_entanglement"],
        "pred4_reversibility": results["pred4_reversibility"],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"results written to {out_path}")


if __name__ == "__main__":
    main()
