"""Tests for the Lindblad kernel.

Each test compares numerical evolution to a known closed-form solution
or qualitative invariant.
"""

from __future__ import annotations

import math
import numpy as np

from .lindblad_kernel import (
    I2, SIGMA_X, SIGMA_Y, SIGMA_Z,
    RAISE_TO_PRO, LOWER_TO_CON,
    evolve,
    root_hamiltonian,
    maximally_mixed,
    pure_state,
    population_pro,
    coherence_off_diagonal,
    kron,
    partial_trace_2qubit,
    dissipator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_close(actual, expected, tol=1e-3, label=""):
    diff = abs(actual - expected)
    ok = diff < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: actual={actual:.6f} expected={expected:.6f} diff={diff:.2e}")
    return ok


def assert_close_complex(actual, expected, tol=1e-3, label=""):
    diff = abs(actual - expected)
    ok = diff < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: actual={actual} expected={expected} diff={diff:.2e}")
    return ok


# ---------------------------------------------------------------------------
# Test 1 — pure unitary Rabi oscillation
# ---------------------------------------------------------------------------


def test_rabi_oscillation():
    """H = ω σ_x, no jumps. Starting from |PRO⟩, population should
    oscillate as cos²(ωt). Period = π/ω."""
    print("\n=== Test 1: Rabi oscillation (pure unitary) ===")
    omega = 1.0
    H = omega * SIGMA_X
    rho_0 = pure_state([1.0, 0.0])  # |PRO⟩
    # At t = π/(4ω), population should be cos²(π/4) = 0.5.
    t_quarter = math.pi / (4 * omega)
    rho_t, _ = evolve(rho_0, H, jump_ops=[], t_total=t_quarter, dt=1e-4)
    p = population_pro(rho_t)
    ok1 = assert_close(p, 0.5, tol=1e-3, label="population at t=π/4ω")
    # At t = π/(2ω), should be at |CON⟩, p = 0.
    t_half = math.pi / (2 * omega)
    rho_t, _ = evolve(rho_0, H, jump_ops=[], t_total=t_half, dt=1e-4)
    p = population_pro(rho_t)
    ok2 = assert_close(p, 0.0, tol=1e-3, label="population at t=π/2ω")
    # At t = π/ω, back to |PRO⟩.
    t_full = math.pi / omega
    rho_t, _ = evolve(rho_0, H, jump_ops=[], t_total=t_full, dt=1e-4)
    p = population_pro(rho_t)
    ok3 = assert_close(p, 1.0, tol=1e-3, label="population at t=π/ω")
    return all([ok1, ok2, ok3])


# ---------------------------------------------------------------------------
# Test 2 — pure dephasing (projector jump op)
# ---------------------------------------------------------------------------


def test_pure_dephasing():
    """H = 0, L = √γ σ_z (a non-demolition projector-like op that
    kills coherence at rate 2γ but leaves populations alone).

    Starting from a coherent superposition (|PRO⟩+|CON⟩)/√2, populations
    stay at 0.5 each. Coherence c = 0.5 should decay as 0.5·exp(-2γt).
    """
    print("\n=== Test 2: pure dephasing ===")
    gamma = 1.0
    H = np.zeros((2, 2), dtype=complex)
    L = SIGMA_Z  # Hermitian projector-like
    jump_ops = [(L, gamma)]
    rho_0 = pure_state([1.0, 1.0])  # equal superposition
    t = 0.5
    rho_t, _ = evolve(rho_0, H, jump_ops=jump_ops, t_total=t, dt=1e-4)
    p = population_pro(rho_t)
    c = coherence_off_diagonal(rho_t)
    expected_c = 0.5 * math.exp(-2 * gamma * t)
    ok1 = assert_close(p, 0.5, tol=1e-3, label="population stays at 0.5")
    ok2 = assert_close(c.real, expected_c, tol=1e-3, label="coherence decays as exp(-2γt)")
    return all([ok1, ok2])


# ---------------------------------------------------------------------------
# Test 3 — amplitude damping toward PRO
# ---------------------------------------------------------------------------


def test_amplitude_damping():
    """H = 0, L_PRO = √γ |PRO⟩⟨CON|. Starting from |CON⟩ (p=0), population
    should grow as 1 - exp(-γt).

    From the doc: the closed form for L = √γ |PRO⟩⟨CON| is
        dp/dt = γ(1-p), so p(t) = 1 - (1-p_0)·exp(-γt).
    Coherence dc/dt = -γc/2.
    """
    print("\n=== Test 3: amplitude damping toward PRO ===")
    gamma = 2.0
    H = np.zeros((2, 2), dtype=complex)
    jump_ops = [(RAISE_TO_PRO, gamma)]
    rho_0 = pure_state([0.0, 1.0])  # |CON⟩, p = 0
    t = 0.5
    rho_t, _ = evolve(rho_0, H, jump_ops=jump_ops, t_total=t, dt=1e-4)
    p = population_pro(rho_t)
    expected_p = 1.0 - math.exp(-gamma * t)
    ok1 = assert_close(p, expected_p, tol=1e-3, label=f"p(t={t}) approaching 1 at rate γ")
    # Test from a coherent superposition: coherence should decay at γ/2.
    rho_0 = pure_state([1.0, 1.0])  # equal superposition, p_0 = 0.5
    rho_t, _ = evolve(rho_0, H, jump_ops=jump_ops, t_total=t, dt=1e-4)
    c = coherence_off_diagonal(rho_t)
    expected_c = 0.5 * math.exp(-gamma * t / 2.0)
    ok2 = assert_close(abs(c), expected_c, tol=1e-3, label=f"coherence decays at γ/2")
    return all([ok1, ok2])


# ---------------------------------------------------------------------------
# Test 4 — two-channel steady state (the substrate-relevant case)
# ---------------------------------------------------------------------------


def test_two_channel_steady_state():
    """Two channels: γ_+ pulling toward PRO, γ_- pulling toward CON.
    Steady-state population should be γ_+ / (γ_+ + γ_-).
    """
    print("\n=== Test 4: two-channel steady state ===")
    gamma_plus = 3.0
    gamma_minus = 1.0
    H = np.zeros((2, 2), dtype=complex)
    jump_ops = [
        (RAISE_TO_PRO, gamma_plus),
        (LOWER_TO_CON, gamma_minus),
    ]
    rho_0 = maximally_mixed()
    # Run for 10 / (γ_+ + γ_-) — that's exp(-10) ≈ 4.5e-5 of the gap remaining.
    t_total = 10.0 / (gamma_plus + gamma_minus)
    rho_t, _ = evolve(rho_0, H, jump_ops=jump_ops, t_total=t_total, dt=1e-3)
    p = population_pro(rho_t)
    expected_p = gamma_plus / (gamma_plus + gamma_minus)
    ok1 = assert_close(p, expected_p, tol=1e-3,
                       label=f"steady-state p = gamma_+/(gamma_+ + gamma_-) = {expected_p:.3f}")
    return ok1


# ---------------------------------------------------------------------------
# Test 5 — relaxation rate
# ---------------------------------------------------------------------------


def test_relaxation_rate():
    """Single channel γ to PRO, starting from |CON⟩.
    p(t) - p_steady should decay as exp(-γt).

    For a single channel L_PRO = √γ |PRO⟩⟨CON| with γ = 1:
      p(t) = 1 - exp(-γt)
      half-life of the gap (1 - p): ln(2) / γ
    """
    print("\n=== Test 5: relaxation rate (half-life check) ===")
    gamma = 1.0
    H = np.zeros((2, 2), dtype=complex)
    jump_ops = [(RAISE_TO_PRO, gamma)]
    rho_0 = pure_state([0.0, 1.0])
    half_life = math.log(2) / gamma
    rho_t, _ = evolve(rho_0, H, jump_ops=jump_ops, t_total=half_life, dt=1e-4)
    p = population_pro(rho_t)
    # At one half-life, gap (1 - p) should be 0.5 of initial gap.
    # Initial gap is 1.0 (since p_0 = 0), so p(half_life) = 0.5.
    ok = assert_close(p, 0.5, tol=1e-3,
                      label=f"p(t=ln2/γ) = 0.5 (one half-life)")
    return ok


# ---------------------------------------------------------------------------
# Test 6 — trace and Hermiticity preserved
# ---------------------------------------------------------------------------


def test_invariants():
    """ρ should remain Hermitian and trace-1 throughout evolution."""
    print("\n=== Test 6: trace and Hermiticity invariants ===")
    omega = 0.5
    zeta = 0.7
    H = root_hamiltonian(omega, zeta)
    jump_ops = [
        (RAISE_TO_PRO, 0.3),
        (LOWER_TO_CON, 0.4),
    ]
    rho_0 = pure_state([0.6, 0.8])  # off-balance superposition
    rho_t, traj = evolve(rho_0, H, jump_ops=jump_ops, t_total=2.0, dt=1e-3,
                          record_every=200)
    # Check trace and Hermiticity for every snapshot.
    all_ok = True
    for t, rho in traj:
        tr = np.trace(rho).real
        herm_err = np.max(np.abs(rho - rho.conj().T))
        ok_tr = abs(tr - 1.0) < 1e-6
        ok_h = herm_err < 1e-6
        if not (ok_tr and ok_h):
            all_ok = False
            print(f"  [FAIL] t={t:.3f} trace={tr:.6f} herm_err={herm_err:.2e}")
    if all_ok:
        print(f"  [PASS] trace=1 and Hermitian for all {len(traj)} snapshots")
    return all_ok


# ---------------------------------------------------------------------------
# Test 7 — substrate-analog: 4 roots, no cross-coupling, observation pulls
# ---------------------------------------------------------------------------


def test_four_root_independent():
    """Four roots tracked as independent 2-level systems.

    Each root has its own (omega, zeta) and its own observation rates.
    Verify that each root settles to γ_+/(γ_+ + γ_-) independently.

    This validates the marginal-tracking strategy used when
    cross-tendency coupling J_ij = 0.
    """
    print("\n=== Test 7: four independent roots ===")
    # Charter-flavored config: 4 roots, each with different stake/observation profile.
    configs = [
        # (omega, zeta, gamma_plus, gamma_minus, label)
        (0.0, 0.0, 5.0, 1.0, "life_precious  - strong PRO observations"),
        (0.0, 0.0, 1.0, 1.0, "self_pres      - balanced 50/50"),
        (0.0, 0.0, 1.0, 4.0, "promo_intel    - strong CON observations"),
        (0.0, 0.0, 2.0, 2.0, "evolution      - balanced 50/50"),
    ]
    all_ok = True
    for omega, zeta, gp, gm, label in configs:
        H = root_hamiltonian(omega, zeta)
        jump_ops = [(RAISE_TO_PRO, gp), (LOWER_TO_CON, gm)]
        rho_0 = maximally_mixed()
        t_total = 5.0 / (gp + gm)
        rho_t, _ = evolve(rho_0, H, jump_ops=jump_ops, t_total=t_total, dt=1e-3)
        p = population_pro(rho_t)
        expected = gp / (gp + gm)
        ok = assert_close(p, expected, tol=1e-2, label=label)
        all_ok = all_ok and ok
    return all_ok


# ---------------------------------------------------------------------------
# Test 8 — substrate-analog: 2 entangled roots via Ising coupling
# ---------------------------------------------------------------------------


def test_two_root_ising_coupling():
    """Two roots with cross-coupling J σ_z ⊗ σ_z (the locality-rule
    analog from the scaffold). Test that:

      - Without coupling (J=0), independent dynamics: each root's
        marginal evolves as if alone.
      - With coupling (J≠0), the joint state becomes entangled; the
        marginals' purity drops (Tr(ρᵢ²) < 1 even if joint state is pure).
    """
    print("\n=== Test 8: two-root Ising coupling, marginal extraction ===")
    omega1, omega2 = 0.0, 0.0
    zeta1, zeta2 = 1.0, 1.0
    # Build 4×4 Hamiltonian for two qubits.
    H_local = (omega1 * kron(SIGMA_Z, I2) + zeta1 * kron(SIGMA_X, I2)
               + omega2 * kron(I2, SIGMA_Z) + zeta2 * kron(I2, SIGMA_X))

    # Initial state: |PRO⟩ ⊗ |PRO⟩ pure.
    rho_0 = pure_state([1.0, 0.0, 0.0, 0.0])  # 4-vector: |00⟩

    # Case A: J = 0
    H_a = H_local
    rho_a, _ = evolve(rho_0, H_a, jump_ops=[], t_total=1.0, dt=1e-4)
    rho_a_marg0 = partial_trace_2qubit(rho_a, keep=0)
    purity_a = np.trace(rho_a_marg0 @ rho_a_marg0).real

    # Case B: J ≠ 0 → entanglement should reduce marginal purity
    J = 1.0
    H_b = H_local + J * kron(SIGMA_Z, SIGMA_Z)
    rho_b, _ = evolve(rho_0, H_b, jump_ops=[], t_total=1.0, dt=1e-4)
    rho_b_marg0 = partial_trace_2qubit(rho_b, keep=0)
    purity_b = np.trace(rho_b_marg0 @ rho_b_marg0).real

    print(f"  J=0: marginal purity of root 0 = {purity_a:.4f}")
    print(f"  J=1: marginal purity of root 0 = {purity_b:.4f}")
    # Joint state is pure throughout (no jumps); marginal purity ≤ 1.
    # With J=0 and identical local dynamics, root 0 stays in a pure state.
    # With J≠0, the two roots entangle and the marginal becomes mixed.
    ok1 = purity_a > 0.999  # essentially pure
    ok2 = purity_b < 0.95   # noticeably mixed due to entanglement
    print(f"  [{'PASS' if ok1 else 'FAIL'}] J=0 → marginal stays pure")
    print(f"  [{'PASS' if ok2 else 'FAIL'}] J≠0 → marginal becomes mixed (entanglement)")
    return ok1 and ok2


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    tests = [
        ("Rabi oscillation", test_rabi_oscillation),
        ("Pure dephasing", test_pure_dephasing),
        ("Amplitude damping toward PRO", test_amplitude_damping),
        ("Two-channel steady state", test_two_channel_steady_state),
        ("Relaxation rate", test_relaxation_rate),
        ("Invariants", test_invariants),
        ("Four-root independent", test_four_root_independent),
        ("Two-root Ising coupling", test_two_root_ising_coupling),
    ]
    results = []
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as e:
            print(f"\n  [EXCEPTION] {name}: {e}")
            import traceback; traceback.print_exc()
            ok = False
        results.append((name, ok))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n{n_pass}/{len(results)} tests passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
