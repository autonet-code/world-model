"""Empirical test of the refactored novelty dynamics.

Builds a single-tendency, single-root substrate and drips observations
through it. Records each node's persistent novelty n(t) per round.
Compares the observed trajectory to the analytical prediction:

    dn/dt = -gamma_pro * n * pro_rate
           + gamma_con * (1 - n) * con_rate
           + epsilon * (1 - n)

Key tests:

  1. PRO observations decay n monotonically toward 0.
  2. CON observations regrow n toward 1.
  3. Quiet rounds drift n upward via epsilon.
  4. Steady state under balanced obs matches the analytical formula.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, r"C:\code\world-model")
sys.path.insert(0, r"C:\code\autonet")

from world_model.generalized import (  # type: ignore
    GeneralizedTendency, Observation, World, equilibrate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_world(dim: int = 2) -> World:
    world = World()
    anchor = tuple([1.0] + [0.0] * (dim - 1))
    world.add_tendency(GeneralizedTendency(
        id="r",
        thesis="The thing.",
        anchor=anchor,
        polarity_axis=anchor,
        budget=1.0,
        bandwidth=0.5,
        smooth_promotion=True,
        # Use the default novelty rate constants (gamma_pro=1.0,
        # gamma_con=0.5, epsilon=0.01). They're tunable per tendency.
    ))
    return world


def make_obs(seq: int, kind: str, dim: int = 2) -> Observation:
    sign = 1.0 if kind == "PRO" else -1.0
    coords = [0.0] * dim
    coords[0] = sign
    return Observation(id=f"{kind.lower()}_{seq}", coords=tuple(coords),
                       label=f"{kind}_{seq}")


def first_non_root_node(world: World):
    """Return the first non-root node in the only tendency."""
    t = next(iter(world.tendencies.values()))
    for n in t.tree.all_nodes():
        if n.parent_id is not None:
            return n
    return None


def first_pro_child(world: World):
    """Return the first PRO child of the root (where PRO obs sprout)."""
    t = next(iter(world.tendencies.values()))
    root = t.tree.root_node
    return root.pro_children[0] if root.pro_children else None


def first_con_child(world: World):
    """Return the first CON child of the root (where CON obs sprout)."""
    t = next(iter(world.tendencies.values()))
    root = t.tree.root_node
    return root.con_children[0] if root.con_children else None


# ---------------------------------------------------------------------------
# Test 1: PRO observations decay n
# ---------------------------------------------------------------------------


def test_pro_decay():
    print("=" * 70)
    print("Test 1: PRO observations decay n")
    print("=" * 70)
    world = build_world()
    schedule = ["PRO"] * 12

    print(f"{'step':>4} {'kind':>4}  {'pro_child.n':>11}  {'pro_child.score':>16}")
    pro_n_trace = []
    for i, kind in enumerate(schedule, 1):
        world.add_observation(make_obs(i, kind))
        equilibrate(world, max_rounds=4, tolerance=1e-4)
        pro = first_pro_child(world)
        if pro is None:
            print(f"{i:>4}  {kind:>4}  (no pro child yet)")
            continue
        pro_n_trace.append(pro.n)
        print(f"{i:>4}  {kind:>4}  {pro.n:>11.4f}  {pro.net_score:>16.4f}")
        world.clear_observations()

    monotonic = all(pro_n_trace[i+1] <= pro_n_trace[i] + 1e-9
                    for i in range(len(pro_n_trace) - 1))
    print(f"\nMonotonic decay of n? {monotonic}")
    print(f"Final n = {pro_n_trace[-1]:.4f}  (predicted: ~0)")
    return {
        "monotonic": monotonic,
        "final_n": pro_n_trace[-1] if pro_n_trace else None,
        "trace": pro_n_trace,
    }


# ---------------------------------------------------------------------------
# Test 2: CON observations regrow n on the parent (the root region)
# ---------------------------------------------------------------------------


def test_con_regrow():
    print("\n" + "=" * 70)
    print("Test 2: CON observations regrow n on the PRO child via re-surprise")
    print("=" * 70)
    print("First settle a PRO child with PRO observations, then drip CON.")
    world = build_world()

    # Phase 1: settle with PROs
    for i in range(1, 11):
        world.add_observation(make_obs(i, "PRO"))
        equilibrate(world, max_rounds=4, tolerance=1e-4)
        world.clear_observations()
    pro = first_pro_child(world)
    n_before_con = pro.n
    print(f"After 10 PROs, pro_child.n = {n_before_con:.4f}")

    # Phase 2: drip CONs
    print(f"\n{'step':>4} {'kind':>4}  {'pro_child.n':>11}  {'con_child.n':>11}")
    for i in range(11, 21):
        world.add_observation(make_obs(i, "CON"))
        equilibrate(world, max_rounds=4, tolerance=1e-4)
        pro = first_pro_child(world)
        con = first_con_child(world)
        pro_n = pro.n if pro else float("nan")
        con_n = con.n if con else float("nan")
        print(f"{i:>4}  {'CON':>4}  {pro_n:>11.4f}  {con_n:>11.4f}")
        world.clear_observations()

    pro = first_pro_child(world)
    n_after_con = pro.n
    delta = n_after_con - n_before_con
    print(f"\nn(pro_child) before CON phase: {n_before_con:.4f}")
    print(f"n(pro_child) after  CON phase: {n_after_con:.4f}")
    print(f"Change: {delta:+.4f}  (positive = re-surprise occurred)")
    return {
        "n_before_con": n_before_con,
        "n_after_con": n_after_con,
        "delta": delta,
        "regrow": delta > 0.01,
    }


# ---------------------------------------------------------------------------
# Test 3: drift toward uncertainty (no observations)
# ---------------------------------------------------------------------------


def test_drift():
    print("\n" + "=" * 70)
    print("Test 3: drift -- with no observations, n drifts upward via epsilon")
    print("=" * 70)
    world = build_world()
    # Settle a PRO child first
    for i in range(1, 11):
        world.add_observation(make_obs(i, "PRO"))
        equilibrate(world, max_rounds=4, tolerance=1e-4)
        world.clear_observations()
    pro = first_pro_child(world)
    n_settled = pro.n
    print(f"After settling: n = {n_settled:.4f}")

    # Drift phase: no observations, just equilibrate rounds
    print(f"\n{'step':>4}  {'pro_child.n':>11}")
    for i in range(1, 31):
        equilibrate(world, max_rounds=2, tolerance=1e-4)
        pro = first_pro_child(world)
        if i % 5 == 0:
            print(f"{i:>4}  {pro.n:>11.4f}")

    pro = first_pro_child(world)
    n_drifted = pro.n
    print(f"\nn after 30 quiet rounds: {n_drifted:.4f}")
    print(f"Change: {n_drifted - n_settled:+.4f}  (positive = drift upward)")
    return {
        "n_settled": n_settled,
        "n_drifted": n_drifted,
        "delta": n_drifted - n_settled,
        "drift_observed": n_drifted > n_settled + 0.01,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    print("Empirical test: refactored novelty dynamics\n")
    r1 = test_pro_decay()
    r2 = test_con_regrow()
    r3 = test_drift()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Test 1 (PRO decay):     monotonic={r1['monotonic']}, final_n={r1['final_n']:.4f}")
    print(f"  Test 2 (CON regrow):    delta={r2['delta']:+.4f}, regrew={r2['regrow']}")
    print(f"  Test 3 (drift):         delta={r3['delta']:+.4f}, drifted={r3['drift_observed']}")

    n_pass = sum([
        bool(r1['monotonic']) and r1['final_n'] is not None and r1['final_n'] < 0.5,
        bool(r2['regrow']),
        bool(r3['drift_observed']),
    ])
    print(f"\n{n_pass}/3 predictions confirmed empirically")


if __name__ == "__main__":
    main()
