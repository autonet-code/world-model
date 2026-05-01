#!/usr/bin/env python3
"""
Does smooth promotion earn its keep?

Test design (A/B): two configurations of the same scenario, run
identically except one has smooth_promotion=True and the other has
smooth_promotion=False. Compare final outcomes.

Scenario: emergent sub-cluster influence
-----------------------------------------

Two opposing tendencies, A=left at -1 and B=right at +1. Observations
arrive in three groups across many rounds:

  - "Edge" obs at the extremes (x=-1, x=+1): clear evidence for the
    matching tendency.
  - "Mid-left" obs at x=-0.3: weakly relevant to A. Sprout PRO sub-
    claims under A's root. With smooth promotion ON, these sub-claims
    eventually accumulate enough PRO stake to develop voice and start
    cross-staking.
  - "Mid-right" obs at x=+0.3: same for B.

Stream order: edge first, then alternating mid-clusters, repeated.
After many rounds, the mid-cluster sub-claims should have accumulated
PRO stake; with smooth promotion their cross-staking creates an
ADDITIONAL channel of influence between A and B that didn't exist
through the roots alone (the roots are anchored at the extremes; only
their sub-claims live in the middle).

Hypothesis: smooth promotion produces a measurable difference in
final root scores AFTER the mid-clusters have had time to mature.
Without smooth promotion, mid-cluster sub-claims are silent, so only
the original root staking matters.

Pass criteria: |delta_with - delta_without| > some threshold, where
delta = score_A - score_B.
"""

from __future__ import annotations

from world_model.generalized import (
    GeneralizedTendency,
    Observation,
    World,
    equilibrate,
)


def build_world(smooth: bool) -> World:
    A = GeneralizedTendency(
        id="A", thesis="left",
        anchor=(-1.0,), polarity_axis=(-1.0,),
        budget=1.0, bandwidth=2.0,
        smooth_promotion=smooth,
    )
    B = GeneralizedTendency(
        id="B", thesis="right",
        anchor=(+1.0,), polarity_axis=(+1.0,),
        budget=1.0, bandwidth=2.0,
        smooth_promotion=smooth,
    )
    world = World()
    world.add_tendency(A)
    world.add_tendency(B)
    return world


def run_scenario(smooth: bool, n_rounds: int = 8) -> tuple[dict, int, int, int]:
    world = build_world(smooth)
    A = world.tendencies["A"]
    B = world.tendencies["B"]

    # Phase 1: edge observations -- establish anchors
    for i, x in enumerate([-1.2, -1.0, -0.8, +0.8, +1.0, +1.2]):
        world.add_observation(Observation(id=f"E{i}", coords=(x,), label=f"E{i}"))
    equilibrate(world, max_rounds=4, tolerance=1e-3)

    # Phase 2-N: ASYMMETRIC mid-cluster injections. Only mid-left
    # observations arrive. They sprout PRO sub-claims under A's tree.
    # If smooth promotion is on, after enough rounds these sub-claims
    # accumulate voice and start cross-staking B's tree -- a NEW
    # influence channel that didn't exist via roots alone.
    for r in range(n_rounds):
        for i, x in enumerate([-0.4, -0.3, -0.2]):
            world.add_observation(Observation(id=f"ML{i}", coords=(x,), label=f"ML{i}"))
        equilibrate(world, max_rounds=6, tolerance=1e-3)

    scores = world.root_scores()

    # Count sub-claims with non-trivial capacity
    n_voiced_A = sum(1 for nid, c in A.node_capacity.items()
                     if c >= A.capacity_threshold and nid != A.tree.root_node.id)
    n_voiced_B = sum(1 for nid, c in B.node_capacity.items()
                     if c >= B.capacity_threshold and nid != B.tree.root_node.id)
    total_nodes = len(A.tree.all_nodes()) + len(B.tree.all_nodes())
    return scores, n_voiced_A, n_voiced_B, total_nodes


def main():
    print()
    print("=" * 70)
    print("DOES SMOOTH PROMOTION EARN ITS KEEP?")
    print("=" * 70)
    print("\n  A/B test: identical scenario run with smooth_promotion ON vs OFF.")
    print("  Mid-cluster observations sprout sub-claims that, with smooth")
    print("  promotion, develop voice over many rounds and influence the")
    print("  equilibrium beyond what root staking alone can do.\n")

    print("-" * 70)
    print("OFF: smooth_promotion = False")
    print("-" * 70)
    scores_off, va_off, vb_off, n_off = run_scenario(smooth=False, n_rounds=8)
    print(f"\n  Final scores: A = {scores_off['A']:+.4f}, B = {scores_off['B']:+.4f}")
    print(f"  Delta (A - B) = {scores_off['A'] - scores_off['B']:+.4f}")
    print(f"  Voiced sub-claims (A): {va_off}, (B): {vb_off}")
    print(f"  Total nodes across both trees: {n_off}")

    print()
    print("-" * 70)
    print("ON: smooth_promotion = True")
    print("-" * 70)
    scores_on, va_on, vb_on, n_on = run_scenario(smooth=True, n_rounds=8)
    print(f"\n  Final scores: A = {scores_on['A']:+.4f}, B = {scores_on['B']:+.4f}")
    print(f"  Delta (A - B) = {scores_on['A'] - scores_on['B']:+.4f}")
    print(f"  Voiced sub-claims (A): {va_on}, (B): {vb_on}")
    print(f"  Total nodes across both trees: {n_on}")

    # Compare
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    delta_off = scores_off['A'] - scores_off['B']
    delta_on = scores_on['A'] - scores_on['B']
    diff = abs(delta_on - delta_off)
    voiced_diff = (va_on + vb_on) - (va_off + vb_off)
    print(f"\n  Delta(A-B) without smooth promotion: {delta_off:+.4f}")
    print(f"  Delta(A-B) with smooth promotion:    {delta_on:+.4f}")
    print(f"  Difference:                          {diff:.4f}")
    print(f"\n  Voiced sub-claims (off): {va_off + vb_off}")
    print(f"  Voiced sub-claims (on):  {va_on + vb_on}")
    print(f"  Promotion effect:        {voiced_diff:+d} additional voiced nodes")

    if diff > 0.5:
        print(f"\n  OK: smooth promotion changed the equilibrium by {diff:.4f}.")
        print("  Sub-claims with earned standing exert real influence beyond")
        print("  what root staking alone produces. The mechanism earns its")
        print("  keep -- removing it would lose this signal.")
    elif diff > 0.05:
        print(f"\n  WEAK: smooth promotion produced a measurable but small")
        print(f"  effect ({diff:.4f}). Worth keeping but not load-bearing")
        print("  at this scale.")
    else:
        print("\n  -- Smooth promotion produced no measurable effect on the")
        print("  equilibrium. Consider whether the scenario actually exercises")
        print("  the mechanism, or whether the mechanism is silent by design")
        print("  at this scale.")
    print()


if __name__ == "__main__":
    main()
