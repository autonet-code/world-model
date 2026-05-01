#!/usr/bin/env python3
"""
Two tendencies, one bit, contested.

The minimum experiment for the generalized model. We set up:

  - A 1D coordinate space (one axis).
  - Tendency A: thesis "left side wins", anchored at x=-1 with
    polarity axis pointing toward -∞ (PRO = more negative).
  - Tendency B: thesis "right side wins", anchored at x=+1 with
    polarity axis pointing toward +∞ (PRO = more positive).
  - A stream of observations along the axis.

Expected behaviour:

  - Observations near x=-1 should INTEGRATE into A's frame and
    CONTRADICT B's. A should stake PRO on its root; B should stake
    CON via cross-staking.
  - Observations near x=+1: symmetric.
  - Observations at x=0 should be ORTHOGONAL or NEUTRAL to both.

What we want to see:

  - Each tendency's root score reflects how much evidence supports
    its thesis (left observations boost A, right observations boost B).
  - Cross-tendency stakes appear: A puts CON on B's root when B has
    integrated evidence A rejects.
  - Equilibration converges in a small number of rounds.
  - Adding more observations on one side shifts the balance.

Why this matters: it tests the core architecture (tendency = thesis +
frame + budget; cross-staking; novelty-driven absorption) on the
simplest possible domain. If this doesn't work, none of the more
elaborate cases can.
"""

from __future__ import annotations

import math

from world_model.generalized import (
    GeneralizedTendency,
    Observation,
    World,
    equilibrate,
)


def banner(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def fmt_stakes(stakes: dict) -> str:
    if not stakes:
        return "(none)"
    items = sorted(stakes.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    return ", ".join(f"{k[0]}/{k[1][:8]}={v:+.3f}" for k, v in items)


def main():
    banner("TWO TENDENCIES, ONE AXIS")

    # 1D world
    A = GeneralizedTendency(
        id="A",
        thesis="left",
        anchor=(-1.0,),
        polarity_axis=(-1.0,),       # PRO = more negative
        budget=1.0,
        bandwidth=0.8,
    )
    B = GeneralizedTendency(
        id="B",
        thesis="right",
        anchor=(+1.0,),
        polarity_axis=(+1.0,),       # PRO = more positive
        budget=1.0,
        bandwidth=0.8,
    )

    world = World()
    world.add_tendency(A)
    world.add_tendency(B)

    # ---- Phase 1: balanced observations on both sides ----
    banner("Phase 1: balanced evidence (3 left, 3 right)")
    coords = [-1.2, -1.0, -0.8, +0.8, +1.0, +1.2]
    for i, x in enumerate(coords):
        world.add_observation(Observation(id=f"o{i}", coords=(x,), label=f"x={x:+.2f}"))

    rounds = equilibrate(world, max_rounds=15, tolerance=1e-3)
    print(f"  equilibrated in {rounds} rounds")
    print(f"  scores: A={world.root_scores()['A']:+.3f}, B={world.root_scores()['B']:+.3f}")
    print(f"  A.stakes: {fmt_stakes(A.last_stakes)}")
    print(f"  B.stakes: {fmt_stakes(B.last_stakes)}")
    print(f"  A absorbed obs: {len(A.frame.integrated)}")
    print(f"  B absorbed obs: {len(B.frame.integrated)}")

    # ---- Phase 2: tilt left ----
    banner("Phase 2: add 5 more left observations (no new right)")
    for i, x in enumerate([-1.5, -1.3, -1.1, -0.9, -0.7]):
        world.add_observation(Observation(id=f"L{i}", coords=(x,), label=f"x={x:+.2f}"))
    rounds = equilibrate(world, max_rounds=15, tolerance=1e-3)
    print(f"  equilibrated in {rounds} rounds")
    print(f"  scores: A={world.root_scores()['A']:+.3f}, B={world.root_scores()['B']:+.3f}")
    print(f"  A absorbed obs: {len(A.frame.integrated)}")
    print(f"  B absorbed obs: {len(B.frame.integrated)}")

    # ---- Phase 3: orthogonal observations ----
    banner("Phase 3: add 3 orthogonal observations near x=0")
    for i, x in enumerate([-0.05, 0.0, +0.05]):
        world.add_observation(Observation(id=f"M{i}", coords=(x,), label=f"x={x:+.2f}"))
    rounds = equilibrate(world, max_rounds=15, tolerance=1e-3)
    print(f"  equilibrated in {rounds} rounds")
    print(f"  scores: A={world.root_scores()['A']:+.3f}, B={world.root_scores()['B']:+.3f}")
    print(f"  A absorbed obs: {len(A.frame.integrated)} (should not grow much)")
    print(f"  B absorbed obs: {len(B.frame.integrated)} (should not grow much)")

    # ---- Verdict ----
    banner("VERDICT")
    s = world.root_scores()
    print(f"\n  Phase 1 scores (balanced):  A and B should be roughly comparable")
    print(f"  Phase 2 scores (left-tilt): A should pull ahead of B")
    print(f"  Phase 3 scores (after mid): no large change\n")
    print(f"  Final: A={s['A']:+.3f}, B={s['B']:+.3f}, A-B={s['A']-s['B']:+.3f}")
    if s['A'] > s['B']:
        print("  A is ahead -- consistent with the left-tilt phase. Architecture")
        print("  responds to evidence in the expected direction.")
    else:
        print("  A is NOT ahead despite left-tilt. Either the staking policy is")
        print("  wrong, the polarity axis is misconfigured, or cross-staking is")
        print("  cancelling defense. Diagnose.")
    print()


if __name__ == "__main__":
    main()
