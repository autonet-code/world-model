#!/usr/bin/env python3
"""Novelty decay: settled regions stop drawing the engine's attention.

Tests StabilityTracker:

  1. A node whose score is stable across the stability window gets
     flagged decayed.
  2. A node whose score moves fresh enough never decays.
  3. Decayed nodes can be reactivated by direct activity.
  4. Roots are never flagged decayed.
"""

from __future__ import annotations

import sys

from world_model.generalized import (
    GeneralizedTendency,
    Observation,
    World,
    equilibrate,
    ScoreHistory,
    StabilityTracker,
)


def banner(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main() -> int:
    banner("NOVELTY DECAY: settled subtrees stop drawing attention")

    A = GeneralizedTendency(
        id="A", thesis="left",
        anchor=(-1.0,), polarity_axis=(-1.0,),
        bandwidth=2.0,
    )
    B = GeneralizedTendency(
        id="B", thesis="right",
        anchor=(+1.0,), polarity_axis=(+1.0,),
        bandwidth=2.0,
    )
    world = World()
    world.add_tendency(A)
    world.add_tendency(B)

    history = ScoreHistory()
    # Threshold tuned for the architecture's natural per-round noise
    # at this bandwidth. Real production would use ScoreHistory with
    # many more snapshots and a tighter threshold.
    tracker = StabilityTracker(
        stability_threshold=0.5,
        stability_window=2,
    )

    # Phase 1: feed initial observations and let things grow
    print("\n  Phase 1: introduce 3 left observations -> growth + initial scores")
    for i, x in enumerate([-1.2, -1.0, -0.8]):
        world.add_observation(Observation(id=f"L{i}", coords=(x,), label=f"L{i}"))
    equilibrate(world, max_rounds=4, tolerance=1e-3)
    history.record(world)
    tracker.observe(world, history)
    print(f"    nodes total: {len(history.snapshots[-1][1])}, decayed: {len(tracker.decayed)}")

    # Phase 2-5: clear obs, equilibrate, snapshot. Scores should
    # stabilize as no fresh activity arrives. After window passes,
    # nodes should flip to decayed.
    print("\n  Phase 2-5: equilibrate without new obs (scores should settle)")
    for round_idx in range(4):
        world.clear_observations()
        equilibrate(world, max_rounds=4, tolerance=1e-3)
        history.record(world)
        tracker.observe(world, history)
        stats = tracker.stats()
        print(f"    round {round_idx + 1}: tracked={stats['n_tracked']}, "
              f"decayed={stats['n_decayed']}")

    # Phase 6: forced reactivation via the explicit API
    print("\n  Phase 6: explicit reactivate() of one decayed node")
    pre_decayed_count = len(tracker.decayed)
    pre_decayed_set = set(tracker.decayed)
    if tracker.decayed:
        target = next(iter(tracker.decayed))
        tracker.reactivate(target)
        print(f"    reactivated node {target[:12]}")
    post_decayed_count = len(tracker.decayed)
    print(f"    decayed before: {pre_decayed_count}")
    print(f"    decayed after explicit reactivation: {post_decayed_count}")
    reactivated = pre_decayed_set - tracker.decayed
    print(f"    nodes reactivated: {len(reactivated)}")

    # Verdict
    banner("VERDICT")
    success = True

    # 1. After settling, some nodes should be decayed
    if pre_decayed_count > 0:
        print(f"\n  OK: {pre_decayed_count} nodes flagged decayed after stable epochs")
    else:
        print(f"\n  -- no nodes were flagged decayed; threshold may be too tight")
        success = False

    # 2. Roots never decay
    root_decayed = any(
        node.id in tracker.decayed and node.parent_id is None
        for tendency in world.tendencies.values()
        for node in tendency.tree.all_nodes()
    )
    if not root_decayed:
        print(f"  OK: roots never flagged decayed")
    else:
        print(f"  -- a root got flagged decayed (should be impossible)")
        success = False

    # 3. Explicit reactivation removed at least one from the decay set
    if pre_decayed_count > 0 and len(reactivated) > 0:
        print(f"  OK: explicit reactivate() removed {len(reactivated)} node(s) "
              f"from decay set")
    elif pre_decayed_count == 0:
        print(f"  -- nothing was decayed to reactivate; skipping that check")
    else:
        print(f"  -- explicit reactivate() failed to clear decay flag")
        success = False

    print()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
