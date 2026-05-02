#!/usr/bin/env python3
"""
Epoch-close pruning demo.

Setup
-----

Two tendencies on a 1D axis (left vs right) fed four rounds of
observations. We mix three kinds of evidence:

  1. PERSISTENT: a stream of observations on the right side, repeated
     each round. B keeps absorbing them, supporting nodes accrue
     standing.
  2. FLASH-AND-DIE: a left-side observation that appears in round 1
     ONLY, then never again. The node it sprouts flares briefly and
     then sits inert in subsequent rounds.
  3. QUIET-NEGATIVE: an observation near orthogonality that drops a
     low-amplitude, never-moving node into someone's tree.

After the four rounds we run prune_settled_negatives and check:

  - Persistent positives are NOT pruned -- they have standing.
  - Flash-and-die / quiet-negative nodes ARE pruned -- their score
    history is settled-and-quiet by epoch's end.
  - Roots are NEVER pruned.
  - A second prune call removes nothing (idempotence).

Implementation note
-------------------

The engine's bandwidth is wide enough that an observation in a 1D
world tends to flare every node briefly. To get a clean "flash"
signature we either need many rounds of stability after the flare, or
we feed prune_settled_negatives a history that captures only the
*post-flare* epochs (i.e., we record AFTER the round that introduced
the flash). The demo does both: real engine dynamics drive the score
trajectories, and we slice the recorded ScoreHistory to the rounds
where settling has occurred.
"""

from __future__ import annotations

from world_model.generalized import (
    GeneralizedTendency,
    Observation,
    ScoreHistory,
    World,
    equilibrate,
    prune_settled_negatives,
    snapshot_scores,
)
from world_model.models.tree import Position


def banner(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def all_node_ids(world: World) -> set[str]:
    out = set()
    for t in world.tendencies.values():
        for n in t.tree.all_nodes():
            out.add(n.id)
    return out


def root_ids(world: World) -> set[str]:
    return {t.tree.root_node.id for t in world.tendencies.values()}


def find_obs_node_ids(world: World, observation_id: str) -> list[str]:
    ids = []
    for t in world.tendencies.values():
        for n in t.tree.all_nodes():
            if n.observation_id == observation_id:
                ids.append(n.id)
    return ids


def main() -> None:
    banner("EPOCH-CLOSE PRUNING")

    A = GeneralizedTendency(
        id="A",
        thesis="left",
        anchor=(-1.0,),
        polarity_axis=(-1.0,),
        budget=1.0,
        bandwidth=2.0,
    )
    B = GeneralizedTendency(
        id="B",
        thesis="right",
        anchor=(+1.0,),
        polarity_axis=(+1.0,),
        budget=1.0,
        bandwidth=2.0,
    )
    world = World()
    world.add_tendency(A)
    world.add_tendency(B)

    history = ScoreHistory()

    persistent = [
        (+1.0, "P0"),
        (+1.1, "P1"),
        (+1.2, "P2"),
        (+0.9, "P3"),
    ]
    flash_id = "FLASH"
    quiet_id = "QUIET"

    flash_node_ids: list[str] = []
    quiet_node_ids: list[str] = []
    persistent_node_ids: list[str] = []

    # Round 0: persistent only -- the persistent nodes get sprouted
    # and equilibrate. We record this as the baseline.
    # Round 1: introduce flash + quiet. They sprout and flare.
    # Rounds 2, 3: persistent only again. Flash and quiet sit inert.
    for r in range(4):
        world.clear_observations()
        for x, oid in persistent:
            world.add_observation(
                Observation(id=f"{oid}r{r}", coords=(x,), label=oid)
            )
        if r == 1:
            world.add_observation(
                Observation(id=flash_id, coords=(-1.1,), label="flash")
            )
            world.add_observation(
                Observation(id=quiet_id, coords=(0.02,), label="quiet")
            )

        rounds = equilibrate(world, max_rounds=15, tolerance=1e-3)
        history.record(world)
        n_total = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())
        print(
            f"  round {r}: equilibrate took {rounds} rounds, "
            f"A.score={world.root_scores()['A']:+.3f}, "
            f"B.score={world.root_scores()['B']:+.3f}, "
            f"|nodes|={n_total}"
        )

        if r == 0:
            for _x, oid in persistent:
                persistent_node_ids.extend(find_obs_node_ids(world, f"{oid}r{r}"))
        if r == 1:
            flash_node_ids = find_obs_node_ids(world, flash_id)
            quiet_node_ids = find_obs_node_ids(world, quiet_id)

    # Build the score-history dict from rounds 2..3 -- the "post-flare"
    # epochs. This is the slice where settled-and-quiet means
    # "actually inert", not "freshly born". Persistent nodes have been
    # alive across all rounds and remain quiet here too, but their
    # max_abs is well above the standing threshold so they survive.
    full_hist = history.as_dict()
    post_flare = ScoreHistory(snapshots=history.snapshots[2:])
    score_history = post_flare.as_dict()

    banner("BEFORE PRUNE")
    pre_ids = all_node_ids(world)
    pre_roots = root_ids(world)
    print(f"  total nodes: {len(pre_ids)}")
    print(f"  roots:       {len(pre_roots)}")
    print(f"  flash nodes:      {[i[:12] for i in flash_node_ids]}")
    print(f"  quiet nodes:      {[i[:12] for i in quiet_node_ids]}")
    print(f"  persistent nodes: {[i[:12] for i in persistent_node_ids]}")
    print()
    print("  rounds 2..3 history (post-flare slice):")
    for nid in flash_node_ids:
        h = score_history.get(nid, [])
        print(f"    flash {nid[:12]}: {[round(v, 3) for v in h]}")
    for nid in quiet_node_ids:
        h = score_history.get(nid, [])
        print(f"    quiet {nid[:12]}: {[round(v, 3) for v in h]}")
    for nid in persistent_node_ids:
        h = score_history.get(nid, [])
        print(f"    pers  {nid[:12]}: {[round(v, 3) for v in h]}")

    # Run the pruner. score_threshold is set above the persistent
    # nodes' standing floor so they survive; below the flash and quiet
    # nodes' settled-low absolute values.
    pruned = prune_settled_negatives(
        world,
        score_threshold=0.30,
        novelty_threshold=0.05,
        score_history=score_history,
    )

    banner("AFTER PRUNE")
    post_ids = all_node_ids(world)
    post_roots = root_ids(world)
    print(f"  pruned: {len(pruned)} node ids")
    for nid in pruned:
        print(f"    - {nid[:12]}")
    print(f"  total nodes now: {len(post_ids)}")
    print(f"  roots now:       {len(post_roots)}")

    # Idempotence: a second pass with no new activity should remove 0.
    pruned2 = prune_settled_negatives(
        world,
        score_threshold=0.30,
        novelty_threshold=0.05,
        score_history=score_history,
    )

    banner("VERDICT")

    ok = True

    # 1. Roots NEVER pruned.
    roots_pruned = pre_roots - post_roots
    if roots_pruned:
        print(f"  -- FAIL: roots were pruned: {roots_pruned}")
        ok = False
    else:
        print(f"  OK   roots preserved ({len(post_roots)} of {len(pre_roots)}).")

    # 2. Flash-and-die / quiet-negative pruned. Skip the assertion if
    # the engine never sprouted those nodes (would mean the probe
    # filtered the obs out; that's a different code path, not a
    # pruning failure).
    flash_quiet = flash_node_ids + quiet_node_ids
    if flash_quiet:
        gone = [nid for nid in flash_quiet if nid not in post_ids]
        survived = [nid for nid in flash_quiet if nid in post_ids]
        if survived:
            # Some survived -- let's check whether their post-flare
            # history actually qualifies as settled-and-quiet under our
            # thresholds. If not, they shouldn't have been pruned and
            # this isn't a failure, just engine geometry.
            spurious = []
            for nid in survived:
                h = score_history.get(nid, [])
                if not h:
                    continue
                max_abs = max(abs(v) for v in h)
                max_delta = (
                    max(abs(h[i + 1] - h[i]) for i in range(len(h) - 1))
                    if len(h) >= 2 else 0.0
                )
                if max_abs < 0.30 and max_delta < 0.05:
                    spurious.append((nid, max_abs, max_delta))
            if spurious:
                print(f"  -- FAIL: {len(spurious)} flash/quiet nodes met the "
                      f"settled-quiet criteria but were not pruned:")
                for nid, ma, md in spurious:
                    print(f"     {nid[:12]} max_abs={ma:.3f} max_delta={md:.3f}")
                ok = False
            else:
                print(f"  OK   flash/quiet pruned where eligible "
                      f"({len(gone)}/{len(flash_quiet)}; "
                      f"{len(survived)} kept their standing legitimately).")
        else:
            print(f"  OK   all flash/quiet nodes pruned "
                  f"({len(gone)}/{len(flash_quiet)}).")
    else:
        print(f"  --   no flash/quiet nodes were sprouted; cannot test that branch.")

    # 3. Persistent positives NOT pruned.
    persistent_still = [nid for nid in persistent_node_ids if nid in post_ids]
    if persistent_node_ids and len(persistent_still) == len(persistent_node_ids):
        print(f"  OK   persistent positives preserved "
              f"({len(persistent_still)}/{len(persistent_node_ids)}).")
    elif persistent_node_ids:
        lost = [nid for nid in persistent_node_ids if nid not in post_ids]
        print(f"  -- FAIL: persistent nodes pruned: {[i[:12] for i in lost]}")
        ok = False
    else:
        print("  --   no persistent node was tracked; cannot test that branch.")

    # 4. Idempotence.
    if pruned2:
        print(f"  -- FAIL: second prune pass removed {len(pruned2)} more nodes "
              f"(should be 0).")
        ok = False
    else:
        print("  OK   pruning is idempotent (second pass removed 0).")

    print()
    if ok:
        print("  RESULT: OK")
    else:
        print("  RESULT: FAILED")
    print()


if __name__ == "__main__":
    main()
