#!/usr/bin/env python3
"""
Lineage queries: recover past states from present-state lineage.

The architecture's claim: present state encodes lineage as a first-class
structural component. Given sufficient resolution, past states are
computable from the present without separate checkpointing.

This test runs an engine through a known sequence of state changes,
captures ground-truth states at several moments, then asks the engine
to reconstruct those past states from its current lineage.

Setup
-----

1. Build a starting state with N tendencies.
2. Apply a sequence of K reseed operations, each emitting allocation-
   shift events into per-tendency lineage rings via the recorder.
3. Capture ground-truth allocations at several engine-time moments
   (T_0 = start, T_mid, T_end).
4. From T_end's state and lineage, reconstruct T_0 and T_mid.
5. Compare reconstructed allocations to ground truth.

What "reconstruct" means
------------------------

The lineage rings hold ALLOCATION_SHIFTED events. Each event records:
  - origin_id: which tendency
  - engine_time: when it happened
  - payload.delta: how much its allocation changed

To reconstruct allocations at engine-time T from current state at T_end:
  - For each tendency, take its current allocation
  - For each event with engine_time > T in its lineage, subtract delta
  - The result is its allocation at engine-time T

This is the engine recovering its own history from present-state
structure, with no checkpoints stored.

Test
----

  - Reconstruction error: for each tendency, |reconstructed - ground_truth|
  - Reconstruction error per moment: T_0, T_mid, T_end (T_end should be
    perfect since we're at it)
  - Cross-tendency error: do all tendencies have small error, or do some
    fail?

If reconstruction error is small (say, < 1% on allocations summing to 1.0)
across all tendencies and all reconstructed moments, the lineage-is-
computable claim holds.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np

from world_model import (
    DefaultTendencyFactory,
    EngineClock,
    EventType,
    Lineage,
    LineageRecorder,
    PresentState,
    StakeWeightGraph,
    Substitution,
    Tendency,
    TendencySpec,
    UnboundedPolicy,
    reseed_and_equilibrate,
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_initial_state(n_tendencies: int = 8, seed: int = 42):
    """Build a starting state with n_tendencies, equal allocations, full graph."""
    factory = DefaultTendencyFactory()
    rng = random.Random(seed)
    specs = [
        TendencySpec(id=f"t_{i}", initial_allocation=1.0/n_tendencies)
        for i in range(n_tendencies)
    ]
    tendencies = factory.build_set(specs)

    # Random graph
    graph = StakeWeightGraph()
    for i in range(n_tendencies):
        for j in range(i+1, n_tendencies):
            w = rng.random() * 0.5 + 0.1
            graph.add_edge(f"t_{i}", f"t_{j}", w)

    # Use UnboundedPolicy so we don't lose events during the test
    lineages = {tid: Lineage(outbox=UnboundedPolicy()) for tid in tendencies.ids()}
    return PresentState(tendencies=tendencies, lineages=lineages, graph=graph)


# ---------------------------------------------------------------------------
# Run a sequence of perturbations
# ---------------------------------------------------------------------------

def run_perturbation_sequence(state: PresentState, recorder: LineageRecorder,
                              n_steps: int = 10, seed: int = 0):
    """Apply n_steps reseed operations, each randomly perturbing one tendency.

    Returns a list of (engine_time_after_step, ground_truth_allocations)
    snapshots taken after each step.

    NOTE: reseed_and_equilibrate emits events through the recorder when
    a recorder is provided, so the lineage captures everything we need.
    """
    rng = random.Random(seed)
    snapshots = []

    # Snapshot at start (before any reseed)
    snapshots.append((
        recorder.clock.now(),
        {t.id: t.allocation for t in state.tendencies.all()},
    ))

    current_state = state
    for step in range(n_steps):
        # Pick a random tendency and shift its allocation by some amount
        tids = current_state.tendencies.ids()
        target_tid = rng.choice(tids)
        current = current_state.tendencies.get(target_tid).allocation
        shift = (rng.random() - 0.5) * 0.4   # in [-0.2, 0.2]
        new_alloc = max(0.01, min(0.99, current + shift))

        # Apply via reseed_and_equilibrate. Recorder gets passed so events fire.
        result = reseed_and_equilibrate(
            current_state,
            substitutions=[Substitution(
                id=target_tid,
                new_tendency=Tendency(id=target_tid, allocation=new_alloc),
            )],
            propagate_via_graph=False,
            learning_rate=0.5,
            tolerance=1e-4,
            max_iterations=100,
            recorder=recorder,
        )
        current_state = result.state
        snapshots.append((
            recorder.clock.now(),
            {t.id: t.allocation for t in current_state.tendencies.all()},
        ))

    return current_state, snapshots


# ---------------------------------------------------------------------------
# The lineage query: reconstruct past state from present + lineage
# ---------------------------------------------------------------------------

def reconstruct_state_at(present_state: PresentState, recorder: LineageRecorder,
                        target_engine_time: int) -> dict[str, float]:
    """Reconstruct allocations at target_engine_time from present state.

    Strategy: start with current allocations. For each tendency, find
    all ALLOCATION_SHIFTED events with engine_time > target_engine_time
    and subtract their deltas from the current allocation. The result
    is the allocation at target_engine_time.
    """
    reconstructed: dict[str, float] = {}
    for tid in present_state.tendencies.ids():
        current_alloc = present_state.tendencies.get(tid).allocation
        lineage = recorder.lineage_of(tid)
        if lineage is None:
            reconstructed[tid] = current_alloc
            continue

        # Walk events; subtract any delta that happened AFTER target_engine_time
        accumulated_undo = 0.0
        for event in lineage.events():
            if (event.type == EventType.ALLOCATION_SHIFTED
                    and event.engine_time > target_engine_time):
                accumulated_undo += event.payload.get("delta", 0.0)

        # Past allocation = current - sum of deltas applied since then
        reconstructed[tid] = current_alloc - accumulated_undo

    return reconstructed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_tendencies: int = 8, n_steps: int = 12, seed: int = 42):
    print()
    print("=" * 70)
    print("LINEAGE QUERY: reconstruct past states from present + lineage")
    print("=" * 70)

    state = build_initial_state(n_tendencies=n_tendencies, seed=seed)
    recorder = LineageRecorder(clock=EngineClock(), graph=state.graph)
    for tid in state.tendencies.ids():
        recorder.register(tid, state.lineages[tid])

    print(f"\n  starting state: {n_tendencies} tendencies, equal allocations")
    print(f"  applying {n_steps} reseed operations...")

    final_state, snapshots = run_perturbation_sequence(
        state, recorder, n_steps=n_steps, seed=seed,
    )

    print(f"\n  final engine_time: {recorder.clock.now()}")
    print(f"  total events emitted: {sum(len(rec.events()) for rec in recorder.lineages.values())}")

    # Check reconstruction at multiple engine-time moments
    print("\n" + "-" * 70)
    print("Reconstruction tests")
    print("-" * 70)

    def total_error(reconstructed: dict, ground_truth: dict) -> float:
        return sum(abs(reconstructed[tid] - ground_truth[tid])
                   for tid in ground_truth)

    def max_error(reconstructed: dict, ground_truth: dict) -> float:
        return max(abs(reconstructed[tid] - ground_truth[tid])
                   for tid in ground_truth)

    print(f"\n  {'moment':<25} {'engine_time':>12} {'total_err':>12} {'max_err':>10}")

    test_indices = [0, len(snapshots) // 4, len(snapshots) // 2,
                   3 * len(snapshots) // 4, len(snapshots) - 1]
    for idx in test_indices:
        et, ground_truth = snapshots[idx]
        reconstructed = reconstruct_state_at(final_state, recorder, et)
        total = total_error(reconstructed, ground_truth)
        maxerr = max_error(reconstructed, ground_truth)
        label = f"snapshot[{idx}/{len(snapshots)-1}]"
        print(f"  {label:<25} {et:>12d} {total:>12.6f} {maxerr:>10.6f}")

    # Show one detailed reconstruction comparison
    print("\n" + "-" * 70)
    print("Detailed reconstruction at first snapshot (T=0)")
    print("-" * 70)
    et_first, gt_first = snapshots[0]
    reconstructed_first = reconstruct_state_at(final_state, recorder, et_first)

    print(f"\n  {'tendency':<10} {'ground_truth':>14} {'reconstructed':>15} {'error':>12}")
    for tid in final_state.tendencies.ids():
        gt = gt_first[tid]
        rec = reconstructed_first[tid]
        err = rec - gt
        print(f"  {tid:<10} {gt:>14.6f} {rec:>15.6f} {err:>+12.6f}")

    # Final verdict
    print()
    print("=" * 70)

    # Use the first snapshot (the most demanding -- most events to undo)
    final_total_err = total_error(
        reconstruct_state_at(final_state, recorder, snapshots[0][0]),
        snapshots[0][1],
    )
    if final_total_err < 0.001:
        print(f"VERDICT: total reconstruction error at T_0 is {final_total_err:.6f}")
        print(f"         (essentially zero -- floating-point noise level)")
        print(f"         The engine reconstructs past states from present + lineage")
        print(f"         exactly. Lineage is computable from present, as claimed.")
    elif final_total_err < 0.05:
        print(f"VERDICT: total reconstruction error at T_0 is {final_total_err:.6f}")
        print(f"         (very small -- well within usable tolerance)")
        print(f"         The architectural claim holds in practice.")
    else:
        print(f"VERDICT: total reconstruction error at T_0 is {final_total_err:.6f}")
        print(f"         (substantial -- the claim does NOT hold cleanly).")
        print(f"         Diagnose: are events being emitted correctly? Is the")
        print(f"         delta accounting capturing all state changes?")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
