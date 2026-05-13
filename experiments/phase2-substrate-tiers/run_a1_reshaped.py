#!/usr/bin/env python3
"""Reshaped A1: substrate n-tracking test.

Drives a four-region single-tendency substrate through a designed
observation schedule and verifies that per-region n values match
the predicted confidence patterns. No LLM, pure substrate dynamics.

See A1_RESHAPED_SPEC.md for the predictions and rationale.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, r"C:\code\world-model")
sys.path.insert(0, r"C:\code\autonet")

from world_model.generalized import (  # type: ignore
    GeneralizedTendency, Observation, World,
)
from world_model.models.tree import Position  # type: ignore


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "a1_reshaped_results.json"
STATUS_PATH = HERE / "a1_reshaped_status.json"
PLOT_PATH = HERE / "a1_reshaped_plot.png"


# ---------------------------------------------------------------------------
# Build a 4-region single-tendency substrate
# ---------------------------------------------------------------------------


REGION_ANCHORS = {
    "A": (1.0, 0.0, 0.0, 0.0),
    "B": (0.0, 1.0, 0.0, 0.0),
    "C": (0.0, 0.0, 1.0, 0.0),
    "D": (0.0, 0.0, 0.0, 1.0),
}


def build_world() -> tuple[World, Dict[str, str]]:
    """Build a world with one tendency 'r' and four pre-sprouted PRO
    sub-claims, one per region. Returns (world, region->node_id map).
    """
    world = World()
    tendency = GeneralizedTendency(
        id="r",
        thesis="The thing.",
        anchor=(0.5, 0.5, 0.5, 0.5),  # center
        polarity_axis=(1.0, 1.0, 1.0, 1.0),
        budget=1.0,
        bandwidth=0.7,
        smooth_promotion=True,
    )
    world.add_tendency(tendency)

    region_to_node: Dict[str, str] = {}
    root_id = tendency.tree.root_node.id
    for label, anchor in REGION_ANCHORS.items():
        new_node = tendency.sprout_child(
            parent_node_id=root_id,
            position=Position.PRO,
            anchor=anchor,
            polarity_axis=anchor,
            content=f"region_{label}",
        )
        # Each fresh node starts at n=1.0 (the dataclass default).
        region_to_node[label] = new_node.id
    return world, region_to_node


def make_obs(region: str, kind: str, seq: int) -> Observation:
    """Observation aimed at a specific region with PRO or CON polarity."""
    base = list(REGION_ANCHORS[region])
    sign = 1.0 if kind == "PRO" else -1.0
    coords = tuple(sign * x for x in base)
    return Observation(
        id=f"{region}_{kind.lower()}_{seq}",
        coords=coords,
        label=f"{region}_{kind}_{seq}",
    )


# ---------------------------------------------------------------------------
# Per-round step (manual, granular)
# ---------------------------------------------------------------------------


def round_step(
    world: World,
    obs_list: List[Observation],
) -> None:
    """Run one round on the world: add the given observations, run
    one act+apply_stakes+update_novelty pass, then clear observations.

    Each call advances every per-node n by exactly one update_novelty
    step regardless of how many obs landed (or zero — quiet rounds
    still drift).
    """
    for obs in obs_list:
        world.add_observation(obs)
    for tendency in world.tendencies.values():
        tendency.act(world)
    world.apply_stakes()
    for tendency in world.tendencies.values():
        tendency.update_novelty(dt=1.0)
    world.clear_observations()


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def schedule_for_round(round_idx: int) -> List[tuple[str, str]]:
    """Return the (region, kind) pairs to fire in this round."""
    obs: List[tuple[str, str]] = []
    # Region A: PRO rounds 1-10, CON rounds 11-20
    if 1 <= round_idx <= 10:
        obs.append(("A", "PRO"))
    elif 11 <= round_idx <= 20:
        obs.append(("A", "CON"))
    # Region B: PRO rounds 1-10
    if 1 <= round_idx <= 10:
        obs.append(("B", "PRO"))
    # Region C: PRO rounds 1-5 only
    if 1 <= round_idx <= 5:
        obs.append(("C", "PRO"))
    # Region D: CON rounds 1-3 only
    if 1 <= round_idx <= 3:
        obs.append(("D", "CON"))
    return obs


# ---------------------------------------------------------------------------
# Status writer
# ---------------------------------------------------------------------------


def write_status(status: dict) -> None:
    status["last_update"] = time.time()
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def main() -> int:
    world, region_to_node = build_world()
    print(f"Built world with regions: {sorted(region_to_node.keys())}")
    print(f"  region -> node id mapping: {region_to_node}")
    print()

    n_rounds = 30
    status = {
        "started_at": time.time(),
        "phase": "running",
        "n_rounds": n_rounds,
        "current_round": 0,
        "last_update": time.time(),
    }
    write_status(status)

    # Trace: per round, per region, (n, score) of the region's leaf node.
    trace: List[Dict] = []

    # Initial state (round 0)
    snapshot = {"round": 0, "schedule": []}
    for region, node_id in region_to_node.items():
        node = world.tendencies["r"].tree.get_node(node_id)
        snapshot[f"n_{region}"] = node.n
        snapshot[f"score_{region}"] = node.net_score
    trace.append(snapshot)

    obs_seq = 0
    print(f"{'rd':>3} {'sched':<22} "
          f"{'n_A':>7} {'n_B':>7} {'n_C':>7} {'n_D':>7}  "
          f"{'sA':>6} {'sB':>6} {'sC':>6} {'sD':>6}")

    for round_idx in range(1, n_rounds + 1):
        sched = schedule_for_round(round_idx)
        obs_list: List[Observation] = []
        for region, kind in sched:
            obs_seq += 1
            obs_list.append(make_obs(region, kind, obs_seq))

        round_step(world, obs_list)

        snapshot = {
            "round": round_idx,
            "schedule": [{"region": r, "kind": k} for r, k in sched],
        }
        for region, node_id in region_to_node.items():
            node = world.tendencies["r"].tree.get_node(node_id)
            snapshot[f"n_{region}"] = node.n
            snapshot[f"score_{region}"] = node.net_score
        trace.append(snapshot)

        sched_str = ",".join(f"{r}{k[0]}" for r, k in sched) or "(quiet)"
        print(f"{round_idx:>3} {sched_str:<22} "
              f"{snapshot['n_A']:>7.4f} {snapshot['n_B']:>7.4f} "
              f"{snapshot['n_C']:>7.4f} {snapshot['n_D']:>7.4f}  "
              f"{snapshot['score_A']:>+6.3f} {snapshot['score_B']:>+6.3f} "
              f"{snapshot['score_C']:>+6.3f} {snapshot['score_D']:>+6.3f}")

        status["current_round"] = round_idx
        write_status(status)

    # ---- Predictions ----
    print()
    print("=" * 76)
    print("Predictions (see A1_RESHAPED_SPEC.md):")
    print("=" * 76)

    pass_count = 0
    fail_count = 0
    predictions = []

    def check(label: str, condition: bool, detail: str) -> None:
        nonlocal pass_count, fail_count
        ok = "PASS" if condition else "FAIL"
        if condition:
            pass_count += 1
        else:
            fail_count += 1
        print(f"  [{ok}] {label}: {detail}")
        predictions.append({"label": label, "condition": condition, "detail": detail})

    n_A_r10 = trace[10]["n_A"]
    n_B_r10 = trace[10]["n_B"]
    check("P1", n_A_r10 < 0.6,
          f"n_A after round 10 = {n_A_r10:.4f} (expected < 0.6)")
    check("P2", n_B_r10 < 0.6 and abs(n_A_r10 - n_B_r10) < 0.1,
          f"n_B={n_B_r10:.4f}, |n_A-n_B|={abs(n_A_r10-n_B_r10):.4f} "
          f"(expected n_B<0.6 and |n_A-n_B|<0.1)")

    n_A_r20 = trace[20]["n_A"]
    n_B_r20 = trace[20]["n_B"]
    check("P3", n_A_r20 - n_B_r20 > 0.1,
          f"n_A-n_B after round 20 = {n_A_r20-n_B_r20:+.4f} "
          f"(expected > 0.1; A re-surprised, B settled)")

    n_C_r5 = trace[5]["n_C"]
    n_C_r30 = trace[30]["n_C"]
    check("P4", n_C_r30 > n_C_r5,
          f"n_C(round 5)={n_C_r5:.4f}, n_C(round 30)={n_C_r30:.4f} "
          f"(expected drift back up after settling)")

    n_D_min = min(t["n_D"] for t in trace)
    check("P5", n_D_min > 0.7,
          f"n_D min across all rounds = {n_D_min:.4f} (expected > 0.7)")

    # P6: A's trajectory is non-monotonic across rounds 1-20.
    a_values = [t["n_A"] for t in trace[:21]]
    a_min_idx = a_values.index(min(a_values))
    nonmonotonic = (a_min_idx > 0 and a_min_idx < len(a_values) - 1
                    and a_values[a_min_idx] < a_values[0]
                    and a_values[a_min_idx] < a_values[-1])
    check("P6", nonmonotonic,
          f"n_A trajectory has interior min at round {a_min_idx} "
          f"(value {a_values[a_min_idx]:.4f}); "
          f"start={a_values[0]:.4f}, end={a_values[-1]:.4f} "
          f"(expected resist-then-yield shape)")

    print()
    print(f"  {pass_count}/{pass_count+fail_count} predictions passed")

    # ---- Save results + plot ----
    RESULTS_PATH.write_text(json.dumps({
        "trace": trace,
        "predictions": predictions,
        "pass_count": pass_count,
        "fail_count": fail_count,
    }, indent=2), encoding="utf-8")
    print(f"\n  trace + predictions saved to {RESULTS_PATH}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rounds = [t["round"] for t in trace]
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for region in ["A", "B", "C", "D"]:
            axes[0].plot(rounds, [t[f"n_{region}"] for t in trace],
                         "o-", label=f"region {region}", markersize=3)
            axes[1].plot(rounds, [t[f"score_{region}"] for t in trace],
                         "o-", label=f"region {region}", markersize=3)
        axes[0].set_ylabel("persistent novelty n")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_title("A1 reshaped: per-region n trajectories")
        axes[0].axvspan(0.5, 10.5, alpha=0.08, color="green",
                        label="rounds 1-10: PRO on A,B (and 1-5 on C, 1-3 CON on D)")
        axes[0].axvspan(10.5, 20.5, alpha=0.08, color="red",
                        label="rounds 11-20: CON on A")
        axes[1].set_ylabel("net_score")
        axes[1].set_xlabel("round")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=120)
        plt.close()
        print(f"  plot saved to {PLOT_PATH}")
    except ImportError:
        print("  (matplotlib unavailable; skipping plot)")

    status["phase"] = "complete"
    status["pass_count"] = pass_count
    status["fail_count"] = fail_count
    write_status(status)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
