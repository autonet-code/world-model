#!/usr/bin/env python3
"""Tier 0: synthetic three-root substrate test.

Three roots (correctness veto-shaped, simplicity, idiom), six
hand-crafted work-unit observations driven simultaneously every
round so post counts accumulate. Reads out per-W (n, intrinsic_score
in each tree, parent count) and checks falsifiable predictions.

No LLM. Pure substrate dynamics.

See TIER0_SPEC.md for the predictions and rationale.

Schedule rationale: stakes are round-fresh under the post-only
refactor (apply_stakes wipes prior tendency stakes at the start of
each round). To exercise the predictions cleanly, we fire ALL six
work-unit observations every round, so stakes accumulate across all
the work units in parallel. Predictions are then tested at specific
round indices (mid-run snapshots) rather than at end-of-epoch.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, r"C:\code\world-model")

from world_model.generalized import (  # type: ignore
    GeneralizedTendency, Observation, World,
)
from world_model.generalized.tendency import _intrinsic_score  # type: ignore
from world_model.generalized.prune import prune_veto_negatives  # type: ignore
from world_model.models.tree import Position  # type: ignore


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "tier0_results.json"
STATUS_PATH = HERE / "tier0_status.json"
PLOT_PATH = HERE / "tier0_plot.png"


# ---------------------------------------------------------------------------
# Substrate setup
# ---------------------------------------------------------------------------


# bandwidth*1.5 must exceed the diagonal-anchored work-unit's distance
# to each root. (1,1,1) -> (1,0,0) is sqrt(2) ~ 1.41; (1,-1,-1) ->
# (1,0,0) is sqrt(2); (-1,-1,-1) -> any is sqrt(3) ~ 1.73.
# bandwidth=1.5 gives 2.25 -- comfortably bridges.
BANDWIDTH = 1.5


def build_world() -> World:
    world = World()
    correctness = GeneralizedTendency(
        id="correctness",
        thesis="Code is correct.",
        anchor=(1.0, 0.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        bandwidth=BANDWIDTH,
        veto_shaped=True,
        veto_score_floor=-0.5,
        novelty_gamma_pro=1.0,
        novelty_gamma_con=1.5,
    )
    simplicity = GeneralizedTendency(
        id="simplicity",
        thesis="Code is simple.",
        anchor=(0.0, 1.0, 0.0),
        polarity_axis=(0.0, 1.0, 0.0),
        bandwidth=BANDWIDTH,
    )
    idiom = GeneralizedTendency(
        id="idiom",
        thesis="Code is idiomatic.",
        anchor=(0.0, 0.0, 1.0),
        polarity_axis=(0.0, 0.0, 1.0),
        bandwidth=BANDWIDTH,
    )
    world.add_tendency(correctness)
    world.add_tendency(simplicity)
    world.add_tendency(idiom)
    return world


# ---------------------------------------------------------------------------
# Observation set
# ---------------------------------------------------------------------------


WORK_UNITS: Dict[str, Tuple[float, float, float]] = {
    "W1": (+1.0, +1.0, +1.0),   # clean, correct, idiomatic
    "W2": (+1.0, +1.0, -1.0),   # correct, simple, but quirky
    "W3": (+1.0, -1.0, +1.0),   # correct, idiomatic, complex
    "W4": (-1.0, +1.0, +1.0),   # buggy but otherwise nice -> veto target
    "W5": (+1.0,  0.0,  0.0),   # narrow correctness post
    "W6": (-1.0, -1.0, -1.0),   # objectively bad on all axes
}

WORK_UNIT_LABELS: Dict[str, str] = {
    "W1": "clean_correct_idiomatic",
    "W2": "correct_simple_quirky",
    "W3": "correct_idiomatic_complex",
    "W4": "buggy_but_nice",
    "W5": "narrow_correctness",
    "W6": "bad_on_all_axes",
}

N_ROUNDS = 15


def make_obs(work_unit_id: str, seq: int) -> Observation:
    coords = WORK_UNITS[work_unit_id]
    return Observation(
        id=f"{work_unit_id}_{seq}",
        coords=coords,
        label=f"{work_unit_id}:{WORK_UNIT_LABELS[work_unit_id]}_{seq}",
    )


# ---------------------------------------------------------------------------
# Per-round step
# ---------------------------------------------------------------------------


def round_step(world: World, obs_list: List[Observation]) -> None:
    """One round: add observations, run 1 act+apply+update_novelty."""
    for obs in obs_list:
        world.add_observation(obs)
    for tendency in world.tendencies.values():
        tendency.act(world)
    world.apply_stakes()
    for tendency in world.tendencies.values():
        tendency.update_novelty(dt=1.0)
    world.clear_observations()


# ---------------------------------------------------------------------------
# Per-W readout
# ---------------------------------------------------------------------------


def find_w_node_in(world: World, w_id: str, tendency_id: str):
    """Find the substrate node in `tendency_id`'s tree whose obs
    label starts with this W id. Returns None if not present.
    """
    t = world.tendencies.get(tendency_id)
    if t is None:
        return None
    for node in t.tree.all_nodes():
        obs_id = node.observation_id or ""
        if obs_id.startswith(f"{w_id}_"):
            return node
    return None


def find_w_node_anywhere(world: World, w_id: str):
    for tid in world.tendencies:
        node = find_w_node_in(world, w_id, tid)
        if node is not None:
            return node
    return None


def w_snapshot(world: World, w_id: str) -> Dict[str, object]:
    """Per-W readout: id, n, parents tendencies, per-tree
    intrinsic_score, post count.
    """
    node = find_w_node_anywhere(world, w_id)
    if node is None:
        return {
            "w": w_id,
            "id": None,
            "n": None,
            "n_parents": 0,
            "parent_tendencies": [],
            "intrinsic_global": 0.0,
            "post_count": 0,
        }
    parent_tids = sorted({p.tendency_id for p in node.parents})
    return {
        "w": w_id,
        "id": node.id,
        "n": float(node.n),
        "n_parents": len(node.parents),
        "parent_tendencies": parent_tids,
        "intrinsic_global": _intrinsic_score(node),
        "post_count": len(node.stakes),
    }


def root_intrinsic_score_under(world: World, tendency_id: str, w_id: str) -> float:
    """Return intrinsic_score(W's node) as it appears under `tendency_id`'s
    tree. If the node isn't reachable from that tendency's tree at
    all, returns 0.
    """
    node = find_w_node_in(world, w_id, tendency_id)
    if node is None:
        return 0.0
    return _intrinsic_score(node)


# ---------------------------------------------------------------------------
# Status writer
# ---------------------------------------------------------------------------


def write_status(status: dict) -> None:
    status["last_update"] = time.time()
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    world = build_world()
    print(f"built world with tendencies: {sorted(world.tendencies.keys())}")
    print(f"  bandwidth: {BANDWIDTH}, edge-discovery threshold: {BANDWIDTH*1.5:.3f}")
    print(f"  correctness: veto_shaped, gamma_con=1.5, "
          f"floor={world.tendencies['correctness'].veto_score_floor}")
    print(f"  schedule: ALL six work units fire every round, "
          f"for {N_ROUNDS} rounds")
    print()

    status = {
        "started_at": time.time(),
        "phase": "running",
        "n_rounds": N_ROUNDS,
        "current_round": 0,
        "last_update": time.time(),
    }
    write_status(status)

    trace: List[Dict] = []

    initial = {"round": 0}
    for w in WORK_UNITS:
        initial[w] = w_snapshot(world, w)
    trace.append(initial)

    obs_seq = 0
    print(f"{'rd':>3}  {'n_W1':>6} {'n_W2':>6} {'n_W3':>6} {'n_W4':>6} "
          f"{'n_W5':>6} {'n_W6':>6}  "
          f"{'pa1':>3} {'pa4':>3} {'pa6':>3}  "
          f"{'is_W1':>7} {'is_W4':>7} {'is_W6':>7}")

    for round_idx in range(1, N_ROUNDS + 1):
        obs_list: List[Observation] = []
        for w in WORK_UNITS:
            obs_seq += 1
            obs_list.append(make_obs(w, obs_seq))

        round_step(world, obs_list)

        snap = {"round": round_idx}
        for w in WORK_UNITS:
            snap[w] = w_snapshot(world, w)
        trace.append(snap)

        n_str = " ".join(
            f"{snap[w]['n']:>6.3f}" if snap[w]["n"] is not None else "   nan"
            for w in ("W1", "W2", "W3", "W4", "W5", "W6")
        )
        pa1 = snap["W1"]["n_parents"]
        pa4 = snap["W4"]["n_parents"]
        pa6 = snap["W6"]["n_parents"]
        is_w1 = snap["W1"]["intrinsic_global"]
        is_w4 = snap["W4"]["intrinsic_global"]
        is_w6 = snap["W6"]["intrinsic_global"]
        print(f"{round_idx:>3}  {n_str}  "
              f"{pa1:>3} {pa4:>3} {pa6:>3}  "
              f"{is_w1:>+7.2f} {is_w4:>+7.2f} {is_w6:>+7.2f}")

        status["current_round"] = round_idx
        write_status(status)

    # ---- Snapshot the per-tendency intrinsic_score before pruning ----
    print()
    print("per-W intrinsic_score under each tendency at round %d:" % N_ROUNDS)
    print(f"  {'W':>3}  {'correctness':>12} {'simplicity':>12} {'idiom':>12}")
    pre_prune_intrinsic: Dict[str, Dict[str, float]] = {}
    for w in WORK_UNITS:
        c = root_intrinsic_score_under(world, "correctness", w)
        s = root_intrinsic_score_under(world, "simplicity", w)
        i = root_intrinsic_score_under(world, "idiom", w)
        pre_prune_intrinsic[w] = {"correctness": c, "simplicity": s, "idiom": i}
        print(f"  {w:>3}  {c:>+12.2f} {s:>+12.2f} {i:>+12.2f}")

    # ---- Snapshot direct children of correctness root and their intrinsic ----
    print()
    print("direct children of correctness root + their intrinsic_score:")
    correctness = world.tendencies["correctness"]
    for ch in correctness.tree.root_node.all_children:
        # Position from the parent's perspective
        in_pro = ch in correctness.tree.root_node.pro_children
        sign = "PRO" if in_pro else "CON"
        intr = _intrinsic_score(ch)
        print(f"  [{sign}] {ch.id[:14]} "
              f"obs={ch.observation_id} "
              f"intrinsic={intr:+.2f}")

    # ---- Veto prune ----
    print()
    print("running prune_veto_negatives...")
    veto_pruned = prune_veto_negatives(world)
    print(f"  pruned {len(veto_pruned)} node id(s) under veto roots: "
          f"{[p[:14] for p in veto_pruned]}")

    # ---- Post-prune snapshot ----
    post_prune = {"round": "post_prune"}
    for w in WORK_UNITS:
        post_prune[w] = w_snapshot(world, w)
    trace.append(post_prune)

    # Tendency-aware post-prune presence
    post_prune_presence = {}
    for w in WORK_UNITS:
        post_prune_presence[w] = {
            tid: find_w_node_in(world, w, tid) is not None
            for tid in world.tendencies
        }
    print()
    print("post-prune presence (T = node is in this tendency's tree):")
    print(f"  {'W':>3}  {'correctness':>12} {'simplicity':>12} {'idiom':>12}")
    for w in WORK_UNITS:
        row = post_prune_presence[w]
        print(f"  {w:>3}  "
              f"{('T' if row['correctness'] else '-'):>12} "
              f"{('T' if row['simplicity'] else '-'):>12} "
              f"{('T' if row['idiom'] else '-'):>12}")

    # ---- Predictions ----
    print()
    print("=" * 76)
    print("Tier 0 predictions:")
    print("=" * 76)

    pass_count = 0
    fail_count = 0
    predictions: List[dict] = []

    def check(label: str, condition: bool, detail: str) -> None:
        nonlocal pass_count, fail_count
        ok = "PASS" if condition else "FAIL"
        if condition:
            pass_count += 1
        else:
            fail_count += 1
        print(f"  [{ok}] {label}: {detail}")
        predictions.append({"label": label, "condition": bool(condition), "detail": detail})

    # P1: W1 co-parents in all three trees by round 5.
    w1_at_5 = trace[5]["W1"]
    p1_ok = set(w1_at_5["parent_tendencies"]) == {"correctness", "simplicity", "idiom"}
    check("P1 (W1 co-parents 3 trees)", p1_ok,
          f"W1 parent tendencies at round 5 = "
          f"{w1_at_5['parent_tendencies']} (expected all 3)")

    # P2: W1 n decays under PRO observation; check at round 5 when W1
    # has been firing PRO every round.
    w1_n5 = w1_at_5["n"] or 1.0
    check("P2 (W1 n decays under PRO)", w1_n5 < 0.3,
          f"W1.n at round 5 = {w1_n5:.4f} (expected < 0.3)")

    # P3: W4 sits as CON child of correctness (its coords[0] = -1).
    # Its signed contribution to correctness = -intrinsic_score < 0.
    # The veto-prune removes it from correctness's tree.
    correctness_root = world.tendencies.get("correctness")
    # We have to look at PRE-prune state to verify position; rebuild
    # by reading the trace's last-round info plus the now-known prune
    # result.
    w4_in_correctness_post = post_prune_presence["W4"]["correctness"]
    # W4's coords[0] = -1 -> we expect it to be CON of correctness.
    # If it ended up as PRO (meaning the substrate misclassified it),
    # the prune wouldn't fire and the test should also surface that.
    p3_ok = not w4_in_correctness_post
    check("P3 (correctness vetoes W4)", p3_ok,
          f"W4 (CON of correctness, coords={WORK_UNITS['W4']}) "
          f"in correctness post-prune = {w4_in_correctness_post} "
          f"(expected False -- veto should remove it)")

    # P4: W3 = (+1,-1,+1) -- negative on simplicity. Without veto on
    # simplicity, W3 stays in simplicity's tree post veto-prune.
    w3_in_simplicity_post = post_prune_presence["W3"]["simplicity"]
    check("P4 (simplicity does not auto-prune W3)", w3_in_simplicity_post,
          f"W3 still in simplicity tree post veto-prune = "
          f"{w3_in_simplicity_post} (expected True)")

    # P5 (replacing the redundant one): W4 should still be present in
    # SOMEWHERE post-prune (e.g. simplicity or idiom), even though
    # correctness vetoed it. The veto removes the work item from
    # correctness's tree without erasing it from the world.
    w4_anywhere_post = any(post_prune_presence["W4"].values())
    check("P5 (W4 survives outside correctness)", w4_anywhere_post,
          f"W4 still in some non-correctness tree post-prune = "
          f"{[t for t,v in post_prune_presence['W4'].items() if v]}")

    # P6: W6 = (-1,-1,-1) sits as CON in all three trees (coords are
    # all negative). Veto-prune removes it from correctness. Other
    # roots (non-veto) keep it -- the work item is still recorded
    # as anti-simplicity / anti-idiom via CON-position, which is
    # informative metadata for downstream readers.
    w6_in_correctness_post = post_prune_presence["W6"]["correctness"]
    p6_ok = not w6_in_correctness_post
    check("P6 (W6 vetoed from correctness)", p6_ok,
          f"W6 (CON of all three roots, coords={WORK_UNITS['W6']}) "
          f"in correctness post-prune = {w6_in_correctness_post} "
          f"(expected False -- veto should remove it)")

    print()
    print(f"  {pass_count}/{pass_count+fail_count} predictions passed")

    # ---- Save ----
    RESULTS_PATH.write_text(json.dumps({
        "trace": trace,
        "predictions": predictions,
        "pre_prune_intrinsic": pre_prune_intrinsic,
        "post_prune_presence": post_prune_presence,
        "veto_pruned": veto_pruned,
        "pass_count": pass_count,
        "fail_count": fail_count,
    }, indent=2), encoding="utf-8")
    print(f"\n  trace + predictions saved to {RESULTS_PATH}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rounds = list(range(N_ROUNDS + 1))
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for w in WORK_UNITS:
            n_clean = [t[w]["n"] if t[w]["n"] is not None else float("nan")
                       for t in trace if isinstance(t["round"], int)]
            axes[0].plot(rounds, n_clean, "o-", label=w, markersize=3)
            pa = [t[w]["n_parents"] for t in trace if isinstance(t["round"], int)]
            axes[1].plot(rounds, pa, "o-", label=w, markersize=3)
        axes[0].set_ylabel("persistent novelty n")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", ncol=3, fontsize=8)
        axes[0].set_title("Tier 0: per-W n trajectories (all six fire every round)")
        axes[1].set_ylabel("# parent edges")
        axes[1].set_xlabel("round")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right", ncol=3, fontsize=8)
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
