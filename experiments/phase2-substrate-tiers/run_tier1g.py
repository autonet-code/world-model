#!/usr/bin/env python3
"""Tier 1γ: single-axis substrate (correctness only).

Reuses the Tier 1A snippet set + LLM cache. Extracts only the
correctness component from each cached sample, drives observations
through a 1-D single-tendency substrate, and checks three
predictions about veto behavior.

No new LLM calls. Run is sub-second.

See TIER1G_SPEC.md for the predictions and rationale.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

sys.path.insert(0, r"C:\code\world-model")

from world_model.generalized import (  # type: ignore
    GeneralizedTendency, Observation, World,
)
from world_model.generalized.tendency import _intrinsic_score_in_tendency  # type: ignore
from world_model.generalized.prune import prune_veto_negatives  # type: ignore

from tier1a_snippets import SNIPPETS, CATEGORY_PAIRS  # type: ignore


HERE = Path(__file__).resolve().parent
LLM_CACHE_PATH = HERE / "tier1a_llm_cache.jsonl"
TIER1A_RESULTS_PATH = HERE / "tier1a_results.json"
RESULTS_PATH = HERE / "tier1g_results.json"
STATUS_PATH = HERE / "tier1g_status.json"


# ---------------------------------------------------------------------------
# Substrate setup -- single tendency, 1-D coords
# ---------------------------------------------------------------------------


BANDWIDTH = 1.5


def build_world() -> World:
    world = World()
    correctness = GeneralizedTendency(
        id="correctness",
        thesis="Code is correct.",
        anchor=(1.0,),
        polarity_axis=(1.0,),
        bandwidth=BANDWIDTH,
        veto_shaped=True,
        veto_score_floor=-0.5,
        novelty_gamma_pro=1.0,
        novelty_gamma_con=1.5,
    )
    world.add_tendency(correctness)
    return world


def round_step(world: World, obs_list: List[Observation]) -> None:
    for obs in obs_list:
        world.add_observation(obs)
    for tendency in world.tendencies.values():
        tendency.act(world)
    world.apply_stakes()
    for tendency in world.tendencies.values():
        tendency.update_novelty(dt=1.0)
    world.clear_observations()


# ---------------------------------------------------------------------------
# LLM cache reuse: extract correctness only
# ---------------------------------------------------------------------------


def load_correctness_medians() -> Dict[str, Tuple[float, int]]:
    """Walk Tier 1A's cache, extract `parsed["correctness"]` from
    each valid sample, return median per snippet plus count of
    valid samples used.
    """
    if not LLM_CACHE_PATH.exists():
        raise FileNotFoundError(f"Tier 1A LLM cache not found at {LLM_CACHE_PATH}")
    by_snippet: Dict[str, List[float]] = {}
    for line in LLM_CACHE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = row.get("id")
        parsed = row.get("parsed")
        if not isinstance(parsed, dict):
            continue
        c = parsed.get("correctness")
        if not isinstance(c, (int, float)):
            continue
        if c < -1.0:
            c = -1.0
        elif c > 1.0:
            c = 1.0
        by_snippet.setdefault(sid, []).append(float(c))
    out: Dict[str, Tuple[float, int]] = {}
    for sid, vals in by_snippet.items():
        if not vals:
            out[sid] = (0.0, 0)
        else:
            out[sid] = (median(vals), len(vals))
    return out


# ---------------------------------------------------------------------------
# Tier 1A correctness-presence (for the comparison block)
# ---------------------------------------------------------------------------


def load_tier1a_presence() -> Dict[str, bool]:
    """Read Tier 1A's saved post-prune presence, return correctness-only
    presence per snippet. If Tier 1A's results file isn't present,
    returns an empty dict and the comparison block is skipped.
    """
    if not TIER1A_RESULTS_PATH.exists():
        return {}
    try:
        data = json.loads(TIER1A_RESULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    pp = data.get("post_prune_presence", {})
    return {sid: bool(d.get("correctness", False)) for sid, d in pp.items()}


# ---------------------------------------------------------------------------
# Per-snippet readout
# ---------------------------------------------------------------------------


def find_snip_node(world: World, snip_id: str):
    correctness = world.tendencies["correctness"]
    for node in correctness.tree.all_nodes():
        if (node.observation_id or "").startswith(f"{snip_id}_"):
            return node
    return None


def write_status(status: dict) -> None:
    status["last_update"] = time.time()
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("Tier 1g: single-axis substrate (correctness only)")
    print(f"  reusing LLM cache: {LLM_CACHE_PATH}")
    print()

    medians = load_correctness_medians()
    print("=== correctness coord per snippet (median of cached samples) ===")
    for s in SNIPPETS:
        if s.id not in medians:
            print(f"  {s.id} [{s.category:>16}]  MISSING (no cached samples)")
        else:
            c, n = medians[s.id]
            print(f"  {s.id} [{s.category:>16}]  c = {c:+.2f}  ({n} samples)")
    print()

    status = {
        "started_at": time.time(),
        "phase": "running",
        "n_snippets": len(SNIPPETS),
        "last_update": time.time(),
    }
    write_status(status)

    # ---- Substrate run ----
    print("=== substrate run ===")
    world = build_world()
    obs_list: List[Observation] = []
    seq = 0
    for s in SNIPPETS:
        if s.id not in medians:
            continue
        c, n = medians[s.id]
        if n == 0:
            continue
        seq += 1
        obs_list.append(Observation(
            id=f"{s.id}_{seq}",
            coords=(c,),
            label=f"{s.id}:{s.category}",
        ))
    round_step(world, obs_list)

    # Pre-prune snapshot
    pre_prune_presence: Dict[str, bool] = {
        s.id: find_snip_node(world, s.id) is not None for s in SNIPPETS
    }

    # Pre-prune intrinsic walk for direct children of correctness root
    correctness = world.tendencies["correctness"]
    print("  direct children of correctness root pre-prune:")
    for ch in correctness.tree.root_node.pro_children:
        intr = _intrinsic_score_in_tendency(ch, "correctness")
        print(f"    PRO  {ch.id[:14]} obs={ch.observation_id} "
              f"t_intr={intr:+.2f} signed={+intr:+.2f}")
    for ch in correctness.tree.root_node.con_children:
        intr = _intrinsic_score_in_tendency(ch, "correctness")
        print(f"    CON  {ch.id[:14]} obs={ch.observation_id} "
              f"t_intr={intr:+.2f} signed={-intr:+.2f}")
    print(f"  veto floor: {correctness.veto_score_floor}")
    print()

    veto_pruned = prune_veto_negatives(world)
    print(f"  prune_veto_negatives removed {len(veto_pruned)} node id(s)")
    print()

    post_prune_presence: Dict[str, bool] = {
        s.id: find_snip_node(world, s.id) is not None for s in SNIPPETS
    }

    # ---- Comparison block (vs Tier 1A) ----
    tier1a_presence = load_tier1a_presence()
    print("=== presence comparison (T = in correctness tree post-prune) ===")
    print(f"  {'snip':>4} {'category':>16} {'c':>5}   {'1g':>3}   {'1A':>3}   delta")
    for s in SNIPPETS:
        if s.id not in medians:
            continue
        c, _ = medians[s.id]
        g_pres = post_prune_presence[s.id]
        a_pres = tier1a_presence.get(s.id)
        g_str = "T" if g_pres else "-"
        a_str = ("T" if a_pres else "-") if a_pres is not None else "?"
        if a_pres is not None and a_pres != g_pres:
            delta = "(1g vetoed but 1A didn't)" if a_pres and not g_pres \
                else "(1A vetoed but 1g didn't)" if g_pres and not a_pres \
                else ""
        else:
            delta = ""
        print(f"  {s.id:>4} {s.category:>16} {c:>+.2f}   {g_str:>3}   {a_str:>3}   {delta}")
    print()

    # ---- Predictions ----
    print("=" * 76)
    print("Tier 1g predictions:")
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

    # G1: gold (S1, S2) survives correctness post-prune.
    g1_ok = all(post_prune_presence[sid] for sid in CATEGORY_PAIRS["gold"])
    check("G1 (gold survives correctness)", g1_ok,
          f"S1, S2 in correctness post-prune: " +
          ", ".join(f"{sid}={post_prune_presence[sid]}"
                    for sid in CATEGORY_PAIRS["gold"]))

    # G2: clearly buggy (S7, S11, S12) vetoed.
    clearly_buggy = ["S7", "S11", "S12"]
    g2_failures = [sid for sid in clearly_buggy if post_prune_presence.get(sid)]
    g2_ok = len(g2_failures) == 0
    check("G2 (clearly buggy vetoed)", g2_ok,
          (f"all of {clearly_buggy} vetoed" if g2_ok
           else f"NOT vetoed: {g2_failures}; "
                f"(coords: " +
                ", ".join(f"{sid}={medians.get(sid, ('?',0))[0]}"
                          for sid in clearly_buggy) + ")"))

    # G3: subtle/quirky/complex/narrow (S3, S4, S5, S6, S8, S9, S10)
    # all present in correctness post-prune.
    survivors = ["S3", "S4", "S5", "S6", "S8", "S9", "S10"]
    g3_failures = [sid for sid in survivors if not post_prune_presence.get(sid)]
    g3_ok = len(g3_failures) == 0
    check("G3 (non-bug snippets survive correctness)", g3_ok,
          (f"all of {survivors} present" if g3_ok
           else f"unexpectedly absent: {g3_failures}"))

    print()
    print(f"  {pass_count}/{pass_count+fail_count} predictions passed")

    # ---- Save ----
    serialisable_medians = {
        sid: {"correctness": float(c), "n_samples": int(n),
              "category": next(s.category for s in SNIPPETS if s.id == sid)}
        for sid, (c, n) in medians.items()
    }
    RESULTS_PATH.write_text(json.dumps({
        "median_correctness": serialisable_medians,
        "pre_prune_presence": pre_prune_presence,
        "post_prune_presence": post_prune_presence,
        "tier1a_correctness_presence": tier1a_presence,
        "veto_pruned": veto_pruned,
        "predictions": predictions,
        "pass_count": pass_count,
        "fail_count": fail_count,
    }, indent=2), encoding="utf-8")
    print(f"\n  results saved to {RESULTS_PATH}")

    status["phase"] = "complete"
    status["pass_count"] = pass_count
    status["fail_count"] = fail_count
    write_status(status)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
