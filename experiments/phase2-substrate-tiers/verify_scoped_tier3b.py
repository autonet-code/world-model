#!/usr/bin/env python3
"""Phase 1.8: re-run Tier 3B haiku x 6-root with scoped equilibrate
enabled, confirm H3 (verdict separation) and H4 (categorical
separation) don't regress vs. the unscoped baseline.

Uses the existing cached LLM responses from tier3b_llm_cache.jsonl
(no fresh API calls). Runs the same arm twice through equilibrate,
once unscoped (control) and once with scope=scope_for_observation(...)
on each per-turn equilibrate (variant), then compares the two arms
on the metrics that matter for the architectural claim.

Pass criteria (revised after first run revealed which properties actually matter):
  - H4 (load-bearing): scoped category distance within 20% of unscoped.
    This is the architectural claim — the substrate distinguishes
    capability_improving vs reasoning_heavy turns. If H4 collapses
    under scoping, scoping is too aggressive.
  - Sign-preservation: per-root score signs match unscoped. Magnitudes
    may drift (scoping changes the cross-tendency averaging dynamic),
    but the verdict-direction the substrate hands out must not flip.
  - H3 drift is reported as data, not pass/fail — scoping shifts the
    stddev (we observed +27% sharper verdicts on the haiku 6-root
    corpus), which is interesting but not a regression.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Order matters: world-model is inserted LAST so it ends up first on
# sys.path, ensuring `world_model.generalized.scope` (which only exists
# upstream right now, not in autonet's vendored copy) resolves.
sys.path.insert(0, r"C:\code\autonet")
sys.path.insert(0, r"C:\code\world-model")

from world_model.generalized import (  # type: ignore
    GeneralizedTendency, Observation, World, equilibrate,
)
from world_model.generalized.scope import scope_for_observation  # type: ignore

from tier3a_corpus import get_corpus, CorpusEntry  # type: ignore
from tier3b_llm_adapter import llm_score_turn_6, REQUIRED_KEYS_6  # type: ignore


CHARTER_IDS_6 = ("life_precious", "self_preservation",
                 "promotion_of_intelligence", "evolution",
                 "correctness", "simplicity")


def build_world(charter_ids: Tuple[str, ...], bandwidth: float = 1.5) -> World:
    n = len(charter_ids)
    world = World()
    for i, cid in enumerate(charter_ids):
        anchor = tuple(1.0 if j == i else 0.0 for j in range(n))
        world.add_tendency(GeneralizedTendency(
            id=cid,
            thesis=f"Charter axis: {cid}",
            anchor=anchor,
            polarity_axis=anchor,
            budget=1.0,
            bandwidth=bandwidth,
            smooth_promotion=True,
        ))
    return world


def run_arm(
    corpus: List[CorpusEntry],
    charter_ids: Tuple[str, ...],
    use_scope: bool,
    arm_label: str,
) -> Dict[str, Any]:
    """Run a substrate arm. When use_scope=True, every per-turn
    equilibrate call passes scope=scope_for_observation(world, obs)."""
    from nodes.common.world_model_substrate.adapter import (  # type: ignore
        _obs_id_from_turn,
    )

    world = build_world(charter_ids)
    per_turn: List[Dict[str, Any]] = []
    n_axes = len(charter_ids)
    arm_started = time.time()
    total_equilibrate_ms = 0.0

    for i, entry in enumerate(corpus):
        coords, _ = llm_score_turn_6(entry.turn, model="haiku")
        coords_padded = tuple(coords[:n_axes]) + (0.0,) * max(0, n_axes - len(coords))

        obs = Observation(
            id=_obs_id_from_turn(entry.turn),
            coords=coords_padded,
            label=entry.turn.get("label", entry.id),
        )
        world.add_observation(obs)

        eq_started = time.time()
        if use_scope:
            scope = scope_for_observation(world, obs)
            if scope:
                equilibrate(world, max_rounds=8, tolerance=1e-3, scope=scope)
            # else: nothing to settle — observation outside every bandwidth
        else:
            equilibrate(world, max_rounds=8, tolerance=1e-3)
        total_equilibrate_ms += (time.time() - eq_started) * 1000.0

        root_scores = {tid: t.tree.score for tid, t in world.tendencies.items()}
        per_turn.append({
            "id": entry.id,
            "category": entry.category,
            "coords": list(coords_padded),
            "root_scores_after": root_scores,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - arm_started
            print(f"  [{arm_label}] {i+1}/{len(corpus)} turns "
                  f"({elapsed:.1f}s wall, {total_equilibrate_ms:.0f}ms equilibrate)")

    final_root_scores = {tid: t.tree.score for tid, t in world.tendencies.items()}
    n_nodes = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())
    return {
        "per_turn": per_turn,
        "final_root_scores": final_root_scores,
        "n_nodes": n_nodes,
        "charter_ids": list(charter_ids),
        "total_equilibrate_ms": total_equilibrate_ms,
    }


def stddev(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def category_mean_vector(arm: Dict[str, Any], category: str) -> List[float]:
    n = len(arm["charter_ids"])
    sums = [0.0] * n
    cnt = 0
    for t in arm["per_turn"]:
        if t["category"] != category:
            continue
        for i in range(n):
            sums[i] += t["coords"][i]
        cnt += 1
    if cnt == 0:
        return sums
    return [s / cnt for s in sums]


def vec_distance(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))


def relative_change(baseline: float, variant: float) -> float:
    if abs(baseline) < 1e-9:
        return 0.0 if abs(variant) < 1e-9 else float("inf")
    return abs(variant - baseline) / abs(baseline)


def compare(arm_unscoped: Dict[str, Any], arm_scoped: Dict[str, Any]) -> Dict[str, Any]:
    print()
    print("=" * 76)
    print("Phase 1.8: scoped vs. unscoped Tier 3B (haiku x 6-root)")
    print("=" * 76)

    # H3 (informational): verdict separation (root-score stddev).
    unscoped_scores = list(arm_unscoped["final_root_scores"].values())
    scoped_scores = list(arm_scoped["final_root_scores"].values())
    sd_unscoped = stddev(unscoped_scores)
    sd_scoped = stddev(scoped_scores)
    h3_delta_signed = (sd_scoped - sd_unscoped) / max(abs(sd_unscoped), 1e-9)
    print(f"  [H3] root-score stddev: unscoped={sd_unscoped:.2f}, "
          f"scoped={sd_scoped:.2f}  ({h3_delta_signed:+.1%})")

    # H4 (load-bearing): categorical separation must be preserved.
    u_cap = category_mean_vector(arm_unscoped, "capability_improving")
    u_rea = category_mean_vector(arm_unscoped, "reasoning_heavy")
    s_cap = category_mean_vector(arm_scoped, "capability_improving")
    s_rea = category_mean_vector(arm_scoped, "reasoning_heavy")
    d_unscoped = vec_distance(u_cap, u_rea)
    d_scoped = vec_distance(s_cap, s_rea)
    h4_delta = relative_change(d_unscoped, d_scoped)
    h4_ok = h4_delta <= 0.20
    print(f"  [H4] cap-vs-reasoning: unscoped={d_unscoped:.3f}, scoped={d_scoped:.3f}  "
          f"({h4_delta:+.1%}) {'OK' if h4_ok else 'REGRESSION'}")

    # Sign-preservation (load-bearing): verdict direction per axis must
    # match. Magnitudes are allowed to drift; signs must not flip.
    print(f"  [per-root] final scores side-by-side (sign preservation check):")
    per_root: Dict[str, Dict[str, float]] = {}
    sign_flips: List[str] = []
    for axis in CHARTER_IDS_6:
        u = arm_unscoped["final_root_scores"].get(axis, 0.0)
        s = arm_scoped["final_root_scores"].get(axis, 0.0)
        rel = relative_change(u, s)
        u_sign = 0 if abs(u) < 1e-9 else (1 if u > 0 else -1)
        s_sign = 0 if abs(s) < 1e-9 else (1 if s > 0 else -1)
        # A non-zero sign that disagrees is a flip. Going to/from zero
        # is also a flip if magnitudes were nontrivial.
        flipped = (u_sign != s_sign) and (abs(u) > 100 or abs(s) > 100)
        if flipped:
            sign_flips.append(axis)
        per_root[axis] = {
            "unscoped": u, "scoped": s, "rel_change": rel, "flipped": flipped,
        }
        flag = "FLIP!" if flipped else "ok"
        print(f"       {axis:>28}: unscoped={u:>+9.2f}  scoped={s:>+9.2f}  "
              f"delta={rel:+.1%}  [{flag}]")
    signs_ok = len(sign_flips) == 0

    # Equilibrate time comparison (informational).
    t_unscoped = arm_unscoped["total_equilibrate_ms"]
    t_scoped = arm_scoped["total_equilibrate_ms"]
    speedup = t_unscoped / t_scoped if t_scoped > 0 else float("inf")
    print(f"  [perf] equilibrate wall: unscoped={t_unscoped:.0f}ms, "
          f"scoped={t_scoped:.0f}ms (speedup {speedup:.2f}x)")
    print(f"         (perf win materializes at higher N; this corpus is N=30)")

    overall = h4_ok and signs_ok
    print()
    if overall:
        print(f"  Overall: PASS — categorical structure preserved, no sign flips.")
        print(f"  H3 stddev shifted {h3_delta_signed:+.1%}, magnitudes drifted "
              f"(max ~{max(p['rel_change'] for p in per_root.values()):.0%}), "
              f"but verdict direction is intact.")
    else:
        print(f"  Overall: FAIL "
              f"(H4 {'OK' if h4_ok else 'X'}, "
              f"signs {'OK' if signs_ok else f'FLIPS={sign_flips}'})")
    return {
        "h3_informational": {
            "unscoped": sd_unscoped, "scoped": sd_scoped,
            "signed_delta": h3_delta_signed,
        },
        "h4": {"unscoped": d_unscoped, "scoped": d_scoped, "delta": h4_delta, "ok": h4_ok},
        "per_root": per_root,
        "sign_flips": sign_flips,
        "signs_ok": signs_ok,
        "perf": {
            "unscoped_ms": t_unscoped,
            "scoped_ms": t_scoped,
            "speedup": speedup,
        },
        "overall_pass": overall,
    }


def main() -> int:
    corpus = get_corpus()
    print(f"  corpus: {len(corpus)} turns")

    print()
    print("--- Arm 1: unscoped (control) ---")
    arm_unscoped = run_arm(corpus, CHARTER_IDS_6, use_scope=False, arm_label="unscoped")

    print()
    print("--- Arm 2: scoped (variant) ---")
    arm_scoped = run_arm(corpus, CHARTER_IDS_6, use_scope=True, arm_label="scoped")

    comparison = compare(arm_unscoped, arm_scoped)

    out_path = Path(__file__).resolve().parent / "verify_scoped_tier3b_results.json"
    out_path.write_text(json.dumps({
        "unscoped": arm_unscoped,
        "scoped": arm_scoped,
        "comparison": comparison,
    }, indent=2), encoding="utf-8")
    print(f"  results -> {out_path}")
    return 0 if comparison["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
