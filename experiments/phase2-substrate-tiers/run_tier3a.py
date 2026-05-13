#!/usr/bin/env python3
"""Tier 3A: validate substrate stack in autonet's pipeline.

Compares two embedders (heuristic score_turn_4d vs LLM-binary-flag)
on the same corpus of turns, runs both arms through autonet's
build_charter_world + equilibrate, evaluates 5 predictions.

Tests whether dropping a small-LLM call into autonet's existing
seam produces a usable upgrade over the keyword heuristic.

Usage:
  python run_tier3a.py qwen [--real N]
  python run_tier3a.py haiku [--real N]
  python run_tier3a.py both [--real N]

For haiku, the claude-max-proxy must be running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, r"C:\code\world-model")
sys.path.insert(0, r"C:\code\autonet")

from world_model.generalized import equilibrate  # type: ignore
from nodes.common.world_model_substrate.adapter import (  # type: ignore
    build_charter_world, turn_to_observation, _EventRecorder, _all_node_ids,
    serialize_world,
)
from nodes.common.world_model_substrate.aggregate import (  # type: ignore
    aggregate_contributions, apply_events,
)
from tier3a_corpus import get_corpus, CorpusEntry  # type: ignore
from tier3a_llm_adapter import (  # type: ignore
    turn_to_observation_via_llm, llm_score_turn,
)


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "tier3a_results.json"
STATUS_PATH = HERE / "tier3a_status.json"
PLOT_PATH = HERE / "tier3a_plot.png"


AXES = ("life_precious", "self_preservation",
        "promotion_of_intelligence", "evolution")


def write_status(status: dict) -> None:
    status["last_update"] = time.time()
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-arm run
# ---------------------------------------------------------------------------


def run_heuristic_arm(corpus: List[CorpusEntry]) -> Dict[str, Any]:
    """Run autonet's heuristic embedder on the corpus, equilibrate
    the world, return per-turn coords + final root scores + events.
    """
    world = build_charter_world()
    recorder = _EventRecorder(agent_id="tier3a-heuristic")
    per_turn: List[Dict[str, Any]] = []
    for i, entry in enumerate(corpus):
        obs = turn_to_observation(entry.turn, turn_index=i)
        before_ids = _all_node_ids(world)
        world.add_observation(obs)
        recorder.observation_added(obs)
        equilibrate(world, max_rounds=8, tolerance=1e-3)
        recorder.sub_claims_after_equilibrate(world, before_ids)
        per_turn.append({
            "id": entry.id,
            "category": entry.category,
            "coords": list(obs.coords),
            "label": obs.label,
        })
    root_scores = {tid: t.tree.score for tid, t in world.tendencies.items()}
    return {
        "per_turn": per_turn,
        "root_scores": root_scores,
        "events": [e.to_dict() for e in recorder.events],
        "n_events": len(recorder.events),
    }


def run_llm_arm(corpus: List[CorpusEntry], model: str,
                status: Dict) -> Dict[str, Any]:
    """Run LLM-as-embedder on the corpus, equilibrate, return same shape."""
    world = build_charter_world()
    recorder = _EventRecorder(agent_id=f"tier3a-llm-{model}")
    per_turn: List[Dict[str, Any]] = []
    for i, entry in enumerate(corpus):
        status["current_idx"] = i + 1
        status["current_id"] = entry.id
        write_status(status)
        coords, samples = llm_score_turn(entry.turn, model=model)
        # Build the Observation manually so we can use the LLM coords;
        # mirror the autonet adapter's id derivation.
        from world_model.generalized import Observation
        from nodes.common.world_model_substrate.adapter import _obs_id_from_turn
        obs = Observation(
            id=_obs_id_from_turn(entry.turn),
            coords=coords,
            label=entry.turn.get("label", f"turn_{i}"),
        )
        before_ids = _all_node_ids(world)
        world.add_observation(obs)
        recorder.observation_added(obs)
        equilibrate(world, max_rounds=8, tolerance=1e-3)
        recorder.sub_claims_after_equilibrate(world, before_ids)
        per_turn.append({
            "id": entry.id,
            "category": entry.category,
            "coords": list(coords),
            "label": obs.label,
            "n_samples_used": len([s for s in samples
                                    if isinstance(s.get("parsed"), dict)
                                    and all(k in s["parsed"]
                                            for k in AXES)]),
        })
        # Print live progress
        cs = ",".join(f"{int(c):+d}" for c in coords)
        print(f"  [{i+1:>2}/{len(corpus)}] {entry.id:>12} ({entry.category:>20}) -> ({cs})")
    root_scores = {tid: t.tree.score for tid, t in world.tendencies.items()}
    return {
        "per_turn": per_turn,
        "root_scores": root_scores,
        "events": [e.to_dict() for e in recorder.events],
        "n_events": len(recorder.events),
    }


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def manhattan(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    return sum(abs(a[i] - b[i]) for i in range(n))


def signs_agree(a: List[float], b: List[float]) -> Tuple[int, int]:
    """Return (n_axes_agreeing_on_sign, total_axes).

    Agree means same side of zero (treating |x|<0.05 as zero).
    """
    n = min(len(a), len(b))
    agree = 0
    for i in range(n):
        sa = 0 if abs(a[i]) < 0.05 else (1 if a[i] > 0 else -1)
        sb = 0 if abs(b[i]) < 0.05 else (1 if b[i] > 0 else -1)
        if sa == sb:
            agree += 1
    return agree, n


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def evaluate_predictions(
    heuristic: Dict[str, Any],
    llm: Dict[str, Any],
    corpus: List[CorpusEntry],
) -> Tuple[int, int, List[Dict]]:
    pass_count = 0
    fail_count = 0
    predictions: List[Dict] = []

    def check(label: str, condition: bool, detail: str) -> None:
        nonlocal pass_count, fail_count
        ok = "PASS" if condition else "FAIL"
        if condition:
            pass_count += 1
        else:
            fail_count += 1
        print(f"  [{ok}] {label}: {detail}")
        predictions.append({"label": label, "condition": bool(condition), "detail": detail})

    # Build lookup tables
    h_by_id = {t["id"]: t for t in heuristic["per_turn"]}
    l_by_id = {t["id"]: t for t in llm["per_turn"]}
    cat_ids: Dict[str, List[str]] = {}
    for entry in corpus:
        cat_ids.setdefault(entry.category, []).append(entry.id)

    # A1: LLM commits binary on clearly-destructive turns (life or self < 0)
    destructive_ids = cat_ids.get("destructive", [])
    a1_failures = []
    for sid in destructive_ids:
        l = l_by_id.get(sid, {})
        coords = l.get("coords", [0, 0, 0, 0])
        if not (coords[0] <= -0.5 or coords[1] <= -0.5):
            a1_failures.append(f"{sid}={coords}")
    check("A1 (LLM commits on destructive turns)", len(a1_failures) == 0,
          f"{len(destructive_ids)} destructive turns; "
          f"{len(a1_failures)} failures" +
          (f": {a1_failures[:3]}" if a1_failures else ""))

    # A2: LLM agrees with heuristic on safe-banal turns within Manhattan 2
    banal_ids = cat_ids.get("safe_banal", [])
    a2_distances = []
    a2_failures = []
    for sid in banal_ids:
        h = h_by_id.get(sid, {})
        l = l_by_id.get(sid, {})
        d = manhattan(h.get("coords", [0]*4), l.get("coords", [0]*4))
        a2_distances.append(d)
        if d > 2.0:
            a2_failures.append(f"{sid}: dist={d:.1f} h={h.get('coords')} l={l.get('coords')}")
    avg_d = sum(a2_distances) / len(a2_distances) if a2_distances else 0
    check("A2 (LLM doesn't introduce noise on banal turns)",
          len(a2_failures) == 0,
          f"{len(banal_ids)} banal turns; avg Manhattan {avg_d:.2f}; "
          f"{len(a2_failures)} > 2.0" +
          (f": {a2_failures[:2]}" if a2_failures else ""))

    # A3: substrate root scores agree on sign for >= 80% of (turn, axis) pairs
    h_per_turn_signs = []
    l_per_turn_signs = []
    sign_agree_count = 0
    sign_total = 0
    for entry in corpus:
        h = h_by_id.get(entry.id, {}).get("coords", [0]*4)
        l = l_by_id.get(entry.id, {}).get("coords", [0]*4)
        agree, total = signs_agree(h, l)
        sign_agree_count += agree
        sign_total += total
    sign_pct = sign_agree_count / max(sign_total, 1)
    check("A3 (substrate verdicts agree on sign >= 80%)", sign_pct >= 0.80,
          f"{sign_agree_count}/{sign_total} axis-pairs agree on sign "
          f"({sign_pct*100:.1f}%, threshold 80%)")

    # A4: at least one turn where heuristic returns zeros but LLM commits
    a4_witnesses = []
    for entry in corpus:
        if entry.category in ("safe_banal", "mundane_filler"):
            continue   # those should be zero on both
        h = h_by_id.get(entry.id, {}).get("coords", [0]*4)
        l = l_by_id.get(entry.id, {}).get("coords", [0]*4)
        h_all_zero = all(abs(x) < 0.05 for x in h)
        l_committed = any(abs(x) >= 0.5 for x in l)
        if h_all_zero and l_committed:
            a4_witnesses.append({"id": entry.id, "category": entry.category,
                                  "h": h, "l": l})
    check("A4 (LLM catches what heuristic misses on >=1 turn)",
          len(a4_witnesses) >= 1,
          f"{len(a4_witnesses)} witness(es)" +
          (f"; first: {a4_witnesses[0]['id']} cat={a4_witnesses[0]['category']} "
           f"h={a4_witnesses[0]['h']} l={a4_witnesses[0]['l']}"
           if a4_witnesses else " (zero -- LLM didn't add signal)"))

    # A5: events round-trip cleanly through autonet's aggregator
    contrib = {
        "events": llm["events"],
        "agent_id": "tier3a-llm-arm",
    }
    try:
        merged = aggregate_contributions([contrib])
        # Replay onto a fresh world
        fresh = build_charter_world()
        apply_events(fresh, merged["events"])
        post_replay_scores = {tid: t.tree.score for tid, t in fresh.tendencies.items()}
        n_post_nodes = sum(len(t.tree.all_nodes()) for t in fresh.tendencies.values())
        a5_ok = n_post_nodes > len(fresh.tendencies)   # more than just the roots
        check("A5 (events round-trip through aggregator)", a5_ok,
              f"replayed {len(merged['events'])} events; "
              f"post-replay world has {n_post_nodes} nodes "
              f"(need > {len(fresh.tendencies)} root-only); "
              f"root scores: " +
              ", ".join(f"{tid}={s:.2f}" for tid, s in post_replay_scores.items()))
    except Exception as e:
        check("A5 (events round-trip through aggregator)", False,
              f"aggregator/replay raised {type(e).__name__}: {e}")

    return pass_count, fail_count, predictions


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_comparison(heuristic: Dict[str, Any], llm: Dict[str, Any],
                    corpus: List[CorpusEntry], model: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable; skipping plot)")
        return
    h_by_id = {t["id"]: t for t in heuristic["per_turn"]}
    l_by_id = {t["id"]: t for t in llm["per_turn"]}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    cat_color = {
        "destructive": "tab:red",
        "safe_banal": "tab:gray",
        "reasoning_heavy": "tab:blue",
        "capability_improving": "tab:green",
        "mundane_filler": "tab:olive",
        "real_unlabeled": "tab:purple",
    }
    for ax_idx, axis_name in enumerate(AXES):
        ax = axes[ax_idx]
        for entry in corpus:
            h = h_by_id.get(entry.id, {}).get("coords", [0]*4)
            l = l_by_id.get(entry.id, {}).get("coords", [0]*4)
            color = cat_color.get(entry.category, "k")
            ax.scatter(h[ax_idx], l[ax_idx], color=color, alpha=0.7, s=40)
        ax.plot([-1.1, 1.1], [-1.1, 1.1], "k--", linewidth=0.4, alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.3)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel(f"heuristic {axis_name[:10]}")
        ax.set_title(axis_name.replace("_", "\n"))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"LLM ({model}) score")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          color=c, markersize=8, label=lbl)
               for lbl, c in cat_color.items()]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=8)
    plt.suptitle(f"Tier 3A: heuristic vs {model} per-axis scores")
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  plot saved to {PLOT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen", "haiku"], default="qwen",
                        nargs="?")
    parser.add_argument("--real", type=int, default=0,
                        help="N real autonet turns to supplement synthesized corpus")
    args = parser.parse_args()

    corpus = get_corpus(real_supplement=args.real)
    print(f"Tier 3A: {len(corpus)} turns ({sum(1 for e in corpus if e.category != 'real_unlabeled')} synth + "
          f"{sum(1 for e in corpus if e.category == 'real_unlabeled')} real)")
    print(f"  LLM: {args.model}")
    print()

    status = {
        "started_at": time.time(),
        "phase": "heuristic",
        "n_turns": len(corpus),
        "current_idx": 0,
        "current_id": None,
        "last_update": time.time(),
    }
    write_status(status)

    print("=== heuristic arm ===")
    h_started = time.time()
    heuristic = run_heuristic_arm(corpus)
    h_elapsed = time.time() - h_started
    print(f"  ran in {h_elapsed:.1f}s")
    print(f"  root scores: " +
          ", ".join(f"{tid}={s:.2f}" for tid, s in heuristic["root_scores"].items()))
    print(f"  emitted {heuristic['n_events']} events")
    print()

    print(f"=== LLM arm ({args.model}) ===")
    status["phase"] = "llm"
    write_status(status)
    l_started = time.time()
    llm = run_llm_arm(corpus, args.model, status)
    l_elapsed = time.time() - l_started
    print(f"  ran in {l_elapsed:.1f}s")
    print(f"  root scores: " +
          ", ".join(f"{tid}={s:.2f}" for tid, s in llm["root_scores"].items()))
    print(f"  emitted {llm['n_events']} events")
    print()

    print("=" * 76)
    print("Tier 3A predictions:")
    print("=" * 76)
    pass_count, fail_count, predictions = evaluate_predictions(
        heuristic, llm, corpus
    )
    print()
    print(f"  {pass_count}/{pass_count+fail_count} predictions passed")

    print()
    plot_comparison(heuristic, llm, corpus, args.model)

    # Save trimmed results
    RESULTS_PATH.write_text(json.dumps({
        "model": args.model,
        "n_turns": len(corpus),
        "n_real": sum(1 for e in corpus if e.category == "real_unlabeled"),
        "heuristic": {
            "per_turn": heuristic["per_turn"],
            "root_scores": heuristic["root_scores"],
            "n_events": heuristic["n_events"],
            "elapsed_s": h_elapsed,
        },
        "llm": {
            "per_turn": llm["per_turn"],
            "root_scores": llm["root_scores"],
            "n_events": llm["n_events"],
            "elapsed_s": l_elapsed,
        },
        "predictions": predictions,
        "pass_count": pass_count,
        "fail_count": fail_count,
    }, indent=2), encoding="utf-8")
    print(f"  results saved to {RESULTS_PATH}")

    status["phase"] = "complete"
    status["pass_count"] = pass_count
    status["fail_count"] = fail_count
    write_status(status)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
