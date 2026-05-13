#!/usr/bin/env python3
"""Tier 3B: do extra usefulness roots produce sharper substrate verdicts?

Compares 4-charter-root substrate (Arm A) vs 4-charter + correctness +
simplicity 6-root substrate (Arm B), using haiku-4-5 as the embedder
on both arms, on the same 30-turn corpus.

For Arm A we reuse the Tier 3A haiku cache (same 4-axis prompt). For
Arm B we use the new 6-axis prompt + cache.

Usage: python run_tier3b.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, r"C:\code\world-model")
sys.path.insert(0, r"C:\code\autonet")

from world_model.generalized import (  # type: ignore
    GeneralizedTendency, Observation, World, equilibrate,
)

from tier3a_corpus import get_corpus, CorpusEntry  # type: ignore
from tier3a_llm_adapter import llm_score_turn  # 4-axis  # type: ignore
from tier3b_llm_adapter import llm_score_turn_6  # type: ignore


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "tier3b_results.json"
STATUS_PATH = HERE / "tier3b_status.json"
PLOT_PATH = HERE / "tier3b_plot.png"


CHARTER_IDS_4 = ("life_precious", "self_preservation",
                 "promotion_of_intelligence", "evolution")

CHARTER_IDS_6 = CHARTER_IDS_4 + ("correctness", "simplicity")


def write_status(status: dict) -> None:
    status["last_update"] = time.time()
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# World builders
# ---------------------------------------------------------------------------


def build_world(charter_ids: Tuple[str, ...], bandwidth: float = 1.5) -> World:
    """Build a world with one tendency per charter id, anchored on its axis."""
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


# ---------------------------------------------------------------------------
# Per-arm run
# ---------------------------------------------------------------------------


def run_arm(
    corpus: List[CorpusEntry],
    charter_ids: Tuple[str, ...],
    score_fn,
    arm_label: str,
    status: Dict,
) -> Dict[str, Any]:
    """Run a substrate arm: feed each turn through equilibrate, snapshot
    per-turn coords + per-turn root scores. Surfaces failures inline."""
    world = build_world(charter_ids)
    per_turn: List[Dict[str, Any]] = []
    n_axes = len(charter_ids)
    n_llm_failures = 0
    n_zero_coords = 0
    arm_started = time.time()
    for i, entry in enumerate(corpus):
        turn_started = time.time()
        status["arm"] = arm_label
        status["current_idx"] = i + 1
        status["current_id"] = entry.id
        status["current_category"] = entry.category
        status["n_llm_failures_so_far"] = n_llm_failures
        write_status(status)

        # LLM call with try/except so one bad turn doesn't kill the run
        try:
            coords, samples = score_fn(entry.turn)
        except Exception as e:
            n_llm_failures += 1
            print(f"  [{arm_label}] [{i+1:>2}/{len(corpus)}] {entry.id:>12} "
                  f"  !! LLM CALL FAILED: {type(e).__name__}: {e}")
            status["last_error"] = f"{entry.id}: {type(e).__name__}: {e}"
            write_status(status)
            coords = tuple(0.0 for _ in range(n_axes))
            samples = []

        # Detect "all zeros" (LLM gave no signal — usually a parse failure
        # for hard-categorized turns like destructive)
        coords_padded = tuple(coords[:n_axes]) + (0.0,) * max(0, n_axes - len(coords))
        all_zero = all(abs(c) < 0.5 for c in coords_padded)
        if all_zero and entry.category in ("destructive", "capability_improving",
                                           "reasoning_heavy"):
            n_zero_coords += 1

        from nodes.common.world_model_substrate.adapter import (  # type: ignore
            _obs_id_from_turn,
        )
        obs = Observation(
            id=_obs_id_from_turn(entry.turn),
            coords=coords_padded,
            label=entry.turn.get("label", entry.id),
        )

        try:
            world.add_observation(obs)
            equilibrate(world, max_rounds=8, tolerance=1e-3)
        except Exception as e:
            print(f"  [{arm_label}] [{i+1:>2}/{len(corpus)}] {entry.id:>12} "
                  f"  !! EQUILIBRATE FAILED: {type(e).__name__}: {e}")
            status["last_error"] = f"equilibrate {entry.id}: {type(e).__name__}: {e}"
            write_status(status)
            raise   # equilibrate failure is fatal -- don't pretend it didn't happen

        root_scores = {tid: t.tree.score for tid, t in world.tendencies.items()}
        per_turn.append({
            "id": entry.id,
            "category": entry.category,
            "coords": list(coords_padded),
            "root_scores_after": root_scores,
        })

        cs = ",".join(f"{int(c):+d}" for c in coords_padded)
        n_committed = sum(1 for c in coords_padded if abs(c) >= 0.5)
        warn = ""
        if all_zero and entry.category in ("destructive", "capability_improving",
                                           "reasoning_heavy"):
            warn = "  [!! all-zero on hard-categorized turn]"
        elapsed = time.time() - turn_started
        print(f"  [{arm_label}] [{i+1:>2}/{len(corpus)}] {entry.id:>12} "
              f"({entry.category:>20}) -> ({cs}) committed={n_committed}/{n_axes} "
              f"{elapsed:>5.1f}s{warn}")

        # Periodic progress summary every 10 turns
        if (i + 1) % 10 == 0:
            elapsed_total = time.time() - arm_started
            rate = (i + 1) / elapsed_total
            eta = (len(corpus) - i - 1) / rate if rate > 0 else 0
            print(f"  [{arm_label}] -- progress: {i+1}/{len(corpus)} "
                  f"({elapsed_total:.0f}s elapsed, ~{eta:.0f}s remaining); "
                  f"failures so far: llm={n_llm_failures}, zero-coords={n_zero_coords}")

    final_root_scores = {tid: t.tree.score for tid, t in world.tendencies.items()}
    n_nodes = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())
    print(f"  [{arm_label}] arm summary: llm_failures={n_llm_failures}, "
          f"unexpected_zero_coords={n_zero_coords}/{len(corpus)}")
    return {
        "per_turn": per_turn,
        "final_root_scores": final_root_scores,
        "n_nodes": n_nodes,
        "charter_ids": list(charter_ids),
        "n_llm_failures": n_llm_failures,
        "n_zero_coords": n_zero_coords,
    }


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------


def n_committed_pairs(arm: Dict[str, Any], axis_filter=None) -> int:
    """Count axis-pairs where coord != 0. If axis_filter is provided, only
    count those axes (by index into the arm's charter_ids)."""
    cnt = 0
    charter_ids = arm["charter_ids"]
    if axis_filter is None:
        idxs = list(range(len(charter_ids)))
    else:
        idxs = [charter_ids.index(a) for a in axis_filter if a in charter_ids]
    for t in arm["per_turn"]:
        for i in idxs:
            if abs(t["coords"][i]) >= 0.5:
                cnt += 1
    return cnt


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


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


def stddev(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def evaluate_hypotheses(arm_a: Dict[str, Any], arm_b: Dict[str, Any],
                        corpus: List[CorpusEntry]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    print()
    print("=" * 76)
    print("Tier 3B hypotheses:")
    print("=" * 76)

    # H1: signal density on shared axes (Arm B should match-or-exceed Arm A
    # on the 4 charter axes both arms share), plus extra signal on new axes.
    n_a_shared = n_committed_pairs(arm_a, axis_filter=CHARTER_IDS_4)
    n_b_shared = n_committed_pairs(arm_b, axis_filter=CHARTER_IDS_4)
    n_b_new = n_committed_pairs(arm_b, axis_filter=("correctness", "simplicity"))
    h1_shared_ok = n_b_shared >= int(0.85 * n_a_shared)  # allow 15% softening
    h1_new_signal_ok = n_b_new >= 5
    h1_pass = h1_shared_ok and h1_new_signal_ok
    print(f"  [H1] signal density on shared 4 axes: A={n_a_shared}, B={n_b_shared} "
          f"(B/A = {n_b_shared/max(n_a_shared,1):.2f}); new-axis signal: {n_b_new}")
    print(f"       H1 = {'PASS' if h1_pass else 'FAIL'}  "
          f"(shared OK: {h1_shared_ok}; new signal OK: {h1_new_signal_ok})")
    results["H1"] = {
        "pass": h1_pass,
        "shared_axes_a": n_a_shared,
        "shared_axes_b": n_b_shared,
        "new_axes_b": n_b_new,
        "shared_ok": h1_shared_ok,
        "new_signal_ok": h1_new_signal_ok,
    }

    # H2: correlation between new roots and charter roots in Arm B.
    # Build per-turn vectors for Arm B; compute correlations.
    b_coords_per_axis: Dict[str, List[float]] = {a: [] for a in CHARTER_IDS_6}
    for t in arm_b["per_turn"]:
        for i, axis in enumerate(CHARTER_IDS_6):
            b_coords_per_axis[axis].append(float(t["coords"][i]))

    correlations = {}
    for new_axis in ("correctness", "simplicity"):
        for charter_axis in CHARTER_IDS_4:
            r = pearson(b_coords_per_axis[new_axis],
                        b_coords_per_axis[charter_axis])
            correlations[f"{new_axis}_vs_{charter_axis}"] = round(r, 3)
            print(f"       corr({new_axis}, {charter_axis}) = {r:+.3f}")

    # H2 passes if at least one new-axis-vs-charter correlation is in
    # the moderate range [0.3, 0.7] AND no correlation is >= 0.95
    # (which would mean the new axis is degenerate with a charter axis).
    abs_corrs = [abs(r) for r in correlations.values()]
    moderate_present = any(0.3 <= r <= 0.7 for r in abs_corrs)
    no_degenerate = all(r < 0.95 for r in abs_corrs)
    h2_pass = moderate_present and no_degenerate
    print(f"       H2 = {'PASS' if h2_pass else 'FAIL'}  "
          f"(moderate present: {moderate_present}; no degenerate: {no_degenerate})")
    results["H2"] = {
        "pass": h2_pass,
        "correlations": correlations,
        "moderate_present": moderate_present,
        "no_degenerate": no_degenerate,
    }

    # H3: verdict separation. Compute per-arm distribution of node-level
    # scores; standard deviation in B should be >= A.
    a_root_scores = list(arm_a["final_root_scores"].values())
    b_root_scores = list(arm_b["final_root_scores"].values())
    sd_a = stddev(a_root_scores)
    sd_b = stddev(b_root_scores)
    h3_pass = sd_b >= sd_a * 0.9  # allow 10% softening for noise
    print(f"  [H3] root-score stddev: A={sd_a:.2f}, B={sd_b:.2f}")
    print(f"       H3 = {'PASS' if h3_pass else 'FAIL'}")
    results["H3"] = {
        "pass": h3_pass,
        "stddev_a": sd_a,
        "stddev_b": sd_b,
        "root_scores_a": arm_a["final_root_scores"],
        "root_scores_b": arm_b["final_root_scores"],
    }

    # H4: categorical separation between capability_improving and reasoning_heavy.
    a_cap = category_mean_vector(arm_a, "capability_improving")
    a_rea = category_mean_vector(arm_a, "reasoning_heavy")
    b_cap = category_mean_vector(arm_b, "capability_improving")
    b_rea = category_mean_vector(arm_b, "reasoning_heavy")
    # For fair comparison, project Arm B onto the 4 shared axes.
    b_cap_4 = b_cap[:4]
    b_rea_4 = b_rea[:4]
    d_a_4 = vec_distance(a_cap, a_rea)
    d_b_4 = vec_distance(b_cap_4, b_rea_4)
    d_b_6 = vec_distance(b_cap, b_rea)
    # H4 passes if Arm B 6-D separation > Arm A 4-D separation.
    h4_pass = d_b_6 > d_a_4
    print(f"  [H4] cap-vs-reasoning distance: A(4d)={d_a_4:.3f}, B(4d)={d_b_4:.3f}, B(6d)={d_b_6:.3f}")
    print(f"       H4 = {'PASS' if h4_pass else 'FAIL'}")
    results["H4"] = {
        "pass": h4_pass,
        "d_a_4d": d_a_4,
        "d_b_4d": d_b_4,
        "d_b_6d": d_b_6,
        "a_cap_mean": a_cap,
        "a_rea_mean": a_rea,
        "b_cap_mean": b_cap,
        "b_rea_mean": b_rea,
    }

    n_pass = sum(1 for h in ("H1", "H2", "H3", "H4") if results[h]["pass"])
    h2_pass_required = results["H2"]["pass"]
    overall_pass = n_pass >= 3 and h2_pass_required
    print()
    print(f"  {n_pass}/4 hypotheses pass (H2 required: {h2_pass_required})")
    print(f"  Tier 3B: {'PASS' if overall_pass else 'FAIL'}")
    results["n_pass"] = n_pass
    results["overall_pass"] = overall_pass
    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_results(arm_a: Dict[str, Any], arm_b: Dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top: bar chart of root scores
    ax = axes[0]
    a_ids = list(arm_a["final_root_scores"].keys())
    a_vals = [arm_a["final_root_scores"][k] for k in a_ids]
    b_ids = list(arm_b["final_root_scores"].keys())
    b_vals = [arm_b["final_root_scores"][k] for k in b_ids]
    x_a = list(range(len(a_ids)))
    x_b = [x + len(a_ids) + 1 for x in range(len(b_ids))]
    ax.bar(x_a, a_vals, color="tab:blue", label="Arm A (4 roots)")
    ax.bar(x_b, b_vals, color="tab:orange", label="Arm B (6 roots)")
    ax.set_xticks(x_a + x_b)
    ax.set_xticklabels(a_ids + b_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("final root score")
    ax.set_title("Final root scores by arm")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: per-turn coord heatmap for Arm B (showing the 6-axis structure)
    ax = axes[1]
    n_turns = len(arm_b["per_turn"])
    n_axes = len(arm_b["charter_ids"])
    matrix = [[0.0] * n_turns for _ in range(n_axes)]
    for j, t in enumerate(arm_b["per_turn"]):
        for i in range(n_axes):
            matrix[i][j] = t["coords"][i]
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(list(range(n_axes)))
    ax.set_yticklabels(arm_b["charter_ids"], fontsize=8)
    ax.set_xticks(list(range(n_turns)))
    ax.set_xticklabels([t["id"] for t in arm_b["per_turn"]], rotation=90, fontsize=6)
    ax.set_title("Arm B (6-root) per-turn coords")
    plt.colorbar(im, ax=ax, fraction=0.02)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  plot saved to {PLOT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def smoke_check_proxy() -> bool:
    """Hit the proxy with a trivial request to fail fast if it's down."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://127.0.0.1:3456/v1/messages",
            data=json.dumps({
                "model": "claude-haiku-4-5",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": "dummy",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  !! Proxy smoke check failed: {type(e).__name__}: {e}")
        return False


def smoke_check_ollama() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  !! Ollama smoke check failed: {type(e).__name__}: {e}")
        return False


def run_condition(label, corpus, charter_ids, score_fn, status):
    print(f"=== {label} ===")
    started = time.time()
    arm = run_arm(corpus, charter_ids, score_fn, label, status)
    elapsed = time.time() - started
    print(f"  {label} done in {elapsed:.1f}s. Final root scores:")
    for k, v in arm["final_root_scores"].items():
        print(f"    {k:30s} {v:+.4f}")
    print(f"  nodes after: {arm['n_nodes']}")
    snap_path = HERE / f"tier3b_snapshot_{label.replace('-','_')}.json"
    snap_path.write_text(
        json.dumps({**arm, "elapsed_s": elapsed, "label": label}, indent=2),
        encoding="utf-8")
    print(f"  snapshot saved to {snap_path.name}")
    print()
    return arm, elapsed


def main() -> int:
    corpus = get_corpus(real_supplement=0)
    print(f"Tier 3B: {len(corpus)} synthetic turns")
    print("  Conditions: {haiku, qwen} x {4-root, 6-root} = 4 conditions")

    print("  Smoke-checking claude-max-proxy...", end=" ")
    proxy_ok = smoke_check_proxy()
    print("OK" if proxy_ok else "FAIL")
    print("  Smoke-checking ollama...", end=" ")
    ollama_ok = smoke_check_ollama()
    print("OK" if ollama_ok else "FAIL")
    if not (proxy_ok and ollama_ok):
        print("  Aborting: both LLM backends required.")
        return 2
    print()

    status = {
        "started_at": time.time(),
        "n_turns": len(corpus),
        "current_idx": 0,
        "arm": "init",
        "last_update": time.time(),
    }
    write_status(status)

    haiku_a, haiku_a_e = run_condition(
        "haiku-4root", corpus, CHARTER_IDS_4,
        lambda t: llm_score_turn(t, model="haiku"), status)
    haiku_b, haiku_b_e = run_condition(
        "haiku-6root", corpus, CHARTER_IDS_6,
        lambda t: llm_score_turn_6(t, model="haiku"), status)
    qwen_a, qwen_a_e = run_condition(
        "qwen-4root", corpus, CHARTER_IDS_4,
        lambda t: llm_score_turn(t, model="qwen"), status)
    qwen_b, qwen_b_e = run_condition(
        "qwen-6root", corpus, CHARTER_IDS_6,
        lambda t: llm_score_turn_6(t, model="qwen"), status)

    print("=" * 76)
    print("HAIKU: 4-root vs 6-root hypothesis tests")
    print("=" * 76)
    haiku_hyp = evaluate_hypotheses(haiku_a, haiku_b, corpus)
    print()
    print("=" * 76)
    print("QWEN: 4-root vs 6-root hypothesis tests")
    print("=" * 76)
    qwen_hyp = evaluate_hypotheses(qwen_a, qwen_b, corpus)

    plot_results(haiku_a, haiku_b)

    pack_arm = lambda arm, e: {
        "charter_ids": arm["charter_ids"],
        "per_turn": arm["per_turn"],
        "final_root_scores": arm["final_root_scores"],
        "n_nodes": arm["n_nodes"],
        "n_llm_failures": arm.get("n_llm_failures", 0),
        "n_zero_coords": arm.get("n_zero_coords", 0),
        "elapsed_s": e,
    }
    RESULTS_PATH.write_text(json.dumps({
        "n_turns": len(corpus),
        "conditions": {
            "haiku_4root": pack_arm(haiku_a, haiku_a_e),
            "haiku_6root": pack_arm(haiku_b, haiku_b_e),
            "qwen_4root": pack_arm(qwen_a, qwen_a_e),
            "qwen_6root": pack_arm(qwen_b, qwen_b_e),
        },
        "hypotheses": {"haiku": haiku_hyp, "qwen": qwen_hyp},
    }, indent=2), encoding="utf-8")
    print(f"  results saved to {RESULTS_PATH}")

    status["phase"] = "complete"
    status["haiku_overall_pass"] = haiku_hyp["overall_pass"]
    status["qwen_overall_pass"] = qwen_hyp["overall_pass"]
    write_status(status)
    return 0 if (haiku_hyp["overall_pass"] or qwen_hyp["overall_pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())
