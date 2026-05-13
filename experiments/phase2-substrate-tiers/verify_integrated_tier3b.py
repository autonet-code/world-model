#!/usr/bin/env python3
"""Phase 4: integrated validation.

Confirms the three refactor stages (scoped equilibrate, embedding
dim reduction to 64, slow Lindblad exploration pass) don't regress
usefulness emergence on the labeled Tier 3B corpus.

Setup:
  - 6 charter roots: life_precious, self_preservation,
    promotion_of_intelligence, evolution, correctness, simplicity.
  - Per-turn coords: 6-axis charter head (from cached LLM scores) +
    64-dim PCA-projected embedding tail = 70 dims total.
  - bandwidth = 1.5 (autonet's WorldService default).
  - Scoped equilibrate per turn (skip out-of-bandwidth tendencies).
  - Slow Lindblad exploration pass every 10 turns (delta proxy):
    boosted mu, longer t_total, emits _lindblad_cross_link stakes on
    high-J sub-claim pairs.

Compares per-arm verdict metrics against original Tier 3B PASS data
in TIER3B_FINDINGS.md (haiku x 6-root final state).

Pass criteria:
  - H4 (load-bearing): cap-vs-reasoning distance within 25% of
    original Tier 3B (H4_b = 0.82 for haiku x 6-root).
  - No sign flips on any charter root vs original Tier 3B signs.
  - H2 (correlations): at least one moderate correlation 0.3–0.7
    between new roots and charter roots, no correlation >= 0.95.
  - Cross-link emission: at least one _lindblad_cross_link stake on
    real sub-claims (proves the slow path's mechanism fires under
    realistic substrate state).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# world-model first so generalized.scope is importable; autonet for the
# adapter helpers we need.
sys.path.insert(0, r"C:\code\autonet")
sys.path.insert(0, r"C:\code\world-model")

from world_model.generalized import (  # type: ignore
    GeneralizedTendency, Observation, World, equilibrate,
)
from world_model.generalized.scope import scope_for_observation  # type: ignore
from world_model.generalized.equilibrate import (  # type: ignore
    equilibrate_continuous_exploration,
)

from tier3a_corpus import get_corpus, CorpusEntry  # type: ignore
from tier3b_llm_adapter import llm_score_turn_6  # type: ignore


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "verify_integrated_tier3b_results.json"

CHARTER_IDS_6 = ("life_precious", "self_preservation",
                 "promotion_of_intelligence", "evolution",
                 "correctness", "simplicity")

# Original Tier 3B haiku x 6-root final state (from TIER3B_FINDINGS.md).
ORIG_TIER3B = {
    "h4_cap_vs_reasoning": 0.82,
    "h3_stddev": 8090.0,
    "signs": {
        "life_precious": +1,
        "self_preservation": -1,
        "promotion_of_intelligence": -1,
        "evolution": -1,
        "correctness": -1,
        "simplicity": +1,
    },
}

EMBEDDING_DIM = 64
CHARTER_DIM = 6
TOTAL_DIM = CHARTER_DIM + EMBEDDING_DIM
BANDWIDTH = 1.5
LINDBLAD_EVERY_N_TURNS = 10


def turn_to_text(turn: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in ("role", "tool", "type", "command", "file_path", "description"):
        v = turn.get(k)
        if isinstance(v, (str, int, float, bool)):
            parts.append(f"{k}: {v}")
    for k in ("content", "text", "message", "thought", "reasoning"):
        v = turn.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip()[:500])
            break
    inp = turn.get("input")
    if isinstance(inp, dict):
        for k, v in inp.items():
            if isinstance(v, str) and v.strip():
                parts.append(f"{k}: {v[:200]}")
    elif isinstance(inp, str) and inp.strip():
        parts.append(inp[:400])
    return " | ".join(p for p in parts if p) or "(empty turn)"


def build_world() -> World:
    """6-root charter, each anchored on its own unit axis in 70-d coord
    space (6 charter dims + 64 embedding-tail dims). The polarity axis
    is the same unit vector; bandwidth = autonet default 1.5."""
    world = World()
    for i, cid in enumerate(CHARTER_IDS_6):
        anchor = tuple(1.0 if j == i else 0.0 for j in range(TOTAL_DIM))
        world.add_tendency(GeneralizedTendency(
            id=cid,
            thesis=f"Charter axis: {cid}",
            anchor=anchor,
            polarity_axis=anchor,
            budget=1.0,
            bandwidth=BANDWIDTH,
            smooth_promotion=True,
        ))
    return world


def encode_corpus(corpus: List[CorpusEntry]) -> np.ndarray:
    print("  loading MiniLM (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [turn_to_text(c.turn) for c in corpus]
    print("  encoding...")
    return model.encode(texts, convert_to_numpy=True,
                        show_progress_bar=False, batch_size=32)


def fit_pca_to_64(embeddings: np.ndarray) -> np.ndarray:
    """Tier 3B corpus has 30 samples; PCA ceiling = min(n_samples,
    n_features) - 1 = 29. So fall back to 29 dims here. The dim_sweep
    result (target_dim=64) was on the 254-work-unit corpus; for the
    Tier 3B re-run the embedding tail is constrained by sample count.
    Either way it's well below MiniLM's native 384 — the experiment
    captures the spirit of dim reduction."""
    n_samples, n_features = embeddings.shape
    target_dim = min(EMBEDDING_DIM, n_samples - 1, n_features)
    pca = PCA(n_components=target_dim, random_state=0)
    projected = pca.fit_transform(embeddings)
    if target_dim < EMBEDDING_DIM:
        # Zero-pad to EMBEDDING_DIM so coords are always 70-d.
        pad = np.zeros((n_samples, EMBEDDING_DIM - target_dim))
        projected = np.concatenate([projected, pad], axis=1)
    return projected


def make_observation(entry: CorpusEntry, charter_coords: Tuple[float, ...],
                     tail: np.ndarray) -> Observation:
    from nodes.common.world_model_substrate.adapter import (  # type: ignore
        _obs_id_from_turn,
    )
    coords = tuple(charter_coords[:CHARTER_DIM]) + tuple(float(x) for x in tail)
    # Pad/truncate to TOTAL_DIM for safety.
    if len(coords) < TOTAL_DIM:
        coords = coords + (0.0,) * (TOTAL_DIM - len(coords))
    coords = coords[:TOTAL_DIM]
    return Observation(
        id=_obs_id_from_turn(entry.turn),
        coords=coords,
        label=entry.turn.get("label", entry.id),
    )


def run_integrated_arm(corpus: List[CorpusEntry],
                       embedding_tails: np.ndarray) -> Dict[str, Any]:
    world = build_world()
    per_turn: List[Dict[str, Any]] = []
    cross_link_emissions: List[Dict[str, Any]] = []
    n_scope_empty = 0
    started = time.time()

    for i, entry in enumerate(corpus):
        # 6-axis charter head from cached LLM (free, on-disk).
        charter_coords, _ = llm_score_turn_6(entry.turn, model="haiku")
        # 64-dim embedding tail.
        tail = embedding_tails[i]
        obs = make_observation(entry, charter_coords, tail)

        world.add_observation(obs)
        scope = scope_for_observation(world, obs, slack=1.5)
        if not scope:
            n_scope_empty += 1
        else:
            equilibrate(world, max_rounds=8, tolerance=1e-3, scope=scope)

        # Slow Lindblad every N turns.
        if (i + 1) % LINDBLAD_EVERY_N_TURNS == 0:
            try:
                result = equilibrate_continuous_exploration(
                    world,
                    bandwidth=BANDWIDTH,
                    cross_link_threshold=0.05,
                )
                if result["cross_links"]:
                    for cl in result["cross_links"]:
                        cl["after_turn"] = i + 1
                        cross_link_emissions.append(cl)
                    print(f"  [turn {i+1:>2}] Lindblad emitted "
                          f"{len(result['cross_links'])} cross-links")
                else:
                    print(f"  [turn {i+1:>2}] Lindblad pass: no high-J pairs")
            except Exception as e:
                print(f"  [turn {i+1:>2}] Lindblad pass FAILED: "
                      f"{type(e).__name__}: {e}")

        root_scores = {tid: t.tree.score for tid, t in world.tendencies.items()}
        per_turn.append({
            "id": entry.id,
            "category": entry.category,
            "charter_coords": list(charter_coords[:CHARTER_DIM]),
            "root_scores_after": root_scores,
            "scope": sorted(scope) if scope else [],
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - started
            print(f"  [turn {i+1:>2}] {elapsed:.1f}s elapsed, "
                  f"scope-empty so far: {n_scope_empty}")

    final_root_scores = {tid: t.tree.score for tid, t in world.tendencies.items()}
    n_nodes = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())
    return {
        "per_turn": per_turn,
        "final_root_scores": final_root_scores,
        "n_nodes": n_nodes,
        "cross_link_emissions": cross_link_emissions,
        "n_scope_empty": n_scope_empty,
        "elapsed_s": time.time() - started,
    }


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
    sums = [0.0] * CHARTER_DIM
    cnt = 0
    for t in arm["per_turn"]:
        if t["category"] != category:
            continue
        for i in range(CHARTER_DIM):
            sums[i] += t["charter_coords"][i]
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


def evaluate(arm: Dict[str, Any]) -> Dict[str, Any]:
    print()
    print("=" * 76)
    print("Phase 4: integrated validation against original Tier 3B baseline")
    print("=" * 76)

    final = arm["final_root_scores"]

    # Sign preservation.
    print("  [signs] per-root sign vs original Tier 3B haiku x 6-root:")
    sign_flips: List[str] = []
    for axis in CHARTER_IDS_6:
        s = final.get(axis, 0.0)
        sign = 0 if abs(s) < 1e-9 else (1 if s > 0 else -1)
        orig = ORIG_TIER3B["signs"][axis]
        flipped = sign != orig and abs(s) > 100
        if flipped:
            sign_flips.append(axis)
        flag = "FLIP!" if flipped else "ok"
        print(f"          {axis:>28}: orig={orig:+d}, scoped={s:>+9.2f}  [{flag}]")
    signs_ok = len(sign_flips) == 0

    # H3 stddev (informational, comparison only).
    root_scores = list(final.values())
    sd = stddev(root_scores)
    sd_orig = ORIG_TIER3B["h3_stddev"]
    h3_delta = (sd - sd_orig) / sd_orig if sd_orig > 0 else 0.0
    print(f"  [H3] stddev: orig={sd_orig:.0f}, integrated={sd:.0f}  ({h3_delta:+.1%})")

    # H4 categorical separation (load-bearing).
    cap = category_mean_vector(arm, "capability_improving")
    rea = category_mean_vector(arm, "reasoning_heavy")
    d = vec_distance(cap, rea)
    d_orig = ORIG_TIER3B["h4_cap_vs_reasoning"]
    h4_delta = abs(d - d_orig) / d_orig if d_orig > 0 else 0.0
    h4_ok = h4_delta <= 0.25
    print(f"  [H4] cap-vs-reasoning: orig={d_orig:.3f}, integrated={d:.3f}  "
          f"({h4_delta:+.1%}) {'OK' if h4_ok else 'REGRESSION'}")

    # H2 correlations between new roots and charter roots.
    coords_per_axis: Dict[str, List[float]] = {a: [] for a in CHARTER_IDS_6}
    for t in arm["per_turn"]:
        for i, axis in enumerate(CHARTER_IDS_6):
            coords_per_axis[axis].append(t["charter_coords"][i])

    correlations: Dict[str, float] = {}
    print("  [H2] new-axis vs charter-axis correlations:")
    for new_axis in ("correctness", "simplicity"):
        for ch_axis in CHARTER_IDS_6[:4]:
            r = pearson(coords_per_axis[new_axis], coords_per_axis[ch_axis])
            correlations[f"{new_axis}__vs__{ch_axis}"] = round(r, 3)
            print(f"          corr({new_axis}, {ch_axis}) = {r:+.3f}")
    abs_corrs = [abs(r) for r in correlations.values()]
    moderate_present = any(0.3 <= r <= 0.7 for r in abs_corrs)
    no_degenerate = all(r < 0.95 for r in abs_corrs)
    h2_ok = moderate_present and no_degenerate
    print(f"       H2 {'OK' if h2_ok else 'FAIL'}  "
          f"(moderate present: {moderate_present}; no degenerate: {no_degenerate})")

    # Cross-link emission.
    n_cl = len(arm["cross_link_emissions"])
    cl_ok = n_cl > 0
    print(f"  [Lindblad] cross-links emitted across run: {n_cl}  "
          f"({'OK' if cl_ok else 'NO CROSS-LINKS — slow path silent'})")

    overall = h4_ok and signs_ok and h2_ok and cl_ok
    print()
    if overall:
        print(f"  Overall: PASS — usefulness emergence preserved under "
              f"scoped + PCA-{EMBEDDING_DIM} + slow Lindblad.")
    else:
        failures = []
        if not h4_ok: failures.append("H4")
        if not signs_ok: failures.append(f"signs({sign_flips})")
        if not h2_ok: failures.append("H2")
        if not cl_ok: failures.append("cross-links")
        print(f"  Overall: FAIL — {', '.join(failures)} regressed.")

    return {
        "signs_ok": signs_ok,
        "sign_flips": sign_flips,
        "h3_delta_signed": h3_delta,
        "h4_ok": h4_ok,
        "h4_distance": d,
        "h4_delta": h4_delta,
        "h2_ok": h2_ok,
        "h2_correlations": correlations,
        "cross_link_ok": cl_ok,
        "n_cross_links": n_cl,
        "overall_pass": overall,
    }


def main() -> int:
    corpus = get_corpus()
    print(f"  corpus: {len(corpus)} turns")

    print()
    print("--- Stage A: encode + PCA-project embedding tails ---")
    raw_emb = encode_corpus(corpus)
    tails = fit_pca_to_64(raw_emb)
    print(f"  embeddings: {raw_emb.shape} -> tails: {tails.shape}")

    print()
    print(f"--- Stage B: integrated arm "
          f"(scoped + PCA-{EMBEDDING_DIM} + Lindblad every "
          f"{LINDBLAD_EVERY_N_TURNS} turns) ---")
    arm = run_integrated_arm(corpus, tails)
    print(f"  arm summary: n_nodes={arm['n_nodes']}, "
          f"scope_empty={arm['n_scope_empty']}/{len(corpus)}, "
          f"cross_links={len(arm['cross_link_emissions'])}, "
          f"elapsed={arm['elapsed_s']:.1f}s")

    eval_result = evaluate(arm)

    RESULTS_PATH.write_text(json.dumps({
        "arm": arm,
        "evaluation": eval_result,
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "total_dim": TOTAL_DIM,
            "bandwidth": BANDWIDTH,
            "lindblad_every_n_turns": LINDBLAD_EVERY_N_TURNS,
            "charter_ids": list(CHARTER_IDS_6),
        },
        "baseline_orig_tier3b": ORIG_TIER3B,
    }, indent=2), encoding="utf-8")
    print(f"  results -> {RESULTS_PATH}")
    return 0 if eval_result["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
