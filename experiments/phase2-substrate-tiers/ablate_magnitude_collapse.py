#!/usr/bin/env python3
"""Phase 4 follow-up: ablation grid to isolate which variable causes
the H3 magnitude collapse observed in verify_integrated_tier3b.py.

Original Tier 3B haiku x 6-root: stddev = 8090.
Phase 4 integrated config: stddev = 11.

Variables flipped one at a time from the integrated config to find
which one (or combination) is responsible:

  - EMBED_TAIL: PCA-64 appended vs charter-only-6 coords.
  - BANDWIDTH: 1.5 vs 2.5 vs 5.0.
  - LINDBLAD: slow exploration pass enabled vs disabled.
  - SCOPED: scoped equilibrate vs unscoped.

Total: 16 configurations. Each runs the full 30-turn haiku 6-root
corpus (LLM-cached, free). Reports stddev, H4, signs, sign flips,
per-root final scores for each cell.

Output: ablate_magnitude_results.json plus console table.
"""

from __future__ import annotations

import json
import math
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

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
RESULTS_PATH = HERE / "ablate_magnitude_results.json"

CHARTER_IDS_6 = ("life_precious", "self_preservation",
                 "promotion_of_intelligence", "evolution",
                 "correctness", "simplicity")
CHARTER_DIM = 6
EMBEDDING_DIM = 64
TOTAL_DIM_WITH_EMBED = CHARTER_DIM + EMBEDDING_DIM
LINDBLAD_EVERY_N_TURNS = 10

ORIG_STDDEV = 8090.0
ORIG_H4 = 0.82
ORIG_SIGNS = {
    "life_precious": +1, "self_preservation": -1,
    "promotion_of_intelligence": -1, "evolution": -1,
    "correctness": -1, "simplicity": +1,
}


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


def build_world(total_dim: int, bandwidth: float) -> World:
    world = World()
    for i, cid in enumerate(CHARTER_IDS_6):
        anchor = tuple(1.0 if j == i else 0.0 for j in range(total_dim))
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


def encode_tails(corpus: List[CorpusEntry]) -> np.ndarray:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [turn_to_text(c.turn) for c in corpus]
    raw = model.encode(texts, convert_to_numpy=True,
                       show_progress_bar=False, batch_size=32)
    target = min(EMBEDDING_DIM, raw.shape[0] - 1, raw.shape[1])
    pca = PCA(n_components=target, random_state=0)
    projected = pca.fit_transform(raw)
    if target < EMBEDDING_DIM:
        pad = np.zeros((raw.shape[0], EMBEDDING_DIM - target))
        projected = np.concatenate([projected, pad], axis=1)
    return projected


def stddev(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def category_mean_vector(per_turn: List[Dict[str, Any]], category: str) -> List[float]:
    sums = [0.0] * CHARTER_DIM
    cnt = 0
    for t in per_turn:
        if t["category"] != category:
            continue
        for i in range(CHARTER_DIM):
            sums[i] += t["coords"][i]
        cnt += 1
    if cnt == 0:
        return sums
    return [s / cnt for s in sums]


def vec_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(min(len(a), len(b)))))


def run_config(
    corpus: List[CorpusEntry],
    tails: np.ndarray,
    *,
    use_embed: bool,
    bandwidth: float,
    use_lindblad: bool,
    use_scope: bool,
) -> Dict[str, Any]:
    from nodes.common.world_model_substrate.adapter import (  # type: ignore
        _obs_id_from_turn,
    )

    total_dim = TOTAL_DIM_WITH_EMBED if use_embed else CHARTER_DIM
    world = build_world(total_dim, bandwidth)
    per_turn: List[Dict[str, Any]] = []
    n_cross_links = 0
    started = time.time()

    for i, entry in enumerate(corpus):
        charter, _ = llm_score_turn_6(entry.turn, model="haiku")
        if use_embed:
            coords = tuple(float(x) for x in charter[:CHARTER_DIM]) + \
                     tuple(float(x) for x in tails[i])
        else:
            coords = tuple(float(x) for x in charter[:CHARTER_DIM])
        if len(coords) < total_dim:
            coords = coords + (0.0,) * (total_dim - len(coords))
        coords = coords[:total_dim]

        obs = Observation(
            id=_obs_id_from_turn(entry.turn),
            coords=coords,
            label=entry.turn.get("label", entry.id),
        )
        world.add_observation(obs)

        if use_scope:
            scope = scope_for_observation(world, obs, slack=1.5)
            if scope:
                equilibrate(world, max_rounds=8, tolerance=1e-3, scope=scope)
        else:
            equilibrate(world, max_rounds=8, tolerance=1e-3)

        if use_lindblad and (i + 1) % LINDBLAD_EVERY_N_TURNS == 0:
            try:
                result = equilibrate_continuous_exploration(
                    world, bandwidth=bandwidth, cross_link_threshold=0.05,
                )
                n_cross_links += len(result["cross_links"])
            except Exception:
                pass

        per_turn.append({
            "id": entry.id,
            "category": entry.category,
            "coords": list(coords[:CHARTER_DIM]),
            "root_scores_after": {tid: t.tree.score
                                  for tid, t in world.tendencies.items()},
        })

    final = {tid: t.tree.score for tid, t in world.tendencies.items()}
    sd = stddev(list(final.values()))

    cap = category_mean_vector(per_turn, "capability_improving")
    rea = category_mean_vector(per_turn, "reasoning_heavy")
    h4 = vec_distance(cap, rea)

    sign_flips = []
    for axis in CHARTER_IDS_6:
        s = final.get(axis, 0.0)
        sign = 0 if abs(s) < 1e-9 else (1 if s > 0 else -1)
        orig = ORIG_SIGNS[axis]
        # Lenient flip detection: any non-zero sign mismatch counts.
        if sign != 0 and sign != orig:
            sign_flips.append(axis)

    return {
        "final_root_scores": final,
        "stddev": sd,
        "h4": h4,
        "sign_flips": sign_flips,
        "n_cross_links": n_cross_links,
        "elapsed_s": time.time() - started,
    }


def main() -> int:
    corpus = get_corpus()
    print(f"  corpus: {len(corpus)} turns; loading MiniLM + PCA-64...")
    tails = encode_tails(corpus)

    # Configurations to ablate. Variables:
    #   use_embed: True/False
    #   bandwidth: 1.5, 2.5, 5.0
    #   use_lindblad: True/False
    #   use_scope: True/False
    # That's 2*3*2*2 = 24 cells. To keep it tractable we do a smaller
    # grid: hold use_scope=True (the production target), vary the
    # other 3, then add a few unscoped controls.
    configs: List[Dict[str, Any]] = []
    # Production-target grid (12):
    for use_embed in (False, True):
        for bandwidth in (1.5, 2.5, 5.0):
            for use_lindblad in (False, True):
                configs.append({
                    "label": f"embed={use_embed} bw={bandwidth} lb={use_lindblad} scoped=True",
                    "use_embed": use_embed,
                    "bandwidth": bandwidth,
                    "use_lindblad": use_lindblad,
                    "use_scope": True,
                })
    # Unscoped controls (baseline matches original Tier 3B):
    for use_embed in (False, True):
        for use_lindblad in (False, True):
            configs.append({
                "label": f"embed={use_embed} bw=1.5 lb={use_lindblad} scoped=False",
                "use_embed": use_embed,
                "bandwidth": 1.5,
                "use_lindblad": use_lindblad,
                "use_scope": False,
            })

    results: List[Dict[str, Any]] = []
    print()
    print(f"  Running {len(configs)} configurations...")
    print()
    hdr = f"  {'config':<58}  {'stddev':>9}  {'H4':>6}  {'flips':>30}  {'CL':>3}  {'sec':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cfg in configs:
        try:
            r = run_config(
                corpus, tails,
                use_embed=cfg["use_embed"],
                bandwidth=cfg["bandwidth"],
                use_lindblad=cfg["use_lindblad"],
                use_scope=cfg["use_scope"],
            )
        except Exception as e:
            print(f"  {cfg['label']:<58}  FAILED: {type(e).__name__}: {e}")
            continue
        flip_str = ",".join(r["sign_flips"]) if r["sign_flips"] else "-"
        print(f"  {cfg['label']:<58}  {r['stddev']:>9.1f}  "
              f"{r['h4']:>6.3f}  {flip_str:>30}  {r['n_cross_links']:>3d}  "
              f"{r['elapsed_s']:>5.1f}")
        results.append({**cfg, **r})

    print()
    print(f"  Original Tier 3B baseline:  stddev=8090, H4=0.82")
    print()

    # Headline question: which single variable causes the magnitude collapse?
    print("  === Interpretation ===")
    baseline_match = [r for r in results
                      if not r["use_embed"] and r["bandwidth"] == 1.5
                      and not r["use_lindblad"] and not r["use_scope"]]
    if baseline_match:
        b = baseline_match[0]
        print(f"  Replication baseline (charter-only, bw=1.5, no Lindblad, unscoped):")
        print(f"    stddev = {b['stddev']:.1f}  (orig: 8090; ratio: {b['stddev']/8090:.3f}x)")
        print(f"    H4 = {b['h4']:.3f}  (orig: 0.82)")

    RESULTS_PATH.write_text(json.dumps({
        "orig_baseline": {"stddev": ORIG_STDDEV, "h4": ORIG_H4},
        "configs": results,
    }, indent=2), encoding="utf-8")
    print(f"  results -> {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
