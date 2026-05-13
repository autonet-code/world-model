#!/usr/bin/env python3
"""Phase 6 substrate builder for a given N (tendency count).

Builds a substrate using axes 1..N of the 10 tendencies. Reads the
shared judge_cache.jsonl (all 10 axes) and truncates to the active
subset at sprout time. Writes a snapshot file with node counts and
work-item density.

Usage:
    python build_substrate.py --n 6 --out substrate_N6.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_AUTONET = Path(r"C:\code\autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))

from world_model.generalized import (  # type: ignore  # noqa: E402
    GeneralizedTendency, Observation, World, equilibrate,
)
from world_model.generalized.scope import scope_for_observation  # type: ignore  # noqa: E402
from world_model.models.tree import Position  # type: ignore  # noqa: E402

from nodes.common.world_model_substrate.usefulness_coords import (  # type: ignore  # noqa: E402
    default_usefulness_embedder,
)


HERE = Path(__file__).resolve().parent
CORPUS_PATH = HERE / "corpus.json"
CACHE_PATH = HERE / "judge_cache.jsonl"


TENDENCIES = [
    "correctness", "simplicity", "robustness", "purity",
    "laziness", "composability", "type_flexibility", "error_clarity",
    "efficiency", "documentation_fidelity",
]


EMBEDDING_DIM = 64
BANDWIDTH = 1.5


def build_world(active_tendencies: List[str], total_dim: int) -> World:
    world = World()
    for i, tid in enumerate(active_tendencies):
        anchor = tuple(1.0 if j == i else 0.0 for j in range(total_dim))
        world.add_tendency(GeneralizedTendency(
            id=tid,
            thesis=f"Charter axis: {tid}",
            anchor=anchor,
            polarity_axis=anchor,
            budget=1.0,
            bandwidth=BANDWIDTH,
            smooth_promotion=True,
        ))
    return world


def coords_for_text(text: str, n_axes: int, embedder, axis_idx: int = -1) -> Tuple[float, ...]:
    """Coord vector: charter head [n_axes dims] + embedding tail [EMBEDDING_DIM].
    If axis_idx >= 0, sets a +1 on that charter axis. Otherwise charter head
    is zeros (used for observation nodes that don't lean any specific axis)."""
    tail = embedder(text)
    tail_t = tuple(float(x) for x in tail)[:EMBEDDING_DIM]
    if len(tail_t) < EMBEDDING_DIM:
        tail_t = tail_t + (0.0,) * (EMBEDDING_DIM - len(tail_t))
    head = tuple(1.0 if i == axis_idx else 0.0 for i in range(n_axes))
    return head + tail_t


def load_cache() -> Dict[str, Dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            out[row["name"]] = row
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True,
                        help="Number of active tendencies (axes 1..n).")
    parser.add_argument("--out", required=True, help="Output JSON snapshot path.")
    args = parser.parse_args()

    active = TENDENCIES[:args.n]
    total_dim = args.n + EMBEDDING_DIM
    print(f"  N = {args.n}, active tendencies: {active}")

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    train = corpus["train"]
    cache = load_cache()
    print(f"  train: {len(train)}, cache: {len(cache)}")

    embedder = default_usefulness_embedder(dim=EMBEDDING_DIM)
    world = build_world(active, total_dim)

    started = time.time()
    n_subclaims_emitted = 0
    n_subclaims_in_cache = 0
    per_axis_emitted: Dict[str, int] = {a: 0 for a in active}

    for i, entry in enumerate(train, start=1):
        text = f"{entry['name']}\n\n{entry['docstring']}\n\n{entry.get('impl_full_source','')}"
        obs_coords = coords_for_text(text, args.n, embedder, axis_idx=-1)
        obs = Observation(id=f"f_{entry['name']}", coords=obs_coords, label=entry["name"])
        world.add_observation(obs)

        sprouter = world.tendencies[active[0]]
        obs_node = sprouter.sprout_child(
            parent_node_id=sprouter.tree.root_node.id,
            position=Position.PRO,
            anchor=obs_coords,
            polarity_axis=sprouter.polarity_axis,
            observation=obs,
            content=entry["name"],
            world=world,
        )

        scope = scope_for_observation(world, obs, slack=1.5)
        if scope:
            equilibrate(world, max_rounds=4, tolerance=1e-3, scope=scope)

        parsed = (cache.get(entry["name"]) or {}).get("parsed") or {}
        for axis_idx, axis in enumerate(active):
            claims = parsed.get(axis, []) or []
            n_subclaims_in_cache += len(claims)
            tendency = world.tendencies[axis]
            for claim_text in claims:
                if not claim_text:
                    continue
                claim_coords = coords_for_text(claim_text, args.n, embedder, axis_idx=axis_idx)
                try:
                    tendency.sprout_child(
                        parent_node_id=obs_node.id,
                        position=Position.PRO,
                        anchor=claim_coords,
                        polarity_axis=tendency.polarity_axis,
                        content=claim_text,
                        world=world,
                    )
                    n_subclaims_emitted += 1
                    per_axis_emitted[axis] += 1
                except Exception:
                    pass

        if i % 20 == 0:
            elapsed = time.time() - started
            print(f"  [{i:>3}/{len(train)}] elapsed={elapsed:.0f}s, sub-claims={n_subclaims_emitted}")

    equilibrate(world, max_rounds=4, tolerance=1e-3)

    n_total_nodes = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())
    work_items = 0
    multi_count_dist: Dict[int, int] = {}
    for t in world.tendencies.values():
        for n in t.tree.all_nodes():
            if n.id == t.tree.root_node.id:
                continue
            tendency_ids = {p.tendency_id for p in n.parents}
            multi_count_dist[len(tendency_ids)] = multi_count_dist.get(len(tendency_ids), 0) + 1
            if len(tendency_ids) > 1:
                work_items += 1

    snapshot = {
        "n_tendencies": args.n,
        "active_tendencies": active,
        "n_train": len(train),
        "n_total_nodes": n_total_nodes,
        "n_subclaims_emitted": n_subclaims_emitted,
        "n_subclaims_in_cache": n_subclaims_in_cache,
        "per_axis_emitted": per_axis_emitted,
        "n_work_items_multi_tendency": work_items,
        "multi_count_distribution": multi_count_dist,
        "elapsed_s": time.time() - started,
    }
    Path(args.out).write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print()
    print(f"  TOTAL nodes: {n_total_nodes}")
    print(f"  sub-claims emitted: {n_subclaims_emitted}")
    print(f"  work items (multi-parent): {work_items}")
    print(f"  multi-count distribution: {multi_count_dist}")
    print(f"  snapshot -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
