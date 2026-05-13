#!/usr/bin/env python3
"""Real-traces substrate experiment.

Replays extracted Claude session work units chronologically into the
substrate. At several checkpoints during replay, holds out an eval
set and measures:

  coverage : fraction of held-out queries where locate returns a
             non-empty region within distance threshold.
  density  : avg number of relevant nodes per query.
  cost_ms  : avg per-query locate+render time.
  agreement: fraction of held-out queries whose top-located work
             unit shares a category cluster with the ground truth.
             Categories are derived ahead of time by k-means on the
             eval set's embeddings, so this is internal-consistency,
             not LLM-judged correctness.

Outputs a JSON results file plus a printed table.

Usage
-----

  python run_experiment.py --units work_units_all.jsonl \\
                           --checkpoints 50 100 200 500 1000 \\
                           --eval-size 50 \\
                           --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make autonet substrate importable
_AUTONET = Path("C:/code/autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))

from world_model.generalized import (  # type: ignore
    CoordinateLocator,
    World,
)
from nodes.common.world_model_substrate import (  # type: ignore
    Outcome,
    apply_events,
    build_usefulness_world,
    coords_for_query,
    default_usefulness_embedder,
    train_world_model_on_usefulness,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_work_units(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------------------
# Held-out eval set: pick stable items, cluster for ground truth
# ---------------------------------------------------------------------------


def cluster_eval_set(
    eval_units: List[Dict[str, Any]],
    embedder: Any,
    n_clusters: int,
) -> Dict[str, int]:
    """Cluster eval units by embedding into n_clusters groups.
    Returns map from session_path -> cluster_id.

    Uses simple k-means in the embedding space (the same space the
    substrate uses for locate). Cluster ids are the ground truth for
    the agreement metric.
    """
    # Lazy import to keep dependency local
    try:
        from sklearn.cluster import KMeans
        import numpy as np
        embeddings = []
        keys: List[str] = []
        for unit in eval_units:
            text = unit["problem"] + " " + unit["resolution"][:500]
            vec = embedder(text)
            embeddings.append(list(vec))
            keys.append(unit["session_path"])
        if not embeddings:
            return {}
        arr = np.asarray(embeddings, dtype=float)
        km = KMeans(n_clusters=min(n_clusters, len(eval_units)), n_init=4, random_state=42)
        labels = km.fit_predict(arr)
        return {k: int(l) for k, l in zip(keys, labels)}
    except ImportError:
        # Fallback: assign clusters by hash-modulo. Cheap and deterministic.
        import hashlib
        out: Dict[str, int] = {}
        for unit in eval_units:
            h = int(hashlib.sha256(unit["session_path"].encode()).hexdigest()[:8], 16)
            out[unit["session_path"]] = h % n_clusters
        return out


# ---------------------------------------------------------------------------
# Train substrate up to a checkpoint
# ---------------------------------------------------------------------------


def train_to_checkpoint(
    chronological_units: List[Dict[str, Any]],
    checkpoint: int,
    embedder: Any,
    dim: int,
    bandwidth: float,
) -> Tuple[World, Dict[str, str]]:
    """Train the substrate on the first `checkpoint` units. Returns
    (world, node_id -> session_path index).
    """
    train = chronological_units[:checkpoint]
    work = []
    obs_to_session: Dict[str, str] = {}
    from nodes.common.world_model_substrate.usefulness_training import _obs_id  # type: ignore

    for unit in train:
        outcome = Outcome(*unit["outcome"]) if unit.get("outcome") else Outcome()
        problem = unit["problem"]
        resolution = unit["resolution"]
        work.append((problem, resolution, outcome))
        oid = _obs_id(problem, resolution)
        obs_to_session[oid] = unit["session_path"]

    contribution, _ = train_world_model_on_usefulness(
        work_units=work,
        dim=dim,
        bandwidth=bandwidth,
        epochs=1,
        agent_id="exp",
        embedder=embedder,
    )
    world = build_usefulness_world(dim=dim, bandwidth=bandwidth)
    apply_events(world, contribution["events"])

    # Build node_id -> session_path index for nodes that carry obs_id
    node_index: Dict[str, str] = {}
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            obs_id = getattr(node, "observation_id", None)
            if obs_id and obs_id in obs_to_session:
                node_index[node.id] = obs_to_session[obs_id]
    return world, node_index


# ---------------------------------------------------------------------------
# Eval at checkpoint
# ---------------------------------------------------------------------------


def eval_at_checkpoint(
    world: World,
    node_index: Dict[str, str],
    eval_units: List[Dict[str, Any]],
    eval_clusters: Dict[str, int],
    embedder: Any,
    locator: CoordinateLocator,
    distance_threshold: float = 1.5,
) -> Dict[str, float]:
    """Evaluate the substrate's locate at this checkpoint."""
    n_total = len(eval_units)
    n_covered = 0
    densities: List[int] = []
    costs_ms: List[float] = []
    agreements = 0

    # Build query -> cluster lookup so we can score agreement on
    # nearest-located neighbour cluster.
    eval_session_to_cluster = eval_clusters
    train_session_to_cluster = eval_clusters   # eval clusters are the only ground truth we have

    # Map from each held-out query's session_path -> cluster id, plus
    # node_index for training items. Agreement = does the locator's
    # top-K return ANY node whose session shares a cluster with the
    # query's session?
    for unit in eval_units:
        query = unit["problem"]
        true_cluster = eval_session_to_cluster.get(unit["session_path"])

        t0 = time.perf_counter()
        qcoords = coords_for_query(query, embedder=embedder)
        region = locator(world, qcoords)
        cost_ms = (time.perf_counter() - t0) * 1000
        costs_ms.append(cost_ms)

        # Filter region to nodes within distance_threshold (already done
        # by locator's max_distance, but we count "covered" by whether
        # any nearby node was returned).
        if region:
            n_covered += 1
            densities.append(len(region))
        else:
            densities.append(0)

        # Agreement: do any of the top-K returned nodes belong to the
        # same cluster as the query?
        for tendency_id, node_id, _dist in region[:5]:
            session = node_index.get(node_id)
            if session is None:
                continue
            cand_cluster = train_session_to_cluster.get(session)
            if cand_cluster is None or true_cluster is None:
                continue
            if cand_cluster == true_cluster:
                agreements += 1
                break

    return {
        "coverage": n_covered / n_total if n_total else 0.0,
        "avg_density": (sum(densities) / len(densities)) if densities else 0.0,
        "avg_cost_ms": (sum(costs_ms) / len(costs_ms)) if costs_ms else 0.0,
        "agreement": agreements / n_total if n_total else 0.0,
        "n_eval": n_total,
        "n_world_nodes": sum(len(t.tree.all_nodes()) for t in world.tendencies.values()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", default="work_units_all.jsonl")
    parser.add_argument("--checkpoints", type=int, nargs="+",
                        default=[50, 100, 200, 400, 800])
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--n-clusters", type=int, default=8,
                        help="number of clusters for ground-truth labeling")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--bandwidth", type=float, default=0.6)
    parser.add_argument("--distance-threshold", type=float, default=1.5)
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--status", default="status.json",
                        help="live status file updated during the run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Status writer: small helper for observability.
    status_path = Path(args.status)
    status: Dict[str, Any] = {
        "phase": "starting",
        "started_at": time.time(),
        "checkpoints_done": [],
        "current_checkpoint": None,
        "n_units_total": 0,
    }

    def write_status() -> None:
        try:
            with status_path.open("w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)
        except Exception:
            pass

    write_status()

    units = load_work_units(Path(args.units))
    print(f"  loaded {len(units)} work units from {args.units}")
    status["n_units_total"] = len(units)
    status["phase"] = "loaded_units"
    write_status()

    # Sort chronologically -- if timestamp is missing, sort by session_path
    units.sort(key=lambda u: u.get("timestamp") or u.get("session_path", ""))

    # Hold out the LAST eval_size as the held-out eval set
    if args.eval_size >= len(units):
        print(f"  -- not enough units; need > {args.eval_size}, have {len(units)}")
        return 1
    eval_units = units[-args.eval_size:]
    chronological_train = units[:-args.eval_size]
    print(f"  chronological train pool: {len(chronological_train)}")
    print(f"  held-out eval set: {len(eval_units)}")

    # Set up embedder
    embedder = default_usefulness_embedder(dim=args.dim)
    print(f"  embedder: {type(embedder).__name__} dim={args.dim}")

    # Cluster eval set for ground-truth agreement metric
    print(f"  clustering eval set into {args.n_clusters} clusters...")
    eval_clusters = cluster_eval_set(eval_units, embedder, args.n_clusters)
    print(f"  cluster sizes:", end=" ")
    sizes: Dict[int, int] = {}
    for c in eval_clusters.values():
        sizes[c] = sizes.get(c, 0) + 1
    for c, n in sorted(sizes.items()):
        print(f"c{c}={n}", end=" ")
    print()

    # Run at each checkpoint
    locator = CoordinateLocator(
        max_distance=args.distance_threshold,
        max_results=64,
    )

    print(f"\n  {'checkpoint':>11}  {'nodes':>6}  {'coverage':>9}  "
          f"{'density':>8}  {'cost_ms':>8}  {'agreement':>10}  {'train_s':>8}")
    print(f"  {'-' * 75}")

    results: List[Dict[str, Any]] = []
    for cp in args.checkpoints:
        if cp > len(chronological_train):
            print(f"    -- checkpoint {cp} > train pool {len(chronological_train)}; skipping")
            continue
        status["current_checkpoint"] = {
            "size": cp, "phase": "training", "started_at": time.time(),
        }
        write_status()
        t0 = time.time()
        world, node_index = train_to_checkpoint(
            chronological_train, cp,
            embedder=embedder,
            dim=args.dim,
            bandwidth=args.bandwidth,
        )
        train_s = time.time() - t0
        status["current_checkpoint"]["phase"] = "evaluating"
        status["current_checkpoint"]["train_seconds"] = train_s
        status["current_checkpoint"]["n_world_nodes"] = sum(
            len(t.tree.all_nodes()) for t in world.tendencies.values()
        )
        write_status()

        # Build train index: also include eval clusters for agreement
        # mapping. We measure whether the located node's session shares
        # a cluster with the query's session.
        # For training items not in eval set, give them their own
        # "nearest-eval-cluster" as a proxy: embed each training item
        # and assign to the eval cluster whose centroid it's closest to.
        # Cheap version: just leave node_index alone -- agreement over
        # train items won't fire unless they happen to be in eval set,
        # which they aren't. So we compute a separate train_to_cluster
        # mapping using the eval embedder.
        train_to_cluster = _train_to_cluster(
            chronological_train[:cp], eval_units, eval_clusters, embedder,
        )

        # Combined map: any session_path -> cluster id
        all_clusters = {**eval_clusters, **train_to_cluster}

        result = eval_at_checkpoint_with_clusters(
            world, node_index, eval_units, all_clusters,
            embedder, locator,
        )
        result["checkpoint"] = cp
        result["train_seconds"] = train_s
        results.append(result)
        print(
            f"  {cp:>11d}  {result['n_world_nodes']:>6d}  "
            f"{result['coverage']:>8.1%}  "
            f"{result['avg_density']:>8.1f}  "
            f"{result['avg_cost_ms']:>7.1f}ms  "
            f"{result['agreement']:>9.1%}  "
            f"{train_s:>7.2f}"
        )
        # update status with latest result
        status["checkpoints_done"].append(result)
        status["current_checkpoint"] = None
        write_status()

    # Save full results
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "n_units_total": len(units),
            "n_train_pool": len(chronological_train),
            "n_eval": len(eval_units),
            "results": results,
        }, f, indent=2)
    print(f"\n  results saved to {out_path}")
    status["phase"] = "done"
    status["finished_at"] = time.time()
    write_status()

    # Honest reading
    print()
    print("=" * 70)
    print("HONEST READING")
    print("=" * 70)
    if not results:
        print("  no results to interpret")
        return 1

    first = results[0]
    last = results[-1]
    print(f"\n  At first checkpoint ({first['checkpoint']} train units):")
    print(f"    coverage:  {first['coverage']:.1%}")
    print(f"    agreement: {first['agreement']:.1%}")
    print(f"  At last checkpoint ({last['checkpoint']} train units):")
    print(f"    coverage:  {last['coverage']:.1%}")
    print(f"    agreement: {last['agreement']:.1%}")
    print(f"\n  delta coverage:  {(last['coverage'] - first['coverage']) * 100:+.1f}pp")
    print(f"  delta agreement: {(last['agreement'] - first['agreement']) * 100:+.1f}pp")

    if last["coverage"] > first["coverage"] + 0.05:
        print(f"\n  Coverage is climbing with scale -- substrate accumulates relevant structure.")
    else:
        print(f"\n  Coverage is flat. The graph may not be capturing additional structure")
        print(f"  per added work unit at this scale.")

    if last["agreement"] > first["agreement"] + 0.05:
        print(f"  Agreement is climbing -- located neighbours match query category better")
        print(f"  as more units are seen.")
    elif last["agreement"] >= first["agreement"]:
        print(f"  Agreement is roughly flat -- the substrate's matches don't sharpen with")
        print(f"  more data (within this experiment's resolution).")
    else:
        print(f"  Agreement DROPPED with scale; suggests noise dominates as more diverse")
        print(f"  work units come in. Worth investigating.")
    return 0


def _train_to_cluster(
    train_units: List[Dict[str, Any]],
    eval_units: List[Dict[str, Any]],
    eval_clusters: Dict[str, int],
    embedder: Any,
) -> Dict[str, int]:
    """Assign each training unit to the cluster of its nearest eval unit."""
    try:
        import numpy as np
        eval_embeddings = []
        eval_paths = []
        for u in eval_units:
            v = embedder(u["problem"] + " " + u["resolution"][:500])
            eval_embeddings.append(list(v))
            eval_paths.append(u["session_path"])
        if not eval_embeddings:
            return {}
        eval_arr = np.asarray(eval_embeddings, dtype=float)

        out: Dict[str, int] = {}
        for u in train_units:
            v = embedder(u["problem"] + " " + u["resolution"][:500])
            v_arr = np.asarray(v, dtype=float)
            dists = np.linalg.norm(eval_arr - v_arr[None, :], axis=1)
            i = int(np.argmin(dists))
            cluster = eval_clusters.get(eval_paths[i], 0)
            out[u["session_path"]] = cluster
        return out
    except ImportError:
        return {}


def eval_at_checkpoint_with_clusters(
    world: World,
    node_index: Dict[str, str],
    eval_units: List[Dict[str, Any]],
    all_clusters: Dict[str, int],
    embedder: Any,
    locator: CoordinateLocator,
) -> Dict[str, float]:
    """Same as eval_at_checkpoint but with explicit train_to_cluster
    map merged in so agreement can fire on located training nodes.
    """
    n_total = len(eval_units)
    n_covered = 0
    densities: List[int] = []
    costs_ms: List[float] = []
    agreements = 0

    for unit in eval_units:
        query = unit["problem"]
        true_cluster = all_clusters.get(unit["session_path"])

        t0 = time.perf_counter()
        qcoords = coords_for_query(query, embedder=embedder)
        region = locator(world, qcoords)
        cost_ms = (time.perf_counter() - t0) * 1000
        costs_ms.append(cost_ms)

        if region:
            n_covered += 1
            densities.append(len(region))
        else:
            densities.append(0)

        for tendency_id, node_id, _dist in region[:5]:
            session = node_index.get(node_id)
            if session is None:
                continue
            cand_cluster = all_clusters.get(session)
            if cand_cluster is None or true_cluster is None:
                continue
            if cand_cluster == true_cluster:
                agreements += 1
                break

    return {
        "coverage": n_covered / n_total if n_total else 0.0,
        "avg_density": (sum(densities) / len(densities)) if densities else 0.0,
        "avg_cost_ms": (sum(costs_ms) / len(costs_ms)) if costs_ms else 0.0,
        "agreement": agreements / n_total if n_total else 0.0,
        "n_eval": n_total,
        "n_world_nodes": sum(len(t.tree.all_nodes()) for t in world.tendencies.values()),
    }


if __name__ == "__main__":
    sys.exit(main())
