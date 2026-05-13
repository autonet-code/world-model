#!/usr/bin/env python3
"""Phase 6 contest runner.

For a given N (tendency count), rebuilds the substrate using axes 1..N
of the cached judges, then runs two contestants on each held-out test
problem:

  haiku + substrate (production probe)
  haiku + RAG       (top-k cosine similarity over train embeddings)

Scores via doctest harness. Writes:
  - contest_N{N}.jsonl  (one row per problem, both impls + scores)
  - contest_progress.jsonl  (appended; mid-run readable)

Usage:
    python run_contest.py --n 6 --out contest_N6.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

_AUTONET = Path(r"C:\code\autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))

sys.path.insert(0, str(Path(r"D:\videos\SF\manifesting\from_endstate\new physics\substrate_experiment\phase5")))

from world_model.generalized import (  # type: ignore  # noqa: E402
    GeneralizedTendency, Observation, World, equilibrate,
)
from world_model.generalized.scope import scope_for_observation  # type: ignore  # noqa: E402
from world_model.models.tree import Position  # type: ignore  # noqa: E402

from nodes.common.world_model_substrate.usefulness_coords import (  # type: ignore  # noqa: E402
    default_usefulness_embedder,
)
from nodes.common.world_model_substrate.infer import (  # type: ignore  # noqa: E402
    infer_with_world_model,
)
from atn.providers.bridge import BridgeProvider  # type: ignore  # noqa: E402

from doctest_harness import grade_implementation  # type: ignore  # noqa: E402


HERE = Path(__file__).resolve().parent
CORPUS_PATH = HERE / "corpus.json"
CACHE_PATH = HERE / "judge_cache.jsonl"
PROGRESS_PATH = HERE / "contest_progress.jsonl"


TENDENCIES = [
    "correctness", "simplicity", "robustness", "purity",
    "laziness", "composability", "type_flexibility", "error_clarity",
    "efficiency", "documentation_fidelity",
]


EMBEDDING_DIM = 64
BANDWIDTH = 1.5
TOP_K = 5


CONTESTANT_SYSTEM = """You are completing the body of a Python function.

You will be given:
  - A function signature with a sparse docstring containing 1-2 examples.
  - Optionally, related material to consult.

Your job: write ONLY the function body that satisfies the docstring's examples.

Rules:
  - Output ONLY the Python source for the function (def ...:\\n    body).
  - Do NOT include explanatory text, markdown, or commentary.
  - The function must be named exactly as the signature specifies.
  - Use only Python stdlib (and `itertools`).
"""


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


def coords_for_text(text: str, n_axes: int, embedder, axis_idx: int = -1) -> Tuple[float, ...]:
    tail = embedder(text)
    tail_t = tuple(float(x) for x in tail)[:EMBEDDING_DIM]
    if len(tail_t) < EMBEDDING_DIM:
        tail_t = tail_t + (0.0,) * (EMBEDDING_DIM - len(tail_t))
    head = tuple(1.0 if i == axis_idx else 0.0 for i in range(n_axes))
    return head + tail_t


def build_world_for_n(corpus: Dict[str, Any], n: int, embedder) -> Tuple[World, Dict[str, Any]]:
    active = TENDENCIES[:n]
    total_dim = n + EMBEDDING_DIM
    world = World()
    for i, tid in enumerate(active):
        anchor = tuple(1.0 if j == i else 0.0 for j in range(total_dim))
        world.add_tendency(GeneralizedTendency(
            id=tid, thesis=f"Charter axis: {tid}",
            anchor=anchor, polarity_axis=anchor, budget=1.0,
            bandwidth=BANDWIDTH, smooth_promotion=True,
        ))

    cache = load_cache()
    train = corpus["train"]
    train_index: Dict[str, Dict[str, Any]] = {}

    for entry in train:
        text = f"{entry['name']}\n\n{entry['docstring']}\n\n{entry.get('impl_full_source','')}"
        obs_coords = coords_for_text(text, n, embedder, axis_idx=-1)
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
        # Embedding tail for RAG.
        tail_t = obs_coords[n:]
        train_index[entry["name"]] = {"obs_node_id": obs_node.id, "embedding": tail_t}

        parsed = (cache.get(entry["name"]) or {}).get("parsed") or {}
        for axis_idx, axis in enumerate(active):
            for claim_text in parsed.get(axis, []) or []:
                if not claim_text:
                    continue
                claim_coords = coords_for_text(claim_text, n, embedder, axis_idx=axis_idx)
                try:
                    world.tendencies[axis].sprout_child(
                        parent_node_id=obs_node.id,
                        position=Position.PRO,
                        anchor=claim_coords,
                        polarity_axis=world.tendencies[axis].polarity_axis,
                        content=claim_text,
                        world=world,
                    )
                except Exception:
                    pass

    equilibrate(world, max_rounds=4, tolerance=1e-3)
    return world, train_index


def topk_by_embedding(test_entry, train_index, embedder, k=TOP_K) -> List[Dict[str, Any]]:
    q_tail = embedder(f"{test_entry['name']}\n\n{test_entry['sparse_docstring']}")
    q_arr = np.array(q_tail, dtype=np.float32)
    q_norm = np.linalg.norm(q_arr) + 1e-9
    scored: List[Tuple[float, str]] = []
    for name, idx in train_index.items():
        e_arr = np.array(idx["embedding"], dtype=np.float32)
        cos = float(np.dot(q_arr, e_arr) / (q_norm * (np.linalg.norm(e_arr) + 1e-9)))
        scored.append((cos, name))
    scored.sort(reverse=True)
    return [{"name": n, "similarity": s} for s, n in scored[:k]]


def format_rag_context(top: List[Dict[str, Any]], train_by_name: Dict[str, Dict[str, Any]]) -> str:
    lines = ["RELATED FUNCTIONS (most similar):", ""]
    for item in top:
        e = train_by_name[item["name"]]
        lines.append(f"--- {e['name']} ({e['signature']}) ---")
        lines.append(e["docstring"])
        lines.append("")
        lines.append(e.get("impl_full_source", ""))
        lines.append("")
    return "\n".join(lines)


def format_substrate_context(probe_result: Dict[str, Any]) -> str:
    lines = ["GRAPH REGION (substrate probe):", ""]
    region = probe_result.get("region") or probe_result.get("rendered") or []
    if not isinstance(region, list):
        region = [region]
    for i, item in enumerate(region[:TOP_K * 2]):
        if not isinstance(item, dict):
            lines.append(f"[{i}] {str(item)[:200]}")
            continue
        content = item.get("content") or item.get("label") or ""
        tendency_ids = item.get("tendency_ids") or item.get("tendency") or "?"
        lines.append(f"[{i}] tendency={tendency_ids}: {content[:300]}")
    return "\n".join(lines)


def build_user_prompt(test_entry: Dict[str, Any], context: str = "") -> str:
    prompt = (
        f"Implement this function:\n\n"
        f"def {test_entry['name']}({test_entry['signature']}):\n"
        f'    """{test_entry["sparse_docstring"]}"""\n'
        f"    # YOUR CODE HERE\n"
    )
    if context:
        prompt = context + "\n\n" + prompt
    prompt += "\nReturn ONLY the complete function definition. No prose, no markdown fences."
    return prompt


def strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


async def call_haiku(provider: BridgeProvider, user: str) -> Tuple[str, float]:
    started = time.time()
    try:
        result = await provider.send(
            messages=[{"role": "user", "content": user}],
            system=CONTESTANT_SYSTEM,
            model="haiku",
        )
        text = result.text or ""
    except Exception as e:
        text = f"BRIDGE_ERROR: {type(e).__name__}: {e}"
    return text, time.time() - started


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    test_entries = corpus["test"]
    train_by_name = {e["name"]: e for e in corpus["train"]}
    print(f"  N = {args.n}, test entries: {len(test_entries)}")

    embedder = default_usefulness_embedder(dim=EMBEDDING_DIM)
    world, train_index = build_world_for_n(corpus, args.n, embedder)
    n_nodes = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())
    print(f"  substrate built: {n_nodes} nodes")

    provider = BridgeProvider(model="haiku")
    rows: List[Dict[str, Any]] = []
    started = time.time()

    try:
        for i, t in enumerate(test_entries, start=1):
            row: Dict[str, Any] = {
                "n": args.n,
                "name": t["name"],
                "signature": t["signature"],
                "sparse_docstring": t["sparse_docstring"],
                "doctests": t["doctests"],
            }

            # a2 (haiku+RAG)
            top = topk_by_embedding(t, train_index, embedder)
            rag_ctx = format_rag_context(top, train_by_name)
            text_rag, dt_rag = await call_haiku(provider, build_user_prompt(t, rag_ctx))
            row["rag"] = {"impl": strip_code_fences(text_rag), "raw": text_rag,
                          "elapsed_s": dt_rag, "rag_top": top}

            # a3 (haiku+substrate)
            query_text = f"{t['name']} {t['sparse_docstring']}"
            try:
                probe = infer_with_world_model({"text": query_text}, mode="general")
            except Exception as e:
                probe = {"error": f"{type(e).__name__}: {e}", "region": []}
            substrate_ctx = format_substrate_context(probe)
            text_sub, dt_sub = await call_haiku(provider, build_user_prompt(t, substrate_ctx))
            row["substrate"] = {"impl": strip_code_fences(text_sub), "raw": text_sub,
                                "elapsed_s": dt_sub,
                                "probe_region_size": len(probe.get("region", [])) if isinstance(probe.get("region"), list) else 0}

            # Score immediately, write to progress log.
            for variant in ("rag", "substrate"):
                result = grade_implementation(
                    contestant_source=row[variant]["impl"],
                    fn_name=row["name"],
                    signature=row["signature"],
                    doctests=row["doctests"],
                )
                row[variant]["score"] = result["score"]
                row[variant]["passed"] = result["n_passed"]
                row[variant]["total"] = result["n_doctests"]
                row[variant]["compile_error"] = result["compile_error"]

            with PROGRESS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "n": args.n, "name": t["name"],
                    "rag_score": row["rag"]["score"],
                    "substrate_score": row["substrate"]["score"],
                    "ts": time.time(),
                }, ensure_ascii=False) + "\n")

            rows.append(row)
            elapsed = time.time() - started
            print(f"  [{i:>2}/{len(test_entries)}] {t['name']:>30}  "
                  f"rag={row['rag']['score']:.2f} sub={row['substrate']['score']:.2f}  "
                  f"({elapsed:.0f}s)")
    finally:
        try:
            await provider.close()
        except Exception:
            pass

    # Write per-N output.
    with Path(args.out).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rag_mean = sum(r["rag"]["score"] for r in rows) / max(len(rows), 1)
    sub_mean = sum(r["substrate"]["score"] for r in rows) / max(len(rows), 1)
    print()
    print(f"  N={args.n}  rag_mean={rag_mean:.3f}  substrate_mean={sub_mean:.3f}  "
          f"delta={sub_mean - rag_mean:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
