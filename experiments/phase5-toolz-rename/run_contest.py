#!/usr/bin/env python3
"""Phase 5 contest runner.

Loads corpus.json (test entries with sparse docstrings), runs 4
contestants per test problem:

  a1 haiku-alone         (no retrieval, no substrate)
  a2 haiku + RAG         (top-k similar train functions by embedding similarity)
  a3 haiku + substrate   (production probe path)
  a4 qwen + substrate    (same as a3 with qwen renderer via Ollama)

Each contestant outputs a function implementation. The doctest harness
scores it. Output: contest_results.jsonl + scored_results.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

_AUTONET = Path(r"C:\code\autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))

from world_model.generalized import (  # type: ignore  # noqa: E402
    GeneralizedTendency, Observation, World, equilibrate,
)
from world_model.generalized.scope import scope_for_observation  # type: ignore  # noqa: E402
from world_model.models.tree import Position  # type: ignore  # noqa: E402

from nodes.common.world_model_substrate.usefulness_coords import (  # type: ignore  # noqa: E402
    default_usefulness_embedder, coords_for_query,
)
from nodes.common.world_model_substrate.infer import (  # type: ignore  # noqa: E402
    infer_with_world_model,
)
from atn.providers.bridge import BridgeProvider  # type: ignore  # noqa: E402

from doctest_harness import grade_implementation


HERE = Path(__file__).resolve().parent
CORPUS_PATH = HERE / "corpus.json"
JUDGE_CACHE_PATH = HERE / "judge_cache.jsonl"
OUT_PATH = HERE / "contest_results.jsonl"
SCORED_PATH = HERE / "scored_results.json"
STATUS_PATH = HERE / "status_contest.json"

EMBEDDING_DIM = 64
CHARTER_DIM = 2
TOTAL_DIM = CHARTER_DIM + EMBEDDING_DIM
BANDWIDTH = 1.5
TOP_K = 5

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
QWEN_MODEL = "qwen3.5:4b"


CONTESTANT_SYSTEM = """You are completing the body of a Python function.

You will be given:
  - A function signature with a sparse docstring containing 1-2 examples.
  - Optionally, related material to consult.

Your job: write ONLY the function body that satisfies the docstring's examples.

Rules:
  - Output ONLY the Python source for the function (def ...:\\n    body).
  - Do NOT include explanatory text, markdown, or commentary.
  - The function must be named exactly as the signature specifies.
  - Use only Python stdlib.
"""


def build_world(corpus: Dict[str, Any], embedder) -> Tuple[World, Dict[str, Any]]:
    """Re-build the trained world by replaying corpus + judge cache."""
    world = World()
    for i, tid in enumerate(("correctness", "simplicity")):
        anchor = tuple(1.0 if j == i else 0.0 for j in range(TOTAL_DIM))
        world.add_tendency(GeneralizedTendency(
            id=tid,
            thesis=f"Charter axis: {tid}",
            anchor=anchor,
            polarity_axis=anchor,
            budget=1.0,
            bandwidth=BANDWIDTH,
            smooth_promotion=True,
        ))

    # Load judge cache.
    cache: Dict[str, Dict[str, Any]] = {}
    if JUDGE_CACHE_PATH.exists():
        for line in JUDGE_CACHE_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cache[row["name"]] = row
            except (json.JSONDecodeError, KeyError):
                continue

    train = corpus["train"]
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in train:
        text = f"{entry['name']}\n\n{entry['docstring']}\n\n{entry.get('impl_full_source','')}"
        tail = embedder(text)
        tail_t = tuple(float(x) for x in tail)[:EMBEDDING_DIM]
        if len(tail_t) < EMBEDDING_DIM:
            tail_t = tail_t + (0.0,) * (EMBEDDING_DIM - len(tail_t))
        obs_coords = (0.0, 0.0) + tail_t
        obs = Observation(id=f"f_{entry['name']}", coords=obs_coords, label=entry["name"])
        world.add_observation(obs)

        sprouter = world.tendencies["correctness"]
        obs_node = sprouter.sprout_child(
            parent_node_id=sprouter.tree.root_node.id,
            position=Position.PRO,
            anchor=obs_coords,
            polarity_axis=sprouter.polarity_axis,
            observation=obs,
            content=entry["name"],
            world=world,
        )
        by_name[entry["name"]] = {"obs_node_id": obs_node.id, "embedding": tail_t}

        # Sub-claims from cache (if present).
        parsed = (cache.get(entry["name"]) or {}).get("parsed") or {}
        for axis_idx, axis in enumerate(("correctness", "simplicity")):
            tendency = world.tendencies[axis]
            for claim_text in parsed.get(axis, []):
                if not claim_text:
                    continue
                claim_tail = embedder(claim_text)
                claim_tail_t = tuple(float(x) for x in claim_tail)[:EMBEDDING_DIM]
                if len(claim_tail_t) < EMBEDDING_DIM:
                    claim_tail_t = claim_tail_t + (0.0,) * (EMBEDDING_DIM - len(claim_tail_t))
                head = tuple(1.0 if i == axis_idx else 0.0 for i in range(CHARTER_DIM))
                claim_coords = head + claim_tail_t
                try:
                    tendency.sprout_child(
                        parent_node_id=obs_node.id,
                        position=Position.PRO,
                        anchor=claim_coords,
                        polarity_axis=tendency.polarity_axis,
                        content=claim_text,
                        world=world,
                    )
                except Exception:
                    pass

    return world, by_name


def topk_by_embedding(test_entry, train_index, embedder, k=TOP_K) -> List[Dict[str, Any]]:
    """RAG: cosine-similarity top-k train entries to the test sparse prompt."""
    q_tail = embedder(f"{test_entry['name']}\n\n{test_entry['sparse_docstring']}")
    q_arr = np.array(q_tail, dtype=np.float32)
    q_norm = np.linalg.norm(q_arr) + 1e-9
    scored: List[Tuple[float, str]] = []
    for name, idx in train_index.items():
        e_arr = np.array(idx["embedding"], dtype=np.float32)
        cos = float(np.dot(q_arr, e_arr) / (q_norm * (np.linalg.norm(e_arr) + 1e-9)))
        scored.append((cos, name))
    scored.sort(reverse=True)
    top = scored[:k]
    return [{"name": n, "similarity": s} for s, n in top]


def format_rag_context(top: List[Dict[str, Any]], train_by_name: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = ["RELATED FUNCTIONS (most similar to the prompt):", ""]
    for item in top:
        e = train_by_name[item["name"]]
        lines.append(f"--- {e['name']} (signature: {e['signature']}) ---")
        lines.append(e["docstring"])
        lines.append("")
        lines.append(e.get("impl_full_source", ""))
        lines.append("")
    return "\n".join(lines)


def format_substrate_context(probe_result: Dict[str, Any], world: World) -> str:
    """Format `infer_with_world_model(mode='general')` output for the
    contestant. The probe returns a rendered region structure; we
    surface node content + parent tendency labels."""
    lines: List[str] = ["GRAPH REGION (located by substrate probe):", ""]
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


def build_contestant_user_prompt(test_entry: Dict[str, Any], context: str = "") -> str:
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


# --------------------------------------------------------------------------
# LLM callers
# --------------------------------------------------------------------------


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


def call_qwen(user: str) -> Tuple[str, float]:
    payload = {
        "model": QWEN_MODEL,
        "system": CONTESTANT_SYSTEM,
        "prompt": user,
        "stream": True,
        "options": {"temperature": 0.2, "num_predict": 1500},
    }
    started = time.time()
    parts: List[str] = []
    try:
        req = urllib.request.Request(
            OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "response" in obj and obj["response"]:
                    parts.append(obj["response"])
                if obj.get("done"):
                    break
    except Exception as e:
        parts.append(f"OLLAMA_ERROR: {type(e).__name__}: {e}")
    return "".join(parts), time.time() - started


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def write_status(state: Dict[str, Any]) -> None:
    state["last_update"] = time.time()
    STATUS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    test_entries = corpus["test"]
    train_entries = corpus["train"]
    train_by_name = {e["name"]: e for e in train_entries}

    print(f"  test entries: {len(test_entries)}")
    embedder = default_usefulness_embedder(dim=EMBEDDING_DIM)

    # Re-build the trained world deterministically from corpus + judge cache.
    world, train_index = build_world(corpus, embedder)
    n_nodes = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())
    print(f"  rebuilt world: {n_nodes} nodes")

    provider = BridgeProvider(model="haiku")
    started = time.time()
    rows: List[Dict[str, Any]] = []

    state: Dict[str, Any] = {
        "phase": "contest", "n_test": len(test_entries),
        "current": 0,
    }
    write_status(state)

    try:
        for i, t in enumerate(test_entries, start=1):
            state["current"] = i
            state["current_name"] = t["name"]
            write_status(state)
            row: Dict[str, Any] = {
                "name": t["name"],
                "orig_name": t["orig_name"],
                "signature": t["signature"],
                "sparse_docstring": t["sparse_docstring"],
                "doctests": t["doctests"],
            }

            # a1: haiku alone
            user_a1 = build_contestant_user_prompt(t)
            text_a1, dt_a1 = await call_haiku(provider, user_a1)
            row["a1"] = {"impl": strip_code_fences(text_a1), "raw": text_a1, "elapsed_s": dt_a1}

            # a2: haiku + RAG (top-k from training corpus by embedding similarity)
            top = topk_by_embedding(t, train_index, embedder, k=TOP_K)
            rag_ctx = format_rag_context(top, train_by_name)
            user_a2 = build_contestant_user_prompt(t, context=rag_ctx)
            text_a2, dt_a2 = await call_haiku(provider, user_a2)
            row["a2"] = {"impl": strip_code_fences(text_a2), "raw": text_a2,
                         "elapsed_s": dt_a2, "rag_top": top}

            # a3: haiku + substrate (production probe)
            query_text = f"{t['name']} {t['sparse_docstring']}"
            try:
                probe = infer_with_world_model(
                    {"text": query_text}, mode="general",
                )
            except Exception as e:
                probe = {"error": f"{type(e).__name__}: {e}", "region": []}
            substrate_ctx = format_substrate_context(probe, world)
            user_a3 = build_contestant_user_prompt(t, context=substrate_ctx)
            text_a3, dt_a3 = await call_haiku(provider, user_a3)
            row["a3"] = {
                "impl": strip_code_fences(text_a3), "raw": text_a3,
                "elapsed_s": dt_a3,
                "probe_region_size": len(probe.get("region", [])) if isinstance(probe.get("region"), list) else 0,
            }

            # a4: qwen + substrate
            text_a4, dt_a4 = call_qwen(user_a3)
            row["a4"] = {"impl": strip_code_fences(text_a4), "raw": text_a4,
                         "elapsed_s": dt_a4,
                         "probe_region_size": row["a3"]["probe_region_size"]}

            rows.append(row)
            elapsed = time.time() - started
            print(f"  [{i:>2}/{len(test_entries)}] {t['name']:>22}  "
                  f"a1={dt_a1:.0f}s a2={dt_a2:.0f}s a3={dt_a3:.0f}s a4={dt_a4:.0f}s  "
                  f"({elapsed:.0f}s elapsed)")
    finally:
        try:
            await provider.close()
        except Exception:
            pass

    # Write raw answers.
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Grade each row.
    print()
    print("Grading...")
    contestants = ("a1", "a2", "a3", "a4")
    scores: Dict[str, List[float]] = {c: [] for c in contestants}
    per_row: List[Dict[str, Any]] = []
    for row in rows:
        graded = {"name": row["name"]}
        for c in contestants:
            result = grade_implementation(
                contestant_source=row[c]["impl"],
                fn_name=row["name"],
                signature=row["signature"],
                doctests=row["doctests"],
            )
            scores[c].append(result["score"])
            graded[c] = {
                "score": result["score"],
                "passed": result["n_passed"],
                "total": result["n_doctests"],
                "compile_error": result["compile_error"],
            }
        per_row.append(graded)
        print(f"  {row['name']:>22}: "
              f"a1={graded['a1']['score']:.2f} "
              f"a2={graded['a2']['score']:.2f} "
              f"a3={graded['a3']['score']:.2f} "
              f"a4={graded['a4']['score']:.2f}")

    means = {c: sum(scores[c]) / max(len(scores[c]), 1) for c in contestants}
    pass1 = {c: sum(1 for s in scores[c] if s == 1.0) for c in contestants}

    print()
    print("=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"  {'contestant':<22} {'mean score':>10}  {'pass@1':>8}")
    for c, label in (("a1", "haiku alone"), ("a2", "haiku + RAG"),
                      ("a3", "haiku + substrate"), ("a4", "qwen + substrate")):
        print(f"  {label:<22} {means[c]:>10.3f}  {pass1[c]:>3d}/{len(rows):<2}")

    print()
    print("Q1 (architecture vs RAG):")
    q1 = means["a3"] - means["a2"]
    print(f"  haiku+substrate - haiku+RAG = {q1:+.3f}")
    print(f"  -> {'substrate beats RAG' if q1 > 0.05 else 'RAG beats substrate' if q1 < -0.05 else 'tie'}")

    print()
    print("Q2 (stack viability):")
    q2 = means["a4"] - means["a1"]
    print(f"  qwen+substrate - haiku-alone = {q2:+.3f}")
    print(f"  -> {'stack viable' if q2 > 0.05 else 'stack not viable' if q2 < -0.05 else 'tie'}")

    SCORED_PATH.write_text(json.dumps({
        "means": means,
        "pass_at_1": pass1,
        "n_test": len(rows),
        "Q1_substrate_minus_rag": q1,
        "Q2_qwen_substrate_minus_haiku_alone": q2,
        "per_row": per_row,
    }, indent=2), encoding="utf-8")
    print(f"\n  results -> {SCORED_PATH}")

    state["phase"] = "complete"
    write_status(state)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
