#!/usr/bin/env python3
"""Phase 5 substrate trainer.

Reads corpus.json, builds a 2-tendency world (correctness, simplicity),
posts each train function as an observation, calls sonnet via the
Claude Max bridge to extract multi-axis sub-claims, sprouts those
sub-claims as children of each function's observation node.

Output: trained_world.json (events + final node count for inspection).

Uses ScopedEquilibrate via the production WorldService path. NO API
keys; bridge auth via Claude Max subscription.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from atn.providers.bridge import BridgeProvider  # type: ignore  # noqa: E402


HERE = Path(__file__).resolve().parent
CORPUS_PATH = HERE / "corpus.json"
OUT_PATH = HERE / "trained_world.json"
JUDGE_CACHE_PATH = HERE / "judge_cache.jsonl"
STATUS_PATH = HERE / "status_train.json"


# Embedding tail dim. The 2-tendency charter head is 2 dims; with
# embedding dim=64 the total coord vector is 66 dims.
EMBEDDING_DIM = 64
CHARTER_DIM = 2
TOTAL_DIM = CHARTER_DIM + EMBEDDING_DIM
BANDWIDTH = 1.5


# --------------------------------------------------------------------------
# Substrate setup
# --------------------------------------------------------------------------


def build_world() -> World:
    """2-tendency world: correctness on axis 0, simplicity on axis 1."""
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
    return world


def coords_for_function(entry: Dict[str, Any], embedder) -> Tuple[float, ...]:
    """Embed the renamed function (name + docstring + impl) and return
    coords as [charter_head_zeros | embedding_tail]."""
    text = f"{entry['name']}\n\n{entry['docstring']}\n\n{entry.get('impl_full_source','')}"
    tail = embedder(text)
    tail_t = tuple(float(x) for x in tail)[:EMBEDDING_DIM]
    if len(tail_t) < EMBEDDING_DIM:
        tail_t = tail_t + (0.0,) * (EMBEDDING_DIM - len(tail_t))
    return (0.0, 0.0) + tail_t   # 2 charter axes start zeroed


def coords_for_subclaim(claim_text: str, axis_idx: int, embedder) -> Tuple[float, ...]:
    """Sub-claim coords: charter head has a +1 on the relevant axis,
    embedding tail is the embed of the claim text."""
    tail = embedder(claim_text)
    tail_t = tuple(float(x) for x in tail)[:EMBEDDING_DIM]
    if len(tail_t) < EMBEDDING_DIM:
        tail_t = tail_t + (0.0,) * (EMBEDDING_DIM - len(tail_t))
    head = tuple(1.0 if i == axis_idx else 0.0 for i in range(CHARTER_DIM))
    return head + tail_t


# --------------------------------------------------------------------------
# Sonnet judge call
# --------------------------------------------------------------------------


JUDGE_SYSTEM_PROMPT = """You are a code reviewer extracting structured sub-claims for a graph substrate.

You will be given one Python function (its name, signature, docstring, and implementation).

Your job: produce up to 4 sub-claims per axis on TWO axes:

  correctness: things that are TRUE about whether/how this function does what
               it claims. Examples: edge cases handled, invariants preserved,
               return types, what input shapes are accepted.

  simplicity: things that are TRUE about the structural shape of the
              implementation. Examples: use of higher-order functions,
              branching depth, idioms used, what could be simpler.

Each sub-claim is one short factual sentence about the function. Not
opinions. Not summaries of the docstring. Insightful observations a
reader of the source would make.

Respond with ONLY JSON of this exact shape (no markdown, no preamble):

{
  "correctness": ["claim 1 about correctness", "claim 2 about correctness", ...],
  "simplicity":  ["claim 1 about simplicity", "claim 2 about simplicity", ...]
}

Aim for 3-4 sub-claims per axis. Each must be substantive (15+ words).
"""


def judge_user_prompt(entry: Dict[str, Any]) -> str:
    return (
        f"Function name: {entry['name']}\n"
        f"Signature: {entry['signature']}\n\n"
        f"Docstring:\n{entry['docstring']}\n\n"
        f"Implementation:\n{entry.get('impl_full_source','(unavailable)')}\n\n"
        "Extract sub-claims now."
    )


def cache_key(entry: Dict[str, Any]) -> str:
    payload = json.dumps({
        "name": entry["name"],
        "doc": entry["docstring"],
        "impl": entry.get("impl_full_source", ""),
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_judge_cache() -> Dict[str, Dict[str, Any]]:
    if not JUDGE_CACHE_PATH.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for line in JUDGE_CACHE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            out[row["key"]] = row
        except json.JSONDecodeError:
            continue
    return out


def append_judge_cache(row: Dict[str, Any]) -> None:
    with JUDGE_CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_judge_response(text: str) -> Optional[Dict[str, List[str]]]:
    if not text:
        return None
    candidate = text.strip()
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        first = candidate.find("{")
        last = candidate.rfind("}")
        if first >= 0 and last > first:
            try:
                parsed = json.loads(candidate[first:last + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    if not isinstance(parsed, dict):
        return None
    out: Dict[str, List[str]] = {}
    for axis in ("correctness", "simplicity"):
        v = parsed.get(axis, [])
        if isinstance(v, list):
            cleaned = [str(x).strip() for x in v if str(x).strip()]
            out[axis] = cleaned[:4]   # cap at 4 per axis
        else:
            out[axis] = []
    return out


async def call_judge(provider: BridgeProvider, entry: Dict[str, Any]) -> Tuple[Optional[Dict[str, List[str]]], str]:
    try:
        result = await provider.send(
            messages=[{"role": "user", "content": judge_user_prompt(entry)}],
            system=JUDGE_SYSTEM_PROMPT,
            model="sonnet",
        )
    except Exception as e:
        return None, f"BRIDGE_ERROR: {type(e).__name__}: {e}"
    text = result.text or ""
    parsed = parse_judge_response(text)
    return parsed, text


# --------------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------------


def write_status(state: Dict[str, Any]) -> None:
    state["last_update"] = time.time()
    STATUS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("phase5_trainer")

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    train_entries = corpus["train"]
    print(f"  train entries: {len(train_entries)}")

    embedder = default_usefulness_embedder(dim=EMBEDDING_DIM)
    world = build_world()
    cache = load_judge_cache()
    print(f"  judge cache entries: {len(cache)}")

    provider = BridgeProvider(model="sonnet")
    started = time.time()
    n_calls = 0
    n_cache_hits = 0
    n_parse_fail = 0
    n_subclaims_total = 0

    state: Dict[str, Any] = {
        "phase": "training", "n_entries": len(train_entries),
        "current": 0, "n_calls": 0, "n_cache_hits": 0,
        "n_parse_fail": 0, "n_subclaims": 0,
    }
    write_status(state)

    try:
        for i, entry in enumerate(train_entries, start=1):
            state["current"] = i
            state["current_name"] = entry["name"]
            write_status(state)

            # 1. Observation for the function itself.
            obs_coords = coords_for_function(entry, embedder)
            obs = Observation(
                id=f"f_{entry['name']}",
                coords=obs_coords,
                label=entry["name"],
            )
            world.add_observation(obs)

            # Sprout the function as a child under the nearest tendency root
            # (both are equidistant since charter head is zeros; pick
            # correctness for canonical placement).
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

            # Run a scoped equilibrate (production-canonical path).
            scope = scope_for_observation(world, obs, slack=1.5)
            if scope:
                equilibrate(world, max_rounds=8, tolerance=1e-3, scope=scope)

            # 2. Judge call (sonnet) for sub-claims on both axes.
            key = cache_key(entry)
            if key in cache:
                parsed = cache[key].get("parsed")
                state["n_cache_hits"] += 1
                n_cache_hits += 1
            else:
                parsed, raw = await call_judge(provider, entry)
                state["n_calls"] += 1
                n_calls += 1
                if parsed is None:
                    state["n_parse_fail"] += 1
                    n_parse_fail += 1
                    log.warning(f"[{i}/{len(train_entries)}] {entry['name']}: judge parse failed; raw={raw[:200]!r}")
                else:
                    append_judge_cache({
                        "key": key,
                        "name": entry["name"],
                        "parsed": parsed,
                        "raw_text": raw,
                        "ts": time.time(),
                    })

            # 3. Sprout sub-claims as children of obs_node.
            n_subclaims_this = 0
            if parsed:
                for axis_idx, axis in enumerate(("correctness", "simplicity")):
                    tendency = world.tendencies[axis]
                    for claim_text in parsed.get(axis, []):
                        if not claim_text:
                            continue
                        claim_coords = coords_for_subclaim(claim_text, axis_idx, embedder)
                        try:
                            tendency.sprout_child(
                                parent_node_id=obs_node.id,
                                position=Position.PRO,
                                anchor=claim_coords,
                                polarity_axis=tendency.polarity_axis,
                                content=claim_text,
                                world=world,
                            )
                            n_subclaims_this += 1
                            n_subclaims_total += 1
                        except Exception as e:
                            log.warning(f"sprout failed for {entry['name']}/{axis}: {e}")

            # Equilibrate after sub-claim sprouts too.
            equilibrate(world, max_rounds=6, tolerance=1e-3)

            elapsed = time.time() - started
            print(f"  [{i:>2}/{len(train_entries)}] {entry['name']:>25}  "
                  f"+{n_subclaims_this} sub-claims  "
                  f"({elapsed:.0f}s elapsed)")
            state["n_subclaims"] = n_subclaims_total
            write_status(state)

    finally:
        try:
            await provider.close()
        except Exception:
            pass

    # Summary.
    n_nodes = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())
    work_items = 0
    for t in world.tendencies.values():
        for n in t.tree.all_nodes():
            if n.id == t.tree.root_node.id:
                continue
            if len({p.tendency_id for p in n.parents}) > 1:
                work_items += 1

    # Dump events + summary.
    events = []
    # Re-build event list from observations + sprouts is non-trivial; we
    # don't need full event log for grading, just the world structure.
    # For the contest we'll re-instantiate the world from corpus + cache.

    print()
    print("=" * 64)
    print("DONE")
    print("=" * 64)
    print(f"  train entries:      {len(train_entries)}")
    print(f"  sonnet calls made:  {n_calls}")
    print(f"  cache hits:         {n_cache_hits}")
    print(f"  parse failures:     {n_parse_fail}")
    print(f"  total sub-claims:   {n_subclaims_total}")
    print(f"  world nodes:        {n_nodes}")
    print(f"  work items (multi): {work_items}")
    print(f"  elapsed:            {time.time() - started:.1f}s")

    OUT_PATH.write_text(json.dumps({
        "n_train_entries": len(train_entries),
        "n_world_nodes": n_nodes,
        "n_work_items_multi_tendency": work_items,
        "n_subclaims_total": n_subclaims_total,
        "n_sonnet_calls": n_calls,
        "n_cache_hits": n_cache_hits,
        "n_parse_failures": n_parse_fail,
        "judge_cache_path": str(JUDGE_CACHE_PATH),
    }, indent=2), encoding="utf-8")
    state["phase"] = "complete"
    write_status(state)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
