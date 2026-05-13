#!/usr/bin/env python3
"""Substrate retraining with judge-agent sub-claims.

Replays work units into the world-model substrate (one leaf per unit
under the best-matching root tendency, mirroring train_world_model_on_usefulness),
then for each unit calls a frontier LLM via the Claude Max bridge to
extract 1-2 procedural sub-claims and posts those as CHILDREN of that
leaf. The result: a deeper graph that carries judge-agent commentary
on WHY each work unit landed the way it did.

Auth
----

Uses the BridgeProvider (atn.providers.bridge) which talks to Claude
via the user's Claude Max subscription through a TS subprocess. NO
API keys are involved. One bridge subprocess is shared across all
calls (the system prompt is identical, so it gets prompt-cached after
the first call).

Usage
-----

    python train_with_judges.py \\
        --units work_units_filtered.jsonl \\
        --limit 50 \\
        --model haiku \\
        --out trained_world_with_judges.json \\
        --status status_with_judges.json \\
        --judge-cache judge_cache.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make autonet substrate + bridge provider importable
_AUTONET = Path(r"C:\code\autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))


# ---------------------------------------------------------------------------
# Imports from autonet / world-model
# ---------------------------------------------------------------------------

from world_model.generalized import (  # type: ignore  # noqa: E402
    Observation,
    World,
    equilibrate,
)
from world_model.models.tree import Position  # type: ignore  # noqa: E402

from nodes.common.world_model_substrate.events import (  # type: ignore  # noqa: E402
    ObservationAdded,
    SubClaimSprouted,
    snapshot_node_scores,
)
from nodes.common.world_model_substrate.outcomes import Outcome  # type: ignore  # noqa: E402
from nodes.common.world_model_substrate.usefulness_coords import (  # type: ignore  # noqa: E402
    DEFAULT_DIM,
    default_usefulness_embedder,
    _l2_normalize,
)
from nodes.common.world_model_substrate.usefulness_training import (  # type: ignore  # noqa: E402
    _obs_id,
    build_usefulness_world,
    work_unit_to_observation,
)
from nodes.common.world_model_substrate.aggregate import apply_events  # type: ignore  # noqa: E402

# Bridge provider — Claude Max auth, NO API keys
from atn.providers.bridge import BridgeProvider  # type: ignore  # noqa: E402


log = logging.getLogger("train_with_judges")


# ---------------------------------------------------------------------------
# Judge prompts (system prompt is constant across all calls so the bridge
# can prompt-cache it after the first request).
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a judge agent extracting procedural knowledge from completed software-engineering work.
Given a (problem, resolution, outcome) tuple, produce 1-2 short sub-claims that capture WHY this work landed the way it did, as procedural insights. Each sub-claim:
- A short principle/pattern statement (<= 25 words)
- A stance: PRO if the principle helped this resolution land well; CON if its absence/violation contributed to it not landing.

Respond with ONLY valid JSON in this exact shape, no markdown fences, no preamble:
{"sub_claims": [{"content": "...", "stance": "PRO"}, {"content": "...", "stance": "CON"}]}

If only one sub-claim is warranted, return a single-element list. Always return at least one."""


def build_user_prompt(problem: str, resolution: str, outcome: Outcome) -> str:
    return (
        "PROBLEM:\n"
        f"{problem[:600]}\n\n"
        "RESOLUTION:\n"
        f"{resolution[:1200]}\n\n"
        "OUTCOME (each \u2208 [-1, 1]):\n"
        f"accepted={outcome.accepted:.1f} kept={outcome.kept:.1f} "
        f"built_on={outcome.built_on:.1f} paid={outcome.paid:.1f}\n\n"
        "Extract sub-claims now."
    )


# ---------------------------------------------------------------------------
# Caching: judge responses keyed by (problem, resolution) prefix hash
# ---------------------------------------------------------------------------


def cache_key(problem: str, resolution: str) -> str:
    payload = (problem[:400] + "|||" + resolution[:800]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_judge_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load cache from JSONL on disk into {key: entry} dict."""
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    k = rec.get("key")
                    if k:
                        out[k] = rec
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log.warning("failed to load judge cache %s: %s", path, e)
    return out


def append_judge_cache(path: Path, entry: Dict[str, Any]) -> None:
    """Append a single entry to the JSONL cache file."""
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("failed to append judge cache %s: %s", path, e)


# ---------------------------------------------------------------------------
# JSON parsing robustness — model may emit fences / preamble despite
# system prompt instructions.
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_judge_json(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse the judge's response into a list of sub-claim dicts.

    Tolerant of: code fences, preamble before/after the JSON object,
    `{...}` embedded inside other text. Returns None on failure.
    """
    if not text:
        return None

    candidate = text.strip()

    # Strip markdown fences if present
    fence_match = _FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()

    # Try direct parse first
    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Find first { and last } and try slice
        first = candidate.find("{")
        last = candidate.rfind("}")
        if first >= 0 and last > first:
            slice_ = candidate[first:last + 1]
            try:
                parsed = json.loads(slice_)
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        return None

    sub_claims = parsed.get("sub_claims")
    if not isinstance(sub_claims, list):
        return None

    cleaned: List[Dict[str, Any]] = []
    for sc in sub_claims:
        if not isinstance(sc, dict):
            continue
        content = sc.get("content")
        stance = sc.get("stance", "PRO")
        if not isinstance(content, str) or not content.strip():
            continue
        if stance not in ("PRO", "CON"):
            stance = "PRO"
        cleaned.append({"content": content.strip(), "stance": stance})

    if not cleaned:
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Bridge call wrapper
# ---------------------------------------------------------------------------


async def call_bridge_for_sub_claims(
    provider: BridgeProvider,
    problem: str,
    resolution: str,
    outcome: Outcome,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Call the bridge once for sub-claim extraction.

    Returns (parsed_sub_claims_or_None, raw_response_text).
    """
    user_prompt = build_user_prompt(problem, resolution, outcome)
    try:
        result = await provider.send(
            messages=[{"role": "user", "content": user_prompt}],
            system=JUDGE_SYSTEM_PROMPT,
        )
    except Exception as e:
        log.warning("bridge.send() failed: %s", e)
        return None, ""

    text = result.text or ""
    parsed = parse_judge_json(text)
    return parsed, text


# ---------------------------------------------------------------------------
# Work unit loading (mirror run_experiment.py)
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
# Substrate helpers
# ---------------------------------------------------------------------------


def _best_root(world: World, coords: Tuple[float, ...]):
    """Pick the root tendency whose anchor has the highest cosine
    similarity to the given coords. Mirrors usefulness_training._best_root.
    """
    best = None
    best_dot = -2.0
    for tendency in world.tendencies.values():
        if not tendency.anchor:
            continue
        dot = sum(a * b for a, b in zip(coords, tendency.anchor))
        if dot > best_dot:
            best_dot = dot
            best = tendency
    return best


def _all_node_count(world: World) -> int:
    return sum(len(t.tree.all_nodes()) for t in world.tendencies.values())


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    started_at = time.time()

    units_path = Path(args.units)
    if not units_path.exists():
        print(f"  ERROR: units file not found: {units_path}", file=sys.stderr)
        return 2

    all_units = load_work_units(units_path)
    units = all_units[: args.limit] if args.limit > 0 else all_units
    n_total = len(units)
    print(f"  loaded {len(all_units)} work units; processing first {n_total}")

    # Cost guard print (no confirmation prompt — --limit is the lever)
    print(
        f"\n  This will make ~{n_total} bridge calls to Claude (model={args.model}) "
        f"against your Claude Max subscription."
    )
    print(
        f"  Each call: ~600 input tokens (cached after first), ~80 output tokens."
    )
    print(
        f"  Limit: --limit {args.limit}. To bound usage, lower --limit.\n"
    )

    # Load judge cache
    cache_path = Path(args.judge_cache)
    cache = load_judge_cache(cache_path)
    if cache:
        print(f"  loaded {len(cache)} judge cache entries from {cache_path}")
    else:
        print(f"  no existing judge cache (will write to {cache_path})")

    # Status writer
    status_path = Path(args.status)
    status: Dict[str, Any] = {
        "started_at": started_at,
        "phase": "training",
        "n_units_total": n_total,
        "current_unit": 0,
        "llm_calls_made": 0,
        "llm_cache_hits": 0,
        "llm_parse_failures": 0,
        "n_world_nodes": 0,
        "last_update": started_at,
    }

    def write_status() -> None:
        try:
            status["last_update"] = time.time()
            with status_path.open("w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            log.debug("status write failed: %s", e)

    write_status()

    # Build embedder + world
    dim = args.dim
    embedder = default_usefulness_embedder(dim=dim)
    world = build_usefulness_world(dim=dim, bandwidth=args.bandwidth)

    events: List[Any] = []
    seq = 0
    agent_id = "judge-trainer"

    # Bridge — single subprocess, many sessions
    provider = BridgeProvider(model=args.model)

    try:
        for i, unit in enumerate(units, start=1):
            status["current_unit"] = i
            write_status()

            problem = unit.get("problem", "") or ""
            resolution = unit.get("resolution", "") or ""
            raw_outcome = unit.get("outcome")
            if raw_outcome:
                try:
                    outcome = Outcome(*raw_outcome)
                except TypeError:
                    outcome = Outcome()
            else:
                outcome = Outcome()

            # ---- Observation ----
            obs = work_unit_to_observation(problem, resolution, outcome, embedder)
            world.add_observation(obs)
            seq += 1
            events.append(ObservationAdded(
                seq=seq,
                author_agent=agent_id,
                obs_id=obs.id,
                coords=list(obs.coords),
                label=obs.label,
            ))

            # ---- Leaf sprout (mirror train_world_model_on_usefulness) ----
            target = _best_root(world, obs.coords)
            if target is None:
                log.warning("unit %d: no root tendency matched; skipping", i)
                continue

            pos_signal = outcome.accepted + outcome.kept
            leaf_position = Position.PRO if pos_signal >= 0 else Position.CON

            axis_list = list(obs.coords)
            if any(c != 0 for c in axis_list):
                axis = _l2_normalize(axis_list)
            else:
                axis = target.polarity_axis

            leaf = target.sprout_child(
                parent_node_id=target.tree.root_node.id,
                position=leaf_position,
                anchor=obs.coords,
                polarity_axis=tuple(axis),
                observation=obs,
                content=problem[:80],
            )
            seq += 1
            events.append(SubClaimSprouted(
                seq=seq,
                author_agent=agent_id,
                tendency_id=target.id,
                parent_id=target.tree.root_node.id,
                node_id=leaf.id,
                position=leaf.position.value,
                coords=list(obs.coords),
                polarity_axis=list(axis),
                content=problem[:80],
                observation_id=obs.id,
            ))

            # ---- Judge call → 1-2 sub-claims ----
            key = cache_key(problem, resolution)
            sub_claims: Optional[List[Dict[str, Any]]] = None
            cached = cache.get(key)
            if cached and isinstance(cached.get("sub_claims_parsed"), list):
                sub_claims = cached["sub_claims_parsed"]
                status["llm_cache_hits"] += 1
            else:
                parsed, raw_text = await call_bridge_for_sub_claims(
                    provider, problem, resolution, outcome,
                )
                status["llm_calls_made"] += 1
                if parsed is None:
                    status["llm_parse_failures"] += 1
                    log.warning(
                        "unit %d: judge parse failed; raw=%r (skipping sub-claims for this unit)",
                        i, (raw_text[:200] if raw_text else "<empty>"),
                    )
                else:
                    sub_claims = parsed
                    entry = {
                        "key": key,
                        "response_text": raw_text,
                        "sub_claims_parsed": parsed,
                        "model": args.model,
                    }
                    cache[key] = entry
                    append_judge_cache(cache_path, entry)

            # ---- Sub-claim sprouts (children of the leaf) ----
            if sub_claims:
                for sc_idx, sc in enumerate(sub_claims[:2]):
                    # Deterministic small offset per index so re-runs are
                    # stable and the two sub-claims have distinct anchors.
                    target_dim = sc_idx % dim
                    sign = (sc_idx % 2) * 2 - 1  # 0 -> -1, 1 -> +1
                    offset = [
                        0.05 * sign if (j == target_dim) else 0.0
                        for j in range(dim)
                    ]
                    sc_coords = tuple(
                        o + d for o, d in zip(obs.coords, offset)
                    )
                    sc_axis_list = list(sc_coords)
                    if any(c != 0 for c in sc_axis_list):
                        sc_axis = _l2_normalize(sc_axis_list)
                    else:
                        sc_axis = axis

                    stance = sc.get("stance", "PRO")
                    sc_position = Position.PRO if stance == "PRO" else Position.CON
                    sc_content = (sc.get("content") or "")[:200]

                    sc_node = target.sprout_child(
                        parent_node_id=leaf.id,
                        position=sc_position,
                        anchor=sc_coords,
                        polarity_axis=tuple(sc_axis),
                        observation=obs,
                        content=sc_content,
                    )
                    seq += 1
                    events.append(SubClaimSprouted(
                        seq=seq,
                        author_agent=agent_id,
                        tendency_id=target.id,
                        parent_id=leaf.id,
                        node_id=sc_node.id,
                        position=sc_node.position.value,
                        coords=list(sc_coords),
                        polarity_axis=list(sc_axis),
                        content=sc_content,
                        observation_id=obs.id,
                    ))

            # NOTE: do NOT equilibrate per-unit here. apply_events runs
            # equilibrate at the end during validation/replay, so equilibrating
            # here would diverge the trained world from the replayable one.

            status["n_world_nodes"] = _all_node_count(world)
            write_status()

            if i % 5 == 0 or i == n_total:
                print(
                    f"  [{i}/{n_total}] nodes={status['n_world_nodes']} "
                    f"calls={status['llm_calls_made']} "
                    f"hits={status['llm_cache_hits']} "
                    f"parse_fail={status['llm_parse_failures']}"
                )

        world.clear_observations()

    finally:
        try:
            await provider.close()
        except Exception as e:
            log.warning("provider.close() failed: %s", e)

    # ---- Final snapshot + validation ----
    score_snapshot = snapshot_node_scores(world)
    n_world_nodes = _all_node_count(world)
    elapsed = time.time() - started_at

    event_dicts = [e.to_dict() for e in events]

    # Validate by replaying events on a fresh world.
    fresh = build_usefulness_world(dim=dim, bandwidth=args.bandwidth)
    apply_events(fresh, event_dicts)
    fresh_node_count = _all_node_count(fresh)
    validation_ok = fresh_node_count == n_world_nodes
    if not validation_ok:
        log.warning(
            "validation: replayed world has %d nodes vs trained world %d",
            fresh_node_count, n_world_nodes,
        )

    # ---- Output ----
    out_payload: Dict[str, Any] = {
        "events": event_dicts,
        "score_snapshot_after": score_snapshot,
        "n_work_units": n_total,
        "n_events": len(event_dicts),
        "n_world_nodes": n_world_nodes,
        "n_world_nodes_replayed": fresh_node_count,
        "validation_ok": validation_ok,
        "model_used": args.model,
        "elapsed_seconds": elapsed,
        "llm_calls_made": status["llm_calls_made"],
        "llm_cache_hits": status["llm_cache_hits"],
        "llm_parse_failures": status["llm_parse_failures"],
        "agent_id": agent_id,
        "args": {
            "units": str(units_path),
            "limit": args.limit,
            "model": args.model,
            "dim": dim,
            "bandwidth": args.bandwidth,
            "judge_cache": str(cache_path),
        },
    }

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    status["phase"] = "complete"
    status["n_world_nodes"] = n_world_nodes
    write_status()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  events:           {len(event_dicts)}")
    print(f"  world nodes:      {n_world_nodes}")
    print(f"  replay validated: {validation_ok} (replayed -> {fresh_node_count} nodes)")
    print(f"  llm calls:        {status['llm_calls_made']}")
    print(f"  cache hits:       {status['llm_cache_hits']}")
    print(f"  parse failures:   {status['llm_parse_failures']}")
    print(f"  elapsed:          {elapsed:.1f}s")
    print(f"  output:           {out_path}")
    print(f"  status:           {status_path}")
    print(f"  judge cache:      {cache_path}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the substrate with judge-agent sub-claims posted under "
            "each work unit's leaf, using the Claude Max bridge (NO API keys)."
        )
    )
    parser.add_argument(
        "--units",
        default="work_units_filtered.jsonl",
        help="JSONL of work units (default: work_units_filtered.jsonl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max work units to process (default: 50). The cost lever.",
    )
    parser.add_argument(
        "--model",
        default="haiku",
        choices=["haiku", "sonnet", "opus"],
        help="Bridge model class (default: haiku — cheaper for structured extraction).",
    )
    parser.add_argument(
        "--out",
        default="trained_world_with_judges.json",
        help="Path to write the events + score snapshot JSON.",
    )
    parser.add_argument(
        "--status",
        default="status_with_judges.json",
        help="Live status file, updated after each unit.",
    )
    parser.add_argument(
        "--judge-cache",
        default="judge_cache.jsonl",
        help="JSONL cache for judge responses (keyed by problem+resolution hash).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        help=f"Embedding/coords dim (default: {DEFAULT_DIM}).",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=0.5,
        help="Tendency bandwidth (default: 0.5).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Python logging level (default: WARNING).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
