#!/usr/bin/env python3
"""Three-contestant runner for the substrate-vs-frontier reasoning contest.

For a fixed bounded-domain question set (the autonet world-model substrate
codebase + the generalized world-model engine), produce three answers per
question:

  A1 -- Frontier-LLM-alone (Haiku, no context). Tests what the model knows
        without help.
  A2 -- Frontier-LLM-with-codebase (Sonnet, full code dump in context).
        Tests retrieval-style answering with the entire codebase as one
        flat blob.
  A3 -- Substrate + small LLM (Haiku + locate+render). Tests whether the
        trained graph substrate, surfaced via locate(), gives a small
        model enough material to answer well.

The output is a JSONL file with one row per question, including the three
answers, elapsed time per call, and a region summary for A3 (for later
debugging). An oracle blind-grades the rows in a separate step.

Usage
-----

    python run_contest.py \\
        --questions questions.jsonl \\
        --trained-world trained_world_with_judges.json \\
        --units work_units_filtered.jsonl \\
        --out contest_results.jsonl \\
        --status status_contest.json \\
        --limit 30 \\
        --top-k 12

NO API keys are involved. Bridge subprocesses talk to Claude through the
user's Claude Max subscription via two TS subprocesses (one per model).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make autonet substrate + bridge provider importable. Mirror the import
# pattern from train_with_judges.py exactly.
_AUTONET = Path(r"C:\code\autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))


# ---------------------------------------------------------------------------
# Imports from autonet / world-model
# ---------------------------------------------------------------------------

from world_model.generalized import (  # type: ignore  # noqa: E402
    ChainLocator,
    CoordinateLocator,
    KeywordLocator,
    World,
)

from nodes.common.world_model_substrate.aggregate import apply_events  # type: ignore  # noqa: E402
from nodes.common.world_model_substrate.usefulness_coords import (  # type: ignore  # noqa: E402
    DEFAULT_DIM,
    coords_for_query,
    default_usefulness_embedder,
)
from nodes.common.world_model_substrate.usefulness_training import (  # type: ignore  # noqa: E402
    build_usefulness_world,
)

# Bridge provider -- Claude Max auth, NO API keys.
from atn.providers.bridge import BridgeProvider  # type: ignore  # noqa: E402


log = logging.getLogger("run_contest")


# ---------------------------------------------------------------------------
# System prompts (constant across every call -- prompt-cached on the bridge)
# ---------------------------------------------------------------------------

A1_SYSTEM_PROMPT = (
    "You are answering a technical question about a software codebase. "
    "Answer based only on what you know. If you're not sure, say so. "
    "Be concrete; cite specific module/function names if you can recall "
    "them. Keep answers under 150 words."
)

A2_SYSTEM_PROMPT = (
    "You are answering a technical question about a software codebase. "
    "The full codebase is provided in your context. Answer specifically "
    "with reference to the actual code. Cite file names and line numbers. "
    "Keep answers under 150 words."
)

A3_SYSTEM_PROMPT = (
    "You are answering a technical question by synthesizing the provided "
    "graph regions. Each region is a snippet of related past work and "
    "procedural insights extracted from the codebase. Use ONLY the graph "
    "regions to formulate your answer. If the regions don't contain "
    "enough information, say so. Cite specific node content. Keep under "
    "150 words."
)


# ---------------------------------------------------------------------------
# Codebase dump -- read all .py files in two trees, plus STATUS.md.
# ---------------------------------------------------------------------------

# Roots that get included in the codebase dump for A2.
_DUMP_ROOTS: List[Tuple[Path, str]] = [
    # (directory, label_prefix shown in the --- FILE: header)
    (Path(r"C:\code\autonet\nodes\common\world_model_substrate"),
     "autonet/nodes/common/world_model_substrate"),
    (Path(r"C:\code\world-model\world_model\generalized"),
     "world-model/world_model/generalized"),
]

# Extra single files (relative to one of the dump roots).
_EXTRA_FILES: List[Tuple[Path, str]] = [
    (Path(r"C:\code\world-model\world_model\generalized\STATUS.md"),
     "world-model/world_model/generalized/STATUS.md"),
]


def _iter_py_files(root: Path) -> List[Path]:
    """Return .py files under root, skipping __pycache__ dirs. Sorted for determinism."""
    out: List[Path] = []
    if not root.exists():
        return out
    for path in root.rglob("*.py"):
        # Skip anything inside a __pycache__ directory.
        if "__pycache__" in path.parts:
            continue
        out.append(path)
    out.sort()
    return out


def build_codebase_dump() -> str:
    """Concatenate all relevant .py files (plus STATUS.md) into a single
    string, each preceded by a `--- FILE: <relative_path> ---` header.
    """
    parts: List[str] = []
    for root, label_prefix in _DUMP_ROOTS:
        for path in _iter_py_files(root):
            rel = path.relative_to(root)
            label = f"{label_prefix}/{rel.as_posix()}"
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as e:
                log.warning("failed to read %s: %s", path, e)
                continue
            parts.append(f"--- FILE: {label} ---\n{content}\n")
    for path, label in _EXTRA_FILES:
        if not path.exists():
            log.warning("extra file missing: %s", path)
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            log.warning("failed to read %s: %s", path, e)
            continue
        parts.append(f"--- FILE: {label} ---\n{content}\n")
    return "\n".join(parts)


def load_or_build_codebase_dump(cache_path: Path, no_cache: bool) -> str:
    """Read cached codebase dump if present, otherwise build & cache it.

    The dump is identical for every question in a run (and across runs as
    long as the underlying source hasn't changed), so caching to disk
    means re-runs don't have to re-read 30+ files.
    """
    if not no_cache and cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception as e:
            log.warning("failed to read codebase cache %s: %s", cache_path, e)
    dump = build_codebase_dump()
    try:
        cache_path.write_text(dump, encoding="utf-8")
    except Exception as e:
        log.warning("failed to write codebase cache %s: %s", cache_path, e)
    return dump


# ---------------------------------------------------------------------------
# Question loader
# ---------------------------------------------------------------------------


def load_questions(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("skipping malformed question line: %s", e)
                continue
    return out


# ---------------------------------------------------------------------------
# Trained world loader
# ---------------------------------------------------------------------------


def load_trained_world(path: Path, dim: int, bandwidth: float) -> World:
    """Load the trained world from the events-snapshot JSON produced by
    train_with_judges.py.

    The JSON is shaped {"events": [...], "score_snapshot_after": ...}. The
    events list is a list of dicts (Event.to_dict()) which apply_events
    accepts directly.
    """
    with path.open("r", encoding="utf-8") as f:
        contribution = json.load(f)
    events = contribution.get("events", [])
    if not isinstance(events, list):
        raise ValueError(
            f"trained-world JSON {path} has no 'events' list (got {type(events).__name__})"
        )
    world = build_usefulness_world(dim=dim, bandwidth=bandwidth)
    apply_events(world, events)
    return world


# ---------------------------------------------------------------------------
# A3: format the graph-context prompt from a region
# ---------------------------------------------------------------------------


def _truncate(text: str, n: int) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "..."


def format_graph_context(
    world: World,
    region: List[Tuple[str, str, float]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Render a region into a structured "graph context" prompt.

    For each region member, look up its node and parent in the world,
    then emit a stanza of the form:

        [tendency=<tid> dist=<d> <POS>]
          parent: <parent content>
          node:   <node content>

    Returns (prompt_text, region_summary). The summary is a list of
    dicts saved into the output JSON (for later debugging, NOT for the
    oracle's grading).
    """
    lines: List[str] = ["Graph regions located near the question (top-{n} by relevance):".format(n=len(region)), ""]
    summary: List[Dict[str, Any]] = []
    for tendency_id, node_id, distance in region:
        tendency = world.tendencies.get(tendency_id)
        if tendency is None:
            continue
        node = tendency.tree.get_node(node_id)
        if node is None:
            continue

        # Tendency root thesis (e.g. "good_resolution" or "novel_resolution").
        thesis = tendency.thesis or ""

        # Parent content, if any.
        parent_content = ""
        if node.parent_id:
            parent_node = tendency.tree.get_node(node.parent_id)
            if parent_node is not None:
                # Don't surface the synthetic root_value content -- it's just
                # the tendency thesis again. Prefer empty over redundant.
                if parent_node.parent_id is not None:
                    parent_content = parent_node.content or ""
                else:
                    parent_content = "(tendency root)"

        node_content = node.content or ""
        position = (node.position.value or "").upper()

        # Descend into the node's children -- the judge sub-claim layer where
        # the actual procedural insights live. Without this, the renderer only
        # surfaces the leaf problem-snippet and Haiku has nothing to reason from.
        child_lines: List[str] = []
        child_summaries: List[Dict[str, Any]] = []
        children = list(getattr(node, "all_children", []) or [])
        for child in children[:5]:
            cc = (child.content or "").strip()
            if not cc:
                continue
            cpos = (child.position.value or "").upper()
            child_lines.append(f"    - [{cpos}] {_truncate(cc, 240)}")
            child_summaries.append({
                "node_id": child.id,
                "position": child.position.value,
                "content_preview": _truncate(cc, 160),
            })

        header = (
            f"[tendency={tendency_id} dist={distance:.3f} {position}]"
        )
        lines.append(header)
        lines.append(f"  thesis: {_truncate(thesis, 160)}")
        lines.append(f"  parent: {_truncate(parent_content, 200)}")
        lines.append(f"  node:   {_truncate(node_content, 280)}")
        if child_lines:
            lines.append(f"  insights ({len(child_lines)} sub-claim children):")
            lines.extend(child_lines)
        lines.append("")

        summary.append({
            "tendency": tendency_id,
            "node_id": node_id,
            "distance": float(distance),
            "position": node.position.value,
            "content_preview": _truncate(node_content, 120),
            "n_children_shown": len(child_lines),
            "children": child_summaries,
        })

    return "\n".join(lines).rstrip() + "\n", summary


# ---------------------------------------------------------------------------
# Bridge call wrappers
# ---------------------------------------------------------------------------


async def call_bridge(
    provider: BridgeProvider,
    *,
    system: str,
    user: str,
    model: str,
) -> Tuple[Optional[str], float]:
    """Call the bridge once. Returns (text or None on error, elapsed_seconds)."""
    started = time.time()
    try:
        result = await provider.send(
            messages=[{"role": "user", "content": user}],
            system=system,
            model=model,
        )
    except Exception as e:
        elapsed = time.time() - started
        log.warning("bridge.send() failed (model=%s): %s", model, e)
        raise
    elapsed = time.time() - started
    text = result.text or ""
    return text, elapsed


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------


async def answer_a1(
    haiku: BridgeProvider,
    question: str,
) -> Dict[str, Any]:
    """A1 -- Haiku, no context."""
    out: Dict[str, Any] = {"answer": None, "model": "haiku", "elapsed_s": 0.0}
    try:
        text, elapsed = await call_bridge(
            haiku, system=A1_SYSTEM_PROMPT, user=question, model="haiku",
        )
        out["answer"] = text
        out["elapsed_s"] = elapsed
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


async def answer_a2(
    sonnet: BridgeProvider,
    question: str,
    codebase_dump: str,
) -> Dict[str, Any]:
    """A2 -- Sonnet, full codebase dump in context."""
    out: Dict[str, Any] = {"answer": None, "model": "sonnet", "elapsed_s": 0.0}
    user_text = (
        "CODEBASE:\n"
        f"{codebase_dump}\n\n"
        "QUESTION:\n"
        f"{question}"
    )
    try:
        text, elapsed = await call_bridge(
            sonnet, system=A2_SYSTEM_PROMPT, user=user_text, model="sonnet",
        )
        out["answer"] = text
        out["elapsed_s"] = elapsed
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


async def answer_a3(
    haiku: BridgeProvider,
    question: str,
    world: World,
    embedder: Any,
    top_k: int,
) -> Dict[str, Any]:
    """A3 -- Haiku, with graph regions located via Chain(Keyword, Coordinate)."""
    out: Dict[str, Any] = {
        "answer": None,
        "model": "haiku",
        "elapsed_s": 0.0,
        "region_size": 0,
        "region_summary": [],
    }
    try:
        # Embed question into substrate coords (used by CoordinateLocator).
        coords = coords_for_query(question, embedder=embedder)

        # We have two locators with very different content interfaces:
        #   - KeywordLocator wants TEXT (the raw question string).
        #   - CoordinateLocator wants COORDS (a tuple of floats).
        # ChainLocator passes the same `content` to each locator in turn,
        # so we wrap it as a dict that exposes both views. The locators'
        # heterogeneous extractors (_text_of, _coords_of) will each pull
        # what they need: KeywordLocator finds "text", CoordinateLocator
        # finds "coords".
        content = {"text": question, "coords": list(coords)}

        chain = ChainLocator(locators=[
            KeywordLocator(max_results=top_k),
            CoordinateLocator(max_results=top_k),
        ])
        region = chain(world, content)

        # Truncate to top_k just in case (KeywordLocator already capped, but
        # be defensive).
        region = region[:top_k]

        graph_context, summary = format_graph_context(world, region)
        out["region_size"] = len(region)
        out["region_summary"] = summary

        user_text = (
            f"{graph_context}\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "Answer using ONLY the graph regions above."
        )

        text, elapsed = await call_bridge(
            haiku, system=A3_SYSTEM_PROMPT, user=user_text, model="haiku",
        )
        out["answer"] = text
        out["elapsed_s"] = elapsed
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# ---------------------------------------------------------------------------
# Status writer
# ---------------------------------------------------------------------------


class StatusWriter:
    def __init__(self, path: Path, n_questions: int) -> None:
        self._path = path
        self._state: Dict[str, Any] = {
            "started_at": time.time(),
            "phase": "running",
            "n_questions_total": n_questions,
            "current_question": 0,
            "a1_done": 0,
            "a2_done": 0,
            "a3_done": 0,
            "errors": 0,
            "last_update": time.time(),
        }
        self.write()

    def update(self, **kwargs: Any) -> None:
        self._state.update(kwargs)
        self._state["last_update"] = time.time()

    def write(self) -> None:
        try:
            with self._path.open("w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            log.debug("status write failed: %s", e)

    def bump(self, key: str, by: int = 1) -> None:
        self._state[key] = int(self._state.get(key, 0)) + by
        self._state["last_update"] = time.time()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    started_at = time.time()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"  ERROR: questions file not found: {questions_path}", file=sys.stderr)
        return 2

    trained_world_path = Path(args.trained_world)
    if not trained_world_path.exists():
        print(
            f"  ERROR: trained-world JSON not found: {trained_world_path}",
            file=sys.stderr,
        )
        return 2

    all_questions = load_questions(questions_path)
    if args.limit > 0:
        questions = all_questions[: args.limit]
    else:
        questions = all_questions
    n_total = len(questions)
    print(
        f"  loaded {len(all_questions)} questions; running first {n_total}"
    )

    # Build / reuse codebase dump for A2.
    codebase_cache_path = Path(args.codebase_cache)
    if args.no_codebase_cache and codebase_cache_path.exists():
        print(f"  --no-codebase-cache: rebuilding codebase dump")
    codebase_dump = load_or_build_codebase_dump(
        codebase_cache_path, no_cache=args.no_codebase_cache,
    )
    print(f"  codebase dump: {len(codebase_dump):,} chars (cache: {codebase_cache_path})")

    # Load trained world.
    dim = args.dim
    embedder = default_usefulness_embedder(dim=dim)
    print(f"  loading trained world from {trained_world_path}...")
    world = load_trained_world(trained_world_path, dim=dim, bandwidth=args.bandwidth)
    n_world_nodes = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())
    print(f"  trained world: {len(world.tendencies)} tendencies, {n_world_nodes} nodes")

    # Status file.
    status = StatusWriter(Path(args.status), n_questions=n_total)

    # Bridge providers -- one per model class. Each spawns its own bridge
    # subprocess; both run concurrently.
    haiku_provider = BridgeProvider(model="haiku")
    sonnet_provider = BridgeProvider(model="sonnet")

    # Output writer -- append-mode JSONL so partial progress survives crashes.
    out_path = Path(args.out)
    # Truncate at start so re-runs don't accumulate.
    out_path.write_text("", encoding="utf-8")

    print(
        f"\n  This will make ~{n_total*3} bridge calls "
        f"({n_total} haiku no-ctx + {n_total} sonnet codebase + {n_total} haiku substrate) "
        f"against your Claude Max subscription.\n"
    )

    try:
        for i, q in enumerate(questions, start=1):
            qid = q.get("id", f"q{i:02d}")
            category = q.get("category", "")
            qtext = q.get("question", "") or ""

            status.update(current_question=i)
            status.write()

            print(f"  [{i}/{n_total}] {qid} ({category})")

            # Run A1, A2, A3 concurrently. Each owns its own provider, so
            # they don't contend on the same bridge subprocess.
            a1_task = asyncio.create_task(answer_a1(haiku_provider, qtext))
            a2_task = asyncio.create_task(
                answer_a2(sonnet_provider, qtext, codebase_dump)
            )
            a3_task = asyncio.create_task(
                answer_a3(haiku_provider, qtext, world, embedder, args.top_k)
            )

            # NOTE: a1 and a3 share haiku_provider. The provider's send()
            # serializes via an internal asyncio.Lock, so the two tasks
            # run sequentially against that subprocess. That's fine -- the
            # sonnet task runs in parallel anyway, so total wall-time is
            # roughly max(a1+a3, a2) per question. We don't try to spawn
            # a second haiku subprocess because the cost of a second TS
            # process isn't worth the small gain.

            a1_res, a2_res, a3_res = await asyncio.gather(
                a1_task, a2_task, a3_task, return_exceptions=False,
            )

            # Tally
            err_count = 0
            if a1_res.get("answer") is not None:
                status.bump("a1_done")
            else:
                err_count += 1
            if a2_res.get("answer") is not None:
                status.bump("a2_done")
            else:
                err_count += 1
            if a3_res.get("answer") is not None:
                status.bump("a3_done")
            else:
                err_count += 1
            if err_count:
                status.bump("errors", by=err_count)

            row: Dict[str, Any] = {
                "question_id": qid,
                "category": category,
                "question": qtext,
                "expected_modules": q.get("expected_modules", []),
                "notes": q.get("notes", ""),
                "a1_no_context": a1_res,
                "a2_with_code": a2_res,
                "a3_substrate": a3_res,
            }

            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            status.write()

            print(
                f"    a1={a1_res.get('elapsed_s', 0):.1f}s "
                f"a2={a2_res.get('elapsed_s', 0):.1f}s "
                f"a3={a3_res.get('elapsed_s', 0):.1f}s "
                f"region={a3_res.get('region_size', 0)}"
            )

        status.update(phase="complete")
        status.write()

    except Exception as e:
        log.exception("contest run failed")
        status.update(phase="failed")
        status.write()
        raise
    finally:
        for prov in (haiku_provider, sonnet_provider):
            try:
                await prov.close()
            except Exception as e:
                log.warning("provider.close() failed: %s", e)

    elapsed = time.time() - started_at
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  questions:        {n_total}")
    print(f"  a1 answered:      {status._state['a1_done']}")
    print(f"  a2 answered:      {status._state['a2_done']}")
    print(f"  a3 answered:      {status._state['a3_done']}")
    print(f"  errors:           {status._state['errors']}")
    print(f"  elapsed:          {elapsed:.1f}s")
    print(f"  output:           {out_path}")
    print(f"  status:           {args.status}")
    print(f"  codebase cache:   {codebase_cache_path}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Three-contestant runner for the substrate-vs-frontier reasoning "
            "contest. Produces three answers per question (no-context Haiku, "
            "codebase-dump Sonnet, substrate-augmented Haiku) and writes one "
            "JSONL row per question. NO API keys -- uses the Claude Max "
            "bridge."
        )
    )
    parser.add_argument(
        "--questions",
        default="questions.jsonl",
        help="JSONL of contest questions (default: questions.jsonl).",
    )
    parser.add_argument(
        "--trained-world",
        default="trained_world_with_judges.json",
        help=(
            "Path to the trained-world events JSON produced by "
            "train_with_judges.py (default: trained_world_with_judges.json)."
        ),
    )
    parser.add_argument(
        "--units",
        default="work_units_filtered.jsonl",
        help=(
            "Path to work units JSONL (kept for parity with the other "
            "scripts; not currently consumed during contest runs)."
        ),
    )
    parser.add_argument(
        "--out",
        default="contest_results.jsonl",
        help="Output JSONL path (default: contest_results.jsonl).",
    )
    parser.add_argument(
        "--status",
        default="status_contest.json",
        help="Live status file, updated after each question.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Run only the first N questions (default: 30; 0 = all).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        dest="top_k",
        help="Top-K neighbors to surface for A3 (default: 12).",
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
        help="Tendency bandwidth used to seed the world (default: 0.5).",
    )
    parser.add_argument(
        "--codebase-cache",
        default="_codebase_dump.txt",
        help=(
            "Path to cache the concatenated codebase dump for A2 "
            "(default: _codebase_dump.txt). Reused on subsequent runs."
        ),
    )
    parser.add_argument(
        "--no-codebase-cache",
        action="store_true",
        help="Force rebuild of the codebase dump even if the cache exists.",
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
