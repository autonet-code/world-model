#!/usr/bin/env python3
"""Extract work units from Claude Code session JSONL logs.

Each Claude Code session at ~/.claude/projects/<encoded>/<session>.jsonl
becomes one work unit:

  problem:    the first user message in the session
  resolution: a distilled summary of the assistant's work
              (concatenated text + tool actions, capped)
  outcome:    Outcome from the existing heuristic in outcomes.py,
              based on the user/assistant message sequence

The output is a flat JSONL of work units, one per session, ready to
be fed to the substrate. Cached so re-runs don't re-extract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Make the autonet substrate importable
_AUTONET = Path("C:/code/autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))

from nodes.common.world_model_substrate.outcomes import (  # type: ignore
    Outcome,
    extract_acceptance,
    extract_built_on,
    extract_kept,
    extract_paid,
)


# ---------------------------------------------------------------------------
# Reading Claude Code JSONL
# ---------------------------------------------------------------------------


def read_session(path: Path) -> List[Dict[str, Any]]:
    """Read a Claude Code session JSONL. Returns list of message
    records, each with at least {role, content, timestamp}.
    """
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Claude Code records have nested 'message': {role, content}
                msg = record.get("message", {})
                role = msg.get("role")
                content = msg.get("content")
                # Content can be a string OR a list of content-blocks.
                # Reduce to a string for the heuristics.
                text = _content_to_text(content)
                if role and text:
                    out.append({
                        "role": role,
                        "content": text,
                        "timestamp": record.get("timestamp", ""),
                        "type": record.get("type", ""),
                        "session_id": record.get("sessionId", ""),
                    })
    except FileNotFoundError:
        return []
    return out


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Claude content is often a list of dicts: [{type, text}, {type, tool_use, ...}, ...]
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block.get("text", "")))
                elif "content" in block and isinstance(block["content"], str):
                    parts.append(block["content"])
                elif block.get("type") == "tool_use":
                    name = block.get("name", "")
                    inp = block.get("input", {})
                    if isinstance(inp, dict):
                        # Surface the most common tool params
                        cmd = inp.get("command") or inp.get("file_path") or ""
                        if cmd:
                            parts.append(f"[tool {name}: {cmd[:100]}]")
                        else:
                            parts.append(f"[tool {name}]")
                elif block.get("type") == "tool_result":
                    result = block.get("content")
                    if isinstance(result, str):
                        parts.append(f"[result: {result[:200]}]")
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(p for p in parts if p)
    return str(content)


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------


PROBLEM_MAX = 500
RESOLUTION_MAX = 1500


def _looks_like_system_template(text: str) -> bool:
    """Heuristic: is this first user message actually a system-prompt
    template (e.g., automated agent setup) rather than a real user
    request?
    """
    if len(text) < 30:
        return False
    lower = text.lower()
    # System-template markers
    markers = (
        "you are a ",
        "you have access to",
        "your task is to",
        "follow these instructions",
        "## task ",
        "answer these ",
        "score each ",
        "for each question",
    )
    return any(m in lower for m in markers)


def distill_problem(messages: List[Dict[str, Any]]) -> str:
    """The actual user request. Skips system-prompt-style first
    messages and falls back to the next user message that looks like
    a real ask.
    """
    user_msgs: List[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = (msg.get("content", "") or "").strip()
        if len(text) < 10:
            continue
        user_msgs.append(text)

    if not user_msgs:
        return ""

    # If first message looks like a system template, prefer the second.
    if len(user_msgs) >= 2 and _looks_like_system_template(user_msgs[0]):
        return user_msgs[1][:PROBLEM_MAX]
    return user_msgs[0][:PROBLEM_MAX]


def distill_resolution(messages: List[Dict[str, Any]]) -> str:
    """Concatenate assistant text from the session, capped."""
    parts: List[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        text = msg.get("content", "") or ""
        if not text.strip():
            continue
        parts.append(text)
        if sum(len(p) for p in parts) >= RESOLUTION_MAX * 2:
            break
    joined = " ".join(parts)
    return joined[:RESOLUTION_MAX]


def compute_outcome(messages: List[Dict[str, Any]]) -> Outcome:
    return Outcome(
        accepted=extract_acceptance(messages),
        kept=extract_kept(messages),
        built_on=0.0,   # need cross-session signal; punt for v1
        paid=extract_paid(messages),
    )


# ---------------------------------------------------------------------------
# Project iteration
# ---------------------------------------------------------------------------


def project_dirs(claude_root: Path, project_match: Optional[str] = None) -> List[Path]:
    if not claude_root.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(claude_root.iterdir()):
        if not p.is_dir():
            continue
        if project_match and project_match not in p.name:
            continue
        out.append(p)
    return out


def session_files(project_dir: Path, include_subagents: bool = False) -> List[Path]:
    """Return JSONL session files at the top level of project_dir.

    By default, excludes subagent sessions (they're under subdirs)
    because top-level sessions are the user-driven work units; subagent
    sessions are operational helpers.
    """
    if not project_dir.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(project_dir.iterdir()):
        if p.is_file() and p.suffix == ".jsonl":
            out.append(p)
    if include_subagents:
        for p in project_dir.rglob("*.jsonl"):
            if p not in out:
                out.append(p)
    return out


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------


def extract_one(path: Path) -> Optional[Dict[str, Any]]:
    messages = read_session(path)
    if not messages:
        return None
    problem = distill_problem(messages)
    if len(problem) < 10:
        return None
    resolution = distill_resolution(messages)
    if len(resolution) < 10:
        return None
    outcome = compute_outcome(messages)
    return {
        "session_path": str(path),
        "problem": problem,
        "resolution": resolution,
        "outcome": list(outcome.to_coords()),
        "timestamp": messages[0].get("timestamp", ""),
        "n_messages": len(messages),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-match", default="code",
                        help="substring to match against project dir names")
    parser.add_argument("--exclude-match", default="",
                        help="comma-separated substrings to exclude from project dir names")
    parser.add_argument("--limit", type=int, default=None,
                        help="max sessions to extract")
    parser.add_argument("--out", default="work_units.jsonl",
                        help="output JSONL file")
    parser.add_argument("--include-subagents", action="store_true",
                        help="include subagent sessions (default: top-level only)")
    args = parser.parse_args()

    claude_root = Path(os.path.expanduser("~/.claude/projects"))
    projects = project_dirs(claude_root, project_match=args.project_match)
    if args.exclude_match:
        excludes = [e.strip() for e in args.exclude_match.split(",") if e.strip()]
        projects = [p for p in projects
                    if not any(e in p.name for e in excludes)]
        print(f"  excluded substrings: {excludes}")
    print(f"  found {len(projects)} project dirs matching {args.project_match!r}")

    all_sessions: List[Path] = []
    for proj in projects:
        sessions = session_files(proj, include_subagents=args.include_subagents)
        print(f"    {proj.name}: {len(sessions)} sessions")
        all_sessions.extend(sessions)
    print(f"  total sessions: {len(all_sessions)}")

    if args.limit is not None:
        all_sessions = all_sessions[:args.limit]
        print(f"  limited to: {len(all_sessions)}")

    out_path = Path(args.out)
    n_written = 0
    n_skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for path in all_sessions:
            unit = extract_one(path)
            if unit is None:
                n_skipped += 1
                continue
            f.write(json.dumps(unit) + "\n")
            n_written += 1

    print(f"\n  written: {n_written} work units to {out_path}")
    print(f"  skipped: {n_skipped} (empty / too short)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
