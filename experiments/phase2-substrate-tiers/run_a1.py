#!/usr/bin/env python3
"""Phase 2 A1: does exposing n in the prompt change qwen's answers?

Two arms per question:
  WITH-N:    facts shown with [confidence: settled/medium/contested] tags
             derived from n value.
  WITHOUT-N: same facts, no confidence annotation.

Same model (qwen3.5:4b), same retrieval (all 23 facts -- the graph is
small enough), same questions. Eyeball whether n-aware answers hedge
better on contested claims.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
GRAPH_PATH = HERE / "graph_a1.json"
QUESTIONS_PATH = HERE / "questions_a1.json"
RESULTS_PATH = HERE / "a1_results.json"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3.5:4b"
MAX_TOKENS = 6000
STATUS_PATH = None  # set in main()


def n_to_label(n: float) -> str:
    """Human-readable confidence band from n value."""
    if n < 0.15:
        return "settled"
    if n < 0.40:
        return "established"
    if n < 0.65:
        return "uncertain"
    return "contested"


SYSTEM_BASE = (
    "You are answering a question about the Verdane Trade Union, a fictional "
    "setting. Be concise and concrete. If something is not stated, say so. "
    "Keep your answer under 150 words."
)
SYSTEM_WITH_N = SYSTEM_BASE + (
    "\n\nThe FACTS below come with confidence labels in brackets. "
    "These reflect how settled each claim is in the world's record:\n"
    "  [settled]     -- well-established, multiply confirmed\n"
    "  [established] -- broadly accepted, no recent contestation\n"
    "  [uncertain]   -- meaningful variability or open questions\n"
    "  [contested]   -- recent contradictions or unconfirmed claims\n\n"
    "When facts conflict, weight them by their confidence labels. Settled "
    "facts should be treated as authoritative; contested facts should be "
    "reported with appropriate hedging language ('reportedly', "
    "'unconfirmed', 'allegedly') rather than as established truth."
)
SYSTEM_WITHOUT_N = SYSTEM_BASE + (
    "\n\nUse the FACTS below to answer. If facts conflict, do your best "
    "to reconcile them or note the conflict."
)


def format_facts_with_n(nodes: list[dict]) -> str:
    lines = ["FACTS:"]
    for n in nodes:
        label = n_to_label(n["n"])
        lines.append(f"  [{label}] {n['text']}")
    return "\n".join(lines)


def format_facts_without_n(nodes: list[dict]) -> str:
    lines = ["FACTS:"]
    for n in nodes:
        lines.append(f"  - {n['text']}")
    return "\n".join(lines)


def call_ollama(system: str, user: str) -> tuple[str, float]:
    """Streamed call to ollama. Reads chunks line-by-line so the request
    socket stays active and we don't time out on a single big response.
    Aggregates response and thinking separately, then returns the
    response (or thinking trace if response was empty).
    """
    payload = {
        "model": MODEL,
        "system": system,
        "prompt": user,
        "stream": True,
        "options": {"temperature": 0.0, "num_predict": MAX_TOKENS},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.time()
    response_parts: list[str] = []
    thinking_parts: list[str] = []
    # Long socket-read timeout per chunk (chunks arrive every few seconds).
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "response" in obj and obj["response"]:
                response_parts.append(obj["response"])
            if "thinking" in obj and obj["thinking"]:
                thinking_parts.append(obj["thinking"])
            if obj.get("done"):
                break
    elapsed = time.time() - started
    text = "".join(response_parts).strip()
    if not text:
        thinking = "".join(thinking_parts).strip()
        if thinking:
            text = f"[NO RESPONSE EMITTED -- raw thinking trace below]\n{thinking}"
    return text, elapsed


def write_status(status: dict) -> None:
    """Atomic-ish status write. Updated after each (question, arm) pair."""
    status["last_update"] = time.time()
    (HERE / "a1_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )


def append_result_jsonl(result: dict) -> None:
    """Append one result as a JSONL line. Survives crashes."""
    with open(HERE / "a1_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def load_existing_results() -> dict[str, dict]:
    """Load previously-written results from JSONL so we can resume."""
    path = HERE / "a1_results.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[row["id"]] = row
    return out


def main():
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    nodes = graph["nodes"]

    facts_with_n = format_facts_with_n(nodes)
    facts_without_n = format_facts_without_n(nodes)

    existing = load_existing_results()
    if existing:
        print(f"Resuming: {len(existing)} questions already done -- "
              f"{sorted(existing.keys())}")

    n_total = len(questions["questions"])
    status = {
        "started_at": time.time(),
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "n_questions": n_total,
        "current_question": 0,
        "current_arm": None,
        "completed": list(existing.keys()),
        "errors": 0,
        "last_update": time.time(),
    }
    write_status(status)

    print(f"Running {n_total} questions x 2 arms = {2 * n_total} qwen calls "
          f"(model={MODEL}, max_tokens={MAX_TOKENS})\n")

    for q_idx, q in enumerate(questions["questions"], 1):
        qid = q["id"]
        if qid in existing:
            print(f"[{qid}] -- skipping, already in jsonl")
            continue

        status["current_question"] = q_idx
        write_status(status)

        print("=" * 78)
        print(f"[{q_idx}/{n_total}] {qid}: {q['question']}")
        print(f"  rationale: {q['rationale'][:120]}")
        print()

        try:
            # Arm: WITHOUT-N
            status["current_arm"] = "without_n"
            write_status(status)
            user_wo = f"{facts_without_n}\n\nQUESTION: {q['question']}\n\nAnswer."
            text_wo, t_wo = call_ollama(SYSTEM_WITHOUT_N, user_wo)
            print(f"-- WITHOUT-N ({t_wo:.1f}s, {len(text_wo)} chars) --")
            print("  " + text_wo[:800].replace("\n", "\n  "))
            print()

            # Arm: WITH-N
            status["current_arm"] = "with_n"
            write_status(status)
            user_wn = f"{facts_with_n}\n\nQUESTION: {q['question']}\n\nAnswer."
            text_wn, t_wn = call_ollama(SYSTEM_WITH_N, user_wn)
            print(f"-- WITH-N ({t_wn:.1f}s, {len(text_wn)} chars) --")
            print("  " + text_wn[:800].replace("\n", "\n  "))
            print()

            row = {
                "id": qid,
                "question": q["question"],
                "rationale": q["rationale"],
                "expected_with_n": q["expected_with_n"],
                "expected_without_n": q["expected_without_n"],
                "without_n": {"answer": text_wo, "elapsed_s": t_wo},
                "with_n": {"answer": text_wn, "elapsed_s": t_wn},
            }
            append_result_jsonl(row)
            status["completed"].append(qid)
            write_status(status)
        except Exception as e:
            print(f"  ERROR on {qid}: {e}")
            status["errors"] += 1
            write_status(status)
            raise

    # Final consolidated JSON for convenience
    final_results = list(load_existing_results().values())
    RESULTS_PATH.write_text(
        json.dumps({"results": final_results}, indent=2), encoding="utf-8"
    )
    print(f"\nresults saved to {RESULTS_PATH}")
    status["current_arm"] = None
    status["phase"] = "complete"
    write_status(status)


if __name__ == "__main__":
    main()
