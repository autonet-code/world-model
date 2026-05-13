#!/usr/bin/env python3
"""Tier 0 floor test for the substrate-as-translator claim.

Contestants:
  A: qwen3.5:4b alone, no graph.
  B: qwen3.5:4b given the top-K graph nodes located by keyword overlap.

Both are asked the same question. We print both answers side-by-side
plus the gold answer summary for manual eyeballing. This validates the
floor of the architecture: can a small model render an answer better
when handed the relevant graph regions, on a controlled domain where
we *know* what is in the graph?
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple


HERE = Path(__file__).resolve().parent
GRAPH_PATH = HERE / "graph.json"
QUESTIONS_PATH = HERE / "questions.json"
RESULTS_PATH = HERE / "tier0_results.json"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3.5:4b"
TOP_K = 6
# qwen3.5 is a thinking model. It emits a chain-of-thought into a separate
# `thinking` field before the `response` field. We need enough budget for both.
MAX_TOKENS = 3500
SYSTEM_BASE = (
    "You are answering a question about the Verdane Trade Union, a fictional "
    "setting. Be concise and concrete. If something is not stated, say so. "
    "Keep your answer under 120 words."
)
SYSTEM_WITH_GRAPH = SYSTEM_BASE + (
    "\n\nYou have been given a set of FACTS retrieved from a knowledge graph. "
    "Use ONLY those facts plus simple arithmetic to answer. If the facts "
    "contradict each other, say so. Do not invent facts not in the list."
)


_TOKEN_RE = re.compile(r"[A-Za-z]+")


def tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def locate(query: str, nodes: List[Dict[str, str]], k: int) -> List[Tuple[Dict[str, str], float]]:
    """Keyword-overlap locator. Returns top-k (node, score) pairs."""
    qtokens = set(tokens(query))
    qtokens -= {
        "the", "a", "an", "of", "to", "in", "and", "or", "is", "are",
        "what", "which", "how", "why", "do", "does", "any", "from",
        "for", "on", "with", "by", "be", "that", "as", "at", "it",
        "would", "if", "this", "that", "these", "those", "much", "many",
    }
    scored: List[Tuple[Dict[str, str], float]] = []
    for n in nodes:
        ntokens = set(tokens(n["text"]))
        if not ntokens or not qtokens:
            scored.append((n, 0.0))
            continue
        overlap = len(qtokens & ntokens)
        score = overlap / len(qtokens) if qtokens else 0.0
        scored.append((n, score))
    scored.sort(key=lambda x: (-x[1], x[0]["id"]))
    return scored[:k]


def call_ollama(system: str, user: str, model: str = MODEL, max_tokens: int = MAX_TOKENS) -> Tuple[str, float]:
    """Single-shot generation against ollama. Returns (text, elapsed_seconds)."""
    payload = {
        "model": model,
        "system": system,
        "prompt": user,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        return f"[ollama error: {e}]", time.time() - started
    elapsed = time.time() - started
    try:
        out = json.loads(body)
    except json.JSONDecodeError:
        return f"[bad json: {body[:200]}]", elapsed
    # Ollama splits chain-of-thought into `thinking` and the answer into
    # `response`. If response is empty (model ran out of budget mid-thought),
    # fall back to whatever it produced in thinking.
    text = (out.get("response", "") or "").strip()
    if not text:
        thinking = (out.get("thinking", "") or "").strip()
        if thinking:
            text = f"[NO RESPONSE EMITTED -- raw thinking trace below]\n{thinking}"
    return text, elapsed


def format_facts(located: List[Tuple[Dict[str, str], float]]) -> str:
    """Render located nodes as a numbered FACTS block."""
    lines = ["FACTS retrieved from the graph (top {} by relevance):".format(len(located)), ""]
    for i, (node, score) in enumerate(located, 1):
        lines.append(f"  {i}. [{node['id']} score={score:.2f}] {node['text']}")
    return "\n".join(lines)


def run() -> None:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    nodes = graph["nodes"]
    items = questions["questions"]

    results = []
    for idx, q in enumerate(items, 1):
        qid = q["id"]
        question = q["question"]
        gold_summary = q["gold_answer_summary"]
        gold_nodes = set(q["gold_nodes"])

        print("=" * 74)
        print(f"[{idx}/{len(items)}] {qid}")
        print(f"  question: {question}")

        # Contestant A: qwen alone, no graph
        a_text, a_elapsed = call_ollama(
            system=SYSTEM_BASE,
            user=question,
        )

        # Contestant B: qwen + located graph regions
        located = locate(question, nodes, k=TOP_K)
        located_ids = {n["id"] for n, _ in located}
        recall = len(located_ids & gold_nodes) / max(1, len(gold_nodes))
        facts_block = format_facts(located)
        b_user = f"{facts_block}\n\nQUESTION: {question}\n\nAnswer using ONLY the facts above plus simple arithmetic."
        b_text, b_elapsed = call_ollama(
            system=SYSTEM_WITH_GRAPH,
            user=b_user,
        )

        print(f"  -- A (qwen alone, {a_elapsed:.1f}s) --")
        print("    " + a_text.replace("\n", "\n    "))
        print()
        print(f"  -- B (qwen + graph, {b_elapsed:.1f}s) recall@{TOP_K}={recall:.2f} --")
        print(f"    located node ids: {sorted(n['id'] for n,_ in located)}")
        print(f"    gold node ids:    {sorted(gold_nodes)}")
        print("    " + b_text.replace("\n", "\n    "))
        print()
        print(f"  -- gold summary --")
        print("    " + gold_summary.replace("\n", "\n    "))
        print()

        results.append({
            "id": qid,
            "question": question,
            "gold_nodes": sorted(gold_nodes),
            "gold_answer_summary": gold_summary,
            "a_alone": {"answer": a_text, "elapsed_s": a_elapsed},
            "b_with_graph": {
                "answer": b_text,
                "elapsed_s": b_elapsed,
                "located_node_ids": [n["id"] for n, _ in located],
                "located_scores": [s for _, s in located],
                "gold_recall": recall,
            },
        })

    RESULTS_PATH.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print()
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    run()
