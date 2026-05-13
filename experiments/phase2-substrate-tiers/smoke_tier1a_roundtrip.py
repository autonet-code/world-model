#!/usr/bin/env python3
"""Smoke test: one qwen3.5:4b roundtrip for coord embedding.

Sends a single Python snippet, asks for JSON coords on three axes,
and verifies the response is parseable. Tunes prompt + token budget
until reliable.

qwen3.5:4b runs in thinking mode by default. The streaming API
separates `thinking` chunks from `response` chunks; we only care
about parseable JSON in the final response. Budget ~3000 tokens to
allow thinking + JSON output.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3.5:4b"
MAX_TOKENS = 3000


SYSTEM = """\
You are a code reviewer scoring beginner Python snippets on three axes.
For each snippet, return STRICT JSON with these keys:

  "correctness": -1.0 to +1.0
      +1.0 = clearly correct, does what its surface form implies
      -1.0 = clearly buggy or wrong
       0.0 = unclear / can't tell from snippet alone

  "simplicity":  -1.0 to +1.0
      +1.0 = minimal, easy to read at a glance
      -1.0 = unnecessarily complex, hard to follow
       0.0 = neutral / typical

  "idiom":       -1.0 to +1.0
      +1.0 = clearly pythonic (uses idiomatic constructs)
      -1.0 = clearly non-pythonic (C-style, fights the language)
       0.0 = neutral / not enough surface for an idiom call

  "rationale": one short sentence explaining your scores

Return ONLY the JSON object. No prose before or after. No markdown
fences. Just `{...}`.
"""

USER_TEMPLATE = """\
Snippet:

```python
{snippet}
```

Score this snippet on the three axes. Return JSON only.
"""


SNIPPET_GOLD = """\
def total(numbers):
    return sum(numbers)
"""

SNIPPET_BUGGY = """\
def total(numbers):
    result = 0
    for i in range(1, len(numbers)):
        result += numbers[i]
    return result
"""


def call_ollama(system: str, user: str, max_tokens: int = MAX_TOKENS):
    payload = {
        "model": MODEL,
        "system": system,
        "prompt": user,
        "stream": True,
        "options": {"temperature": 0.3, "num_predict": max_tokens},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.time()
    response_parts: list[str] = []
    thinking_parts: list[str] = []
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
    thinking = "".join(thinking_parts).strip()
    return text, thinking, elapsed


def extract_json(text: str):
    """Try to parse text as JSON. If that fails, look for a `{...}`
    block inside the text and try again. Returns the parsed dict or
    None.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Last resort: greedy outermost {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def evaluate_one(label: str, snippet: str):
    print(f"=== {label} ===")
    print("snippet:")
    for ln in snippet.rstrip().split("\n"):
        print(f"    {ln}")
    user = USER_TEMPLATE.format(snippet=snippet.rstrip())
    text, thinking, elapsed = call_ollama(SYSTEM, user)
    print(f"  elapsed: {elapsed:.1f}s, response len: {len(text)} chars, "
          f"thinking len: {len(thinking)} chars")
    print(f"  response (first 500): {text[:500]!r}")
    parsed = extract_json(text)
    if parsed is None:
        print("  FAIL: could not parse JSON")
        if thinking:
            print(f"  thinking trace tail (last 400): {thinking[-400:]!r}")
        return None
    print(f"  parsed: {parsed}")
    # Sanity-check the keys
    required = {"correctness", "simplicity", "idiom"}
    missing = required - set(parsed.keys())
    if missing:
        print(f"  FAIL: missing keys {missing}")
        return None
    for k in required:
        v = parsed[k]
        if not isinstance(v, (int, float)):
            print(f"  FAIL: {k} = {v!r} not a number")
            return None
        if v < -1.0 or v > 1.0:
            print(f"  WARN: {k} = {v} out of [-1, 1]")
    print(f"  OK")
    return parsed


def main():
    print(f"model: {MODEL}, max_tokens: {MAX_TOKENS}, temp: 0.3")
    print()
    gold = evaluate_one("S_GOLD (sum)", SNIPPET_GOLD)
    print()
    buggy = evaluate_one("S_BUGGY (off-by-one)", SNIPPET_BUGGY)
    print()
    if gold and buggy:
        print("=== sanity check ===")
        gold_c = gold.get("correctness", 0)
        buggy_c = buggy.get("correctness", 0)
        print(f"  gold correctness:  {gold_c:+.2f}  (expect > 0)")
        print(f"  buggy correctness: {buggy_c:+.2f}  (expect < 0)")
        if gold_c > 0 and buggy_c < 0:
            print("  SMOKE PASS: LLM distinguishes gold vs buggy on correctness")
            return 0
        else:
            print("  SMOKE FAIL: signs not as expected")
            return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
