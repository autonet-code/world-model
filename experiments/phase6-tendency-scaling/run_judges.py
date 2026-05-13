#!/usr/bin/env python3
"""Phase 6 judge runner.

For each train function, calls sonnet once asking for sub-claims on
ALL 10 toolz-domain tendencies. Caches to judge_cache.jsonl. The
substrate builders for N=2,4,6,8,10 truncate this cached output to
their active axis subset.

Observability:
  - phase6/status.json updated per function
  - phase6/judge_log.jsonl appended one row per call (raw + parsed)
  - Pre-flight smoke check on 1 function before committing.
  - Mid-run quality guard: if 3 consecutive functions have <50% axis
    coverage, halt with summary of raw responses.
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

from atn.providers.bridge import BridgeProvider  # type: ignore  # noqa: E402


HERE = Path(__file__).resolve().parent
CORPUS_PATH = HERE / "corpus.json"
CACHE_PATH = HERE / "judge_cache.jsonl"
LOG_PATH = HERE / "judge_log.jsonl"
STATUS_PATH = HERE / "status.json"


TENDENCIES = [
    "correctness", "simplicity", "robustness", "purity",
    "laziness", "composability", "type_flexibility", "error_clarity",
    "efficiency", "documentation_fidelity",
]


JUDGE_SYSTEM = """You are a code reviewer extracting structured sub-claims for a graph substrate.

Given one Python function (name, signature, docstring, implementation), produce up to 3 sub-claims PER AXIS on these 10 axes:

  correctness: things TRUE about whether/how this function does what it claims (edge cases handled, invariants, return types, valid inputs).
  simplicity: TRUE about the structural shape (branching depth, idioms, what could be simpler).
  robustness: TRUE about behavior on degenerate input (empty, single-element, mixed types, None).
  purity: TRUE about side-effect-freeness (does it mutate inputs? have hidden state? is it deterministic?).
  laziness: TRUE about evaluation strategy (does it return a generator? materialize too eagerly? streaming-safe?).
  composability: TRUE about pipeline ergonomics (return type pipes well? consumer-friendly?).
  type_flexibility: TRUE about input type tolerance (works on lists, tuples, generators, dicts as appropriate?).
  error_clarity: TRUE about failure mode (informative exception? silent wrong answer? graceful degradation?).
  efficiency: TRUE about algorithmic complexity (appropriate big-O? avoidable copies? hot-path concerns?).
  documentation_fidelity: TRUE about whether the implementation actually matches the docstring's claims (or diverges).

Each sub-claim is one short factual sentence. NOT opinion. NOT a docstring summary. An insightful observation a careful reader of the source would make.

Some axes may legitimately have nothing substantive to say about a given function. For those axes, return an empty list. But aim for ≥1 sub-claim per axis when there's anything to say.

Respond with ONLY this JSON (no markdown, no preamble):
{
  "correctness": ["claim", ...],
  "simplicity": ["claim", ...],
  "robustness": ["claim", ...],
  "purity": ["claim", ...],
  "laziness": ["claim", ...],
  "composability": ["claim", ...],
  "type_flexibility": ["claim", ...],
  "error_clarity": ["claim", ...],
  "efficiency": ["claim", ...],
  "documentation_fidelity": ["claim", ...]
}"""


def judge_user_prompt(entry: Dict[str, Any]) -> str:
    return (
        f"Function name: {entry['name']}\n"
        f"Signature: {entry['signature']}\n\n"
        f"Docstring:\n{entry['docstring']}\n\n"
        f"Implementation:\n{entry.get('impl_full_source','(unavailable)')}\n\n"
        "Extract sub-claims on all 10 axes now."
    )


def cache_key(entry: Dict[str, Any]) -> str:
    payload = json.dumps({
        "name": entry["name"],
        "doc": entry["docstring"],
        "impl": entry.get("impl_full_source", ""),
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
            out[row["key"]] = row
        except json.JSONDecodeError:
            continue
    return out


def append_cache(row: Dict[str, Any]) -> None:
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_log(row: Dict[str, Any]) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_response(text: str) -> Optional[Dict[str, List[str]]]:
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
    for axis in TENDENCIES:
        v = parsed.get(axis, [])
        if isinstance(v, list):
            out[axis] = [str(x).strip() for x in v if str(x).strip()][:4]
        else:
            out[axis] = []
    return out


def axis_coverage(parsed: Dict[str, List[str]]) -> float:
    non_empty = sum(1 for axis in TENDENCIES if parsed.get(axis))
    return non_empty / len(TENDENCIES)


def write_status(state: Dict[str, Any]) -> None:
    state["last_update"] = time.time()
    STATUS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


async def call_sonnet(provider: BridgeProvider, user: str) -> Tuple[Optional[Dict[str, List[str]]], str, float]:
    started = time.time()
    try:
        result = await provider.send(
            messages=[{"role": "user", "content": user}],
            system=JUDGE_SYSTEM,
            model="sonnet",
        )
        text = result.text or ""
    except Exception as e:
        return None, f"BRIDGE_ERROR: {type(e).__name__}: {e}", time.time() - started
    parsed = parse_response(text)
    return parsed, text, time.time() - started


async def smoke_check(provider: BridgeProvider, sample_entry: Dict[str, Any]) -> bool:
    print("  smoke check: one sonnet call to verify 10-axis schema...")
    parsed, raw, dt = await call_sonnet(provider, judge_user_prompt(sample_entry))
    if parsed is None:
        print(f"  SMOKE FAIL: response did not parse. raw[:300]={raw[:300]!r}")
        return False
    coverage = axis_coverage(parsed)
    n_claims = sum(len(parsed.get(a, [])) for a in TENDENCIES)
    print(f"  smoke: coverage={coverage:.0%}, total_claims={n_claims}, elapsed={dt:.1f}s")
    if coverage < 0.6:
        print(f"  SMOKE FAIL: axis coverage too low ({coverage:.0%}).")
        for axis in TENDENCIES:
            print(f"    {axis}: {len(parsed.get(axis, []))} claims")
        return False
    print(f"  smoke OK")
    return True


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    train_entries = corpus["train"]
    print(f"  train entries: {len(train_entries)}")

    cache = load_cache()
    print(f"  existing cache entries: {len(cache)}")

    provider = BridgeProvider(model="sonnet")

    try:
        # Smoke check on first uncached entry.
        first_uncached = next(
            (e for e in train_entries if cache_key(e) not in cache), None
        )
        if first_uncached is not None:
            ok = await smoke_check(provider, first_uncached)
            if not ok:
                print("  Halting before main loop (smoke failed).")
                return 2

        n_calls = 0
        n_hits = 0
        n_parse_fail = 0
        consecutive_low_coverage = 0
        coverage_history: List[float] = []
        started = time.time()

        state: Dict[str, Any] = {
            "phase": "judging",
            "n_entries": len(train_entries),
            "current": 0,
            "n_calls": 0, "n_hits": 0, "n_parse_fail": 0,
            "axis_coverage_recent": [],
        }
        write_status(state)

        for i, entry in enumerate(train_entries, start=1):
            state["current"] = i
            state["current_name"] = entry["name"]
            write_status(state)

            key = cache_key(entry)
            if key in cache:
                n_hits += 1
                coverage_history.append(axis_coverage(cache[key]["parsed"]))
                state["n_hits"] = n_hits
                continue

            parsed, raw, dt = await call_sonnet(provider, judge_user_prompt(entry))
            n_calls += 1
            state["n_calls"] = n_calls

            log_row = {
                "name": entry["name"],
                "elapsed_s": dt,
                "raw_text": raw,
                "parsed": parsed,
                "ts": time.time(),
            }
            append_log(log_row)

            if parsed is None:
                n_parse_fail += 1
                state["n_parse_fail"] = n_parse_fail
                consecutive_low_coverage += 1
                print(f"  [{i:>3}/{len(train_entries)}] {entry['name']:>30}  "
                      f"PARSE FAIL (raw[:80]={raw[:80]!r})")
                if consecutive_low_coverage >= 3:
                    print()
                    print("  HALT: 3 consecutive parse failures.")
                    return 3
                continue

            coverage = axis_coverage(parsed)
            n_claims = sum(len(parsed.get(a, [])) for a in TENDENCIES)
            coverage_history.append(coverage)
            state["axis_coverage_recent"] = [round(c, 2) for c in coverage_history[-5:]]

            cache_row = {"key": key, "name": entry["name"], "parsed": parsed,
                         "raw_text": raw, "ts": time.time()}
            cache[key] = cache_row
            append_cache(cache_row)

            if coverage < 0.5:
                consecutive_low_coverage += 1
            else:
                consecutive_low_coverage = 0

            print(f"  [{i:>3}/{len(train_entries)}] {entry['name']:>30}  "
                  f"cov={coverage:.0%} claims={n_claims:>2}  "
                  f"{dt:.0f}s  ({time.time()-started:.0f}s total)")

            if consecutive_low_coverage >= 3:
                print()
                print("  HALT: 3 consecutive functions with <50% axis coverage.")
                print("  Recent raw responses written to judge_log.jsonl for inspection.")
                return 4

            write_status(state)
    finally:
        try:
            await provider.close()
        except Exception:
            pass

    state["phase"] = "complete"
    write_status(state)
    print()
    print("=" * 60)
    print("JUDGES DONE")
    print(f"  calls: {n_calls}, hits: {n_hits}, parse fails: {n_parse_fail}")
    print(f"  avg coverage: {sum(coverage_history)/max(len(coverage_history),1):.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
