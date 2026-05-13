#!/usr/bin/env python3
"""Tier 1C: binary-classification prompt on the same 12 snippets.

Tests whether forcing the LLM to commit per-axis ("clear flaw?
yes=-1 / no=+1 / can't tell=0") closes the remaining failures
from Tier 1A and Tier 1B. Both qwen3.5:4b (local ollama) and
haiku-4-5 (claude-max-proxy) are run; results compared against
their graded-prompt counterparts.

Usage:
  python run_tier1c.py qwen      # qwen arm only
  python run_tier1c.py haiku     # haiku arm only
  python run_tier1c.py both      # both arms (default)

The proxy must be running for the haiku arm:
  cd c:\\code\\opencode-claude-max-proxy && bun run proxy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

sys.path.insert(0, r"C:\code\world-model")

from world_model.generalized import (  # type: ignore
    GeneralizedTendency, Observation, World,
)
from world_model.generalized.tendency import _intrinsic_score_in_tendency  # type: ignore
from world_model.generalized.prune import prune_veto_negatives  # type: ignore
from world_model.models.tree import Position  # type: ignore

from tier1a_snippets import SNIPPETS, CATEGORY_PAIRS  # type: ignore


HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Binary-classification prompt
# ---------------------------------------------------------------------------


SYSTEM = """\
You are a code reviewer flagging concerns on beginner Python snippets.
For each snippet, return STRICT JSON with these keys:

  "correctness": -1 | 0 | +1
      -1 = the snippet has a CLEAR correctness flaw (bug, wrong
           output, undefined behavior, off-by-one, etc.)
      +1 = NO clear correctness flaw -- it does what its surface
           form implies
       0 = the snippet is too small to tell (e.g. a single line
           with no observable behavior)

  "simplicity":  -1 | 0 | +1
      -1 = the snippet has a CLEAR simplicity flaw (unnecessary
           complexity, hard-to-follow control flow, dense code
           where simpler code would do)
      +1 = NO clear simplicity flaw
       0 = too small to tell

  "idiom":       -1 | 0 | +1
      -1 = the snippet has a CLEAR idiom flaw (non-pythonic
           constructs where pythonic ones exist: range(len()),
           manual loops where comprehensions fit, `is` for value
           comparison, etc.)
      +1 = NO clear idiom flaw
       0 = too small to tell

Commit to a verdict. Use 0 only when the snippet is genuinely
too small to evaluate (one-line imports, bare assignments).

Return ONLY the JSON object. No prose before or after. No
markdown fences. Just `{...}`.
"""

USER_TEMPLATE = """\
Snippet:

```python
{snippet}
```

Flag concerns on the three axes. Return JSON only.
"""


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------


N_SAMPLES = 3
TEMPERATURE = 0.3


def call_qwen(system: str, user: str, max_tokens: int = 5500):
    """qwen3.5:4b via local ollama, streaming."""
    payload = {
        "model": "qwen3.5:4b",
        "system": system,
        "prompt": user,
        "stream": True,
        "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.time()
    response_parts: List[str] = []
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
            if obj.get("done"):
                break
    elapsed = time.time() - started
    return "".join(response_parts).strip(), elapsed, {}


def call_haiku(system: str, user: str, max_tokens: int = 500):
    """haiku-4-5 via claude-max-proxy on localhost:3456."""
    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:3456/v1/messages", data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": "dummy",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    started = time.time()
    text_parts: List[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload_str = line[len("data:"):].strip()
            if not payload_str:
                continue
            try:
                obj = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "content_block_delta":
                d = obj.get("delta", {})
                if d.get("type") == "text_delta":
                    text_parts.append(d.get("text", ""))
            elif t == "message_start":
                u = (obj.get("message", {}) or {}).get("usage", {}) or {}
                usage = {
                    "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
                    "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0),
                }
            elif t == "message_delta":
                u = obj.get("usage", {}) or {}
                if "output_tokens" in u:
                    usage["output_tokens"] = u["output_tokens"]
    elapsed = time.time() - started
    return "".join(text_parts).strip(), elapsed, usage


CALLERS = {"qwen": call_qwen, "haiku": call_haiku}


# ---------------------------------------------------------------------------
# Parsing + validation
# ---------------------------------------------------------------------------


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def coerce_ternary(v) -> int:
    """Coerce a value to {-1, 0, +1}. Floats >0.5 -> +1, < -0.5 -> -1,
    else 0. Strings 'yes'/'no'/'true'/'false' similarly."""
    if isinstance(v, bool):
        return +1 if v else -1
    try:
        x = float(v)
        if x > 0.5:
            return +1
        if x < -0.5:
            return -1
        return 0
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("+1", "1", "yes", "true", "no flaw", "ok"):
            return +1
        if s in ("-1", "no", "false", "flaw"):
            return -1
        if s in ("0", "unclear", "n/a"):
            return 0
    return 0


def is_valid_sample(row: Dict) -> bool:
    p = row.get("parsed")
    if not isinstance(p, dict):
        return False
    return all(k in p for k in ("correctness", "simplicity", "idiom"))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def cache_path(model: str) -> Path:
    return HERE / f"tier1c_{model}_llm_cache.jsonl"


def load_cache(model: str) -> Dict[str, List[Dict]]:
    p = cache_path(model)
    if not p.exists():
        return {}
    out: Dict[str, List[Dict]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.setdefault(row["id"], []).append(row)
    return out


def append_cache(model: str, row: Dict) -> None:
    with open(cache_path(model), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def write_status(model: str, status: dict) -> None:
    status["last_update"] = time.time()
    (HERE / f"tier1c_{model}_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )


def embed_snippet(model: str, snippet_id: str, snippet: str,
                  existing: List[Dict], status: dict, max_attempts: int = 5):
    samples = list(existing)
    valid_count = sum(1 for s in samples if is_valid_sample(s))
    user = USER_TEMPLATE.format(snippet=snippet.rstrip())
    attempts = len(samples)
    while valid_count < N_SAMPLES and attempts < max_attempts:
        attempts += 1
        sample_idx = len(samples) + 1
        status["current_snippet"] = snippet_id
        status["current_sample"] = sample_idx
        write_status(model, status)
        text, elapsed, usage = CALLERS[model](SYSTEM, user)
        parsed = extract_json(text)
        row = {
            "id": snippet_id,
            "sample": sample_idx,
            "elapsed_s": elapsed,
            "raw_response": text,
            "usage": usage,
            "parsed": parsed,
        }
        append_cache(model, row)
        samples.append(row)
        ok = is_valid_sample(row)
        if ok:
            valid_count += 1
        cstate = ""
        if usage.get("cache_read_input_tokens"):
            cstate = f" cache_read={usage['cache_read_input_tokens']}"
        print(f"  [{snippet_id}/{sample_idx}] {elapsed:>5.1f}s "
              f"{'OK' if ok else 'BAD'} (valid {valid_count}/{N_SAMPLES}){cstate}  "
              f"{parsed if ok else text[:80]!r}")
    return samples


def median_coords(samples: List[Dict]) -> Tuple[Tuple[int, int, int], int]:
    valid: List[Tuple[int, int, int]] = []
    for s in samples:
        p = s.get("parsed")
        if not isinstance(p, dict):
            continue
        try:
            c = coerce_ternary(p["correctness"])
            si = coerce_ternary(p["simplicity"])
            i = coerce_ternary(p["idiom"])
        except KeyError:
            continue
        valid.append((c, si, i))
    if not valid:
        return (0, 0, 0), 0
    cs = [v[0] for v in valid]
    ss = [v[1] for v in valid]
    iss = [v[2] for v in valid]
    return (round(median(cs)), round(median(ss)), round(median(iss))), len(valid)


# ---------------------------------------------------------------------------
# Substrate (same as Tier 1A/B)
# ---------------------------------------------------------------------------


BANDWIDTH = 1.5


def build_world() -> World:
    world = World()
    correctness = GeneralizedTendency(
        id="correctness",
        thesis="Code is correct.",
        anchor=(1.0, 0.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        bandwidth=BANDWIDTH,
        veto_shaped=True,
        veto_score_floor=-0.5,
        novelty_gamma_pro=1.0,
        novelty_gamma_con=1.5,
    )
    simplicity = GeneralizedTendency(
        id="simplicity",
        thesis="Code is simple.",
        anchor=(0.0, 1.0, 0.0),
        polarity_axis=(0.0, 1.0, 0.0),
        bandwidth=BANDWIDTH,
    )
    idiom = GeneralizedTendency(
        id="idiom",
        thesis="Code is idiomatic.",
        anchor=(0.0, 0.0, 1.0),
        polarity_axis=(0.0, 0.0, 1.0),
        bandwidth=BANDWIDTH,
    )
    world.add_tendency(correctness)
    world.add_tendency(simplicity)
    world.add_tendency(idiom)
    return world


def round_step(world: World, obs_list: List[Observation]) -> None:
    for obs in obs_list:
        world.add_observation(obs)
    for tendency in world.tendencies.values():
        tendency.act(world)
    world.apply_stakes()
    for tendency in world.tendencies.values():
        tendency.update_novelty(dt=1.0)
    world.clear_observations()


def find_snip_node_in(world: World, snip_id: str, tendency_id: str):
    t = world.tendencies.get(tendency_id)
    if t is None:
        return None
    for node in t.tree.all_nodes():
        if (node.observation_id or "").startswith(f"{snip_id}_"):
            return node
    return None


# ---------------------------------------------------------------------------
# Predictions (same Q1-Q5 shape as Tier 1A v2)
# ---------------------------------------------------------------------------


def evaluate_predictions(median_per_snip, post_prune_presence) -> Tuple[int, int, List[dict]]:
    pass_count = 0
    fail_count = 0
    predictions: List[dict] = []

    def check(label: str, condition: bool, detail: str) -> None:
        nonlocal pass_count, fail_count
        ok = "PASS" if condition else "FAIL"
        if condition:
            pass_count += 1
        else:
            fail_count += 1
        print(f"  [{ok}] {label}: {detail}")
        predictions.append({"label": label, "condition": bool(condition), "detail": detail})

    # Q1: per-category sign expectations.
    expected_signs = {
        "gold":            ("+", "+", "+"),
        "quirky":          ("+", "+", "-"),
        "complex_correct": ("+", "-", "+"),
        "buggy":           ("-", None, None),
        "narrow":          (None, None, None),
        "bad_all":         ("-", "-", "-"),
    }

    def sign(x: int) -> str:
        return "+" if x > 0 else "-" if x < 0 else "0"

    q1_failures: List[str] = []
    for cat, sids in CATEGORY_PAIRS.items():
        expected = expected_signs.get(cat)
        if expected is None:
            continue
        for sid in sids:
            coords, _ = median_per_snip[sid]
            for axis_idx, axis_name in enumerate(("correctness", "simplicity", "idiom")):
                exp = expected[axis_idx]
                if exp is None:
                    continue
                got = sign(coords[axis_idx])
                if exp == "+" and got != "+":
                    q1_failures.append(f"{sid}/{axis_name}: expected +, got {got} ({coords[axis_idx]})")
                if exp == "-" and got != "-":
                    q1_failures.append(f"{sid}/{axis_name}: expected -, got {got} ({coords[axis_idx]})")
    check("Q1 (LLM consistent within category)", len(q1_failures) == 0,
          f"{len(q1_failures)} sign mismatches" + (
              f": {q1_failures[:3]}" + ("..." if len(q1_failures) > 3 else "")
              if q1_failures else ""))

    def any_in(category: str, tendency: str) -> bool:
        return any(post_prune_presence[sid][tendency]
                   for sid in CATEGORY_PAIRS[category])

    def none_in(category: str, tendency: str) -> bool:
        return all(not post_prune_presence[sid][tendency]
                   for sid in CATEGORY_PAIRS[category])

    q2_ok = any_in("gold", "correctness")
    check("Q2 (gold category survives correctness)", q2_ok,
          f"any S1/S2 in correctness post-prune: {q2_ok}")

    clearly_buggy = ["S7"]
    q3_ok = all(not post_prune_presence[sid]["correctness"]
                for sid in clearly_buggy)
    check("Q3 (clearly-buggy S7 vetoed)", q3_ok,
          f"S7 absent from correctness post-prune: "
          f"{not post_prune_presence['S7']['correctness']}")

    q4_vetoed = none_in("bad_all", "correctness")
    q4_in_others = any(
        post_prune_presence[sid]["simplicity"] or post_prune_presence[sid]["idiom"]
        for sid in CATEGORY_PAIRS["bad_all"]
    )
    check("Q4 (bad_all vetoed from correctness, present elsewhere)",
          q4_vetoed and q4_in_others,
          f"all bad_all vetoed: {q4_vetoed}; in non-veto trees: {q4_in_others}")

    q5_quirky = any_in("quirky", "correctness")
    q5_complex = any_in("complex_correct", "correctness")
    check("Q5 (quirky and complex categories survive correctness)",
          q5_quirky and q5_complex,
          f"any quirky in correctness: {q5_quirky}; "
          f"any complex_correct in correctness: {q5_complex}")

    return pass_count, fail_count, predictions


# ---------------------------------------------------------------------------
# Per-arm runner
# ---------------------------------------------------------------------------


def run_arm(model: str) -> dict:
    print(f"\n{'='*78}\nTier 1C arm: {model}\n{'='*78}")
    print(f"  cache: {cache_path(model)}")
    cache = load_cache(model)
    cache_hits = sum(1 for s in SNIPPETS
                     if sum(1 for r in cache.get(s.id, []) if is_valid_sample(r)) >= N_SAMPLES)
    print(f"  cache hits (snippets fully cached, valid): {cache_hits}/{len(SNIPPETS)}")
    print()

    status = {
        "started_at": time.time(),
        "phase": "embedding",
        "model": model,
        "n_snippets": len(SNIPPETS),
        "n_samples": N_SAMPLES,
        "current_snippet": None,
        "current_sample": None,
        "last_update": time.time(),
    }
    write_status(model, status)

    print("=== embedding pass ===")
    median_per_snip = {}
    for snip in SNIPPETS:
        existing = cache.get(snip.id, [])
        samples = embed_snippet(model, snip.id, snip.snippet, existing, status)
        coords, n_valid = median_coords(samples)
        median_per_snip[snip.id] = (coords, n_valid)
        print(f"  -> {snip.id} [{snip.category}] median = "
              f"({coords[0]:+}, {coords[1]:+}, {coords[2]:+}) "
              f"({n_valid}/{N_SAMPLES} valid)")
    print()

    print("=== substrate run ===")
    status["phase"] = "substrate"
    write_status(model, status)
    world = build_world()
    obs_seq = 0
    obs_list: List[Observation] = []
    for snip in SNIPPETS:
        coords, n_valid = median_per_snip[snip.id]
        if n_valid == 0:
            continue
        obs_seq += 1
        obs_list.append(Observation(
            id=f"{snip.id}_{obs_seq}",
            coords=tuple(float(c) for c in coords),
            label=f"{snip.id}:{snip.category}_{obs_seq}",
        ))
    round_step(world, obs_list)

    pre_prune_presence = {
        s.id: {tid: find_snip_node_in(world, s.id, tid) is not None
               for tid in world.tendencies}
        for s in SNIPPETS
    }
    veto_pruned = prune_veto_negatives(world)
    print(f"  prune_veto_negatives removed {len(veto_pruned)} node id(s)")
    post_prune_presence = {
        s.id: {tid: find_snip_node_in(world, s.id, tid) is not None
               for tid in world.tendencies}
        for s in SNIPPETS
    }

    print()
    print(f"  {'snip':>5} {'category':>16}  {'coords':>14}  "
          f"{'corr':>5} {'simp':>5} {'idio':>5}")
    for snip in SNIPPETS:
        coords, _ = median_per_snip[snip.id]
        coords_str = f"({coords[0]:+},{coords[1]:+},{coords[2]:+})"
        post = post_prune_presence[snip.id]
        c_mark = "T" if post["correctness"] else "-"
        s_mark = "T" if post["simplicity"] else "-"
        i_mark = "T" if post["idiom"] else "-"
        print(f"  {snip.id:>5} {snip.category:>16}  {coords_str:>14}  "
              f"{c_mark:>5} {s_mark:>5} {i_mark:>5}")

    print()
    print(f"=== predictions ({model}) ===")
    pass_count, fail_count, predictions = evaluate_predictions(
        median_per_snip, post_prune_presence
    )
    print(f"\n  {pass_count}/{pass_count+fail_count} predictions passed")

    serialisable_medians = {
        sid: {"coords": list(coords), "n_valid": n_valid,
              "category": next(s.category for s in SNIPPETS if s.id == sid)}
        for sid, (coords, n_valid) in median_per_snip.items()
    }
    results = {
        "model": model,
        "median_per_snippet": serialisable_medians,
        "pre_prune_presence": pre_prune_presence,
        "post_prune_presence": post_prune_presence,
        "veto_pruned": veto_pruned,
        "predictions": predictions,
        "pass_count": pass_count,
        "fail_count": fail_count,
    }
    (HERE / f"tier1c_{model}_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    status["phase"] = "complete"
    status["pass_count"] = pass_count
    status["fail_count"] = fail_count
    write_status(model, status)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["qwen", "haiku", "both"], default="both",
                        nargs="?")
    args = parser.parse_args()
    arms = ["qwen", "haiku"] if args.arm == "both" else [args.arm]

    all_results = {}
    for model in arms:
        all_results[model] = run_arm(model)

    # ---- Final cross-arm comparison ----
    if len(arms) > 1:
        print()
        print("=" * 78)
        print("Tier 1C cross-arm comparison")
        print("=" * 78)
        for model in arms:
            r = all_results[model]
            print(f"  {model}: {r['pass_count']}/{r['pass_count']+r['fail_count']} predictions passed")

    return 0 if all(r["fail_count"] == 0 for r in all_results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
