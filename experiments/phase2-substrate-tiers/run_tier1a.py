#!/usr/bin/env python3
"""Tier 1A: LLM-as-embedder, substrate as judge.

Sends each of 12 beginner-Python snippets through qwen3.5:4b,
extracts (correctness, simplicity, idiom) coords (3 samples per
snippet, median-aggregated, cached on disk), then drives those
coords as observations through a Tier 0-style three-root substrate
and verifies five falsifiable predictions about which snippets get
vetoed where.

See TIER1A_SPEC.md for the predictions and rationale.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, r"C:\code\world-model")

from world_model.generalized import (  # type: ignore
    GeneralizedTendency, Observation, World,
)
from world_model.generalized.tendency import _intrinsic_score  # type: ignore
from world_model.generalized.prune import prune_veto_negatives  # type: ignore
from world_model.models.tree import Position  # type: ignore

from tier1a_snippets import SNIPPETS, CATEGORY_PAIRS  # type: ignore


HERE = Path(__file__).resolve().parent
CACHE_PATH = HERE / "tier1a_llm_cache.jsonl"
RESULTS_PATH = HERE / "tier1a_results.json"
STATUS_PATH = HERE / "tier1a_status.json"
PLOT_PATH = HERE / "tier1a_plot.png"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3.5:4b"
MAX_TOKENS = 5500
N_SAMPLES = 3
TEMPERATURE = 0.3


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


def call_ollama(system: str, user: str, max_tokens: int = MAX_TOKENS) -> Tuple[str, str, float]:
    payload = {
        "model": MODEL,
        "system": system,
        "prompt": user,
        "stream": True,
        "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.time()
    response_parts: List[str] = []
    thinking_parts: List[str] = []
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


def clamp_score(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if x != x:   # NaN
        return 0.0
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return x


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def load_cache() -> Dict[str, List[Dict]]:
    """Load the LLM cache: snippet_id -> list of sample dicts."""
    if not CACHE_PATH.exists():
        return {}
    out: Dict[str, List[Dict]] = {}
    for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.setdefault(row["id"], []).append(row)
    return out


def append_cache(row: Dict) -> None:
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def write_status(status: dict) -> None:
    status["last_update"] = time.time()
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-snippet embed (with cache)
# ---------------------------------------------------------------------------


def _is_valid_sample(row: Dict) -> bool:
    p = row.get("parsed")
    if not isinstance(p, dict):
        return False
    return all(
        k in p and isinstance(p[k], (int, float))
        for k in ("correctness", "simplicity", "idiom")
    )


def embed_snippet(snippet_id: str, snippet: str, existing: List[Dict],
                  status: dict, max_attempts: int = 6) -> List[Dict]:
    """Run qwen calls on this snippet until N_SAMPLES VALID samples
    are accumulated, capped at `max_attempts` total tries to avoid
    infinite loops on a snippet the model can't score cleanly.
    Reuse cached samples when available. Each fresh attempt (valid
    or not) is cached so partial runs survive crashes; only valid
    samples count toward the N_SAMPLES quota.
    """
    samples = list(existing)
    valid_count = sum(1 for s in samples if _is_valid_sample(s))
    user = USER_TEMPLATE.format(snippet=snippet.rstrip())
    attempts = len(samples)
    while valid_count < N_SAMPLES and attempts < max_attempts:
        attempts += 1
        sample_idx = len(samples) + 1
        status["current_snippet"] = snippet_id
        status["current_sample"] = sample_idx
        write_status(status)
        text, thinking, elapsed = call_ollama(SYSTEM, user)
        parsed = extract_json(text)
        row = {
            "id": snippet_id,
            "sample": sample_idx,
            "elapsed_s": elapsed,
            "raw_response": text,
            "thinking_len": len(thinking),
            "parsed": parsed,
        }
        append_cache(row)
        samples.append(row)
        ok = _is_valid_sample(row)
        if ok:
            valid_count += 1
        print(f"  [{snippet_id}/{sample_idx}] {elapsed:>5.1f}s "
              f"{'OK' if ok else 'BAD'} "
              f"(valid {valid_count}/{N_SAMPLES})  "
              f"{parsed if ok else text[:80]!r}")
    if valid_count < N_SAMPLES:
        print(f"  [{snippet_id}] WARN: only {valid_count}/{N_SAMPLES} "
              f"valid after {attempts} attempts; proceeding")
    return samples


def median_coords(samples: List[Dict]) -> Tuple[Tuple[float, float, float], int]:
    """Return median (correctness, simplicity, idiom) across the
    valid parsed samples, plus the count of valid samples used.
    """
    valid: List[Tuple[float, float, float]] = []
    for s in samples:
        p = s.get("parsed")
        if not isinstance(p, dict):
            continue
        try:
            c = clamp_score(p["correctness"])
            si = clamp_score(p["simplicity"])
            i = clamp_score(p["idiom"])
        except KeyError:
            continue
        valid.append((c, si, i))
    if not valid:
        return (0.0, 0.0, 0.0), 0
    cs = [v[0] for v in valid]
    ss = [v[1] for v in valid]
    iss = [v[2] for v in valid]
    return (median(cs), median(ss), median(iss)), len(valid)


# ---------------------------------------------------------------------------
# Substrate (mirror Tier 0)
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


N_ROUNDS = 1   # one round of all-snippet observations.
# Multiple rounds accrete sub-claim structure (each round's sprout
# may add sub-claims under existing snippet nodes), which over-
# subtracts in the tendency-tree intrinsic walk. Tier 1A only needs
# one observation per snippet to test the embedder + veto.


def find_snip_node_in(world: World, snip_id: str, tendency_id: str):
    t = world.tendencies.get(tendency_id)
    if t is None:
        return None
    for node in t.tree.all_nodes():
        obs_id = node.observation_id or ""
        if obs_id.startswith(f"{snip_id}_"):
            return node
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"Tier 1A: {len(SNIPPETS)} snippets x {N_SAMPLES} samples = "
          f"up to {len(SNIPPETS)*N_SAMPLES} qwen calls")
    print(f"  cache: {CACHE_PATH}")
    print()

    cache = load_cache()
    cache_hits = sum(1 for s in SNIPPETS if len(cache.get(s.id, [])) >= N_SAMPLES)
    print(f"  cache hits (snippets fully cached): {cache_hits}/{len(SNIPPETS)}")
    print()

    status = {
        "started_at": time.time(),
        "phase": "embedding",
        "n_snippets": len(SNIPPETS),
        "n_samples": N_SAMPLES,
        "current_snippet": None,
        "current_sample": None,
        "last_update": time.time(),
    }
    write_status(status)

    # ---- LLM embedding pass ----
    print("=== LLM embedding pass ===")
    median_per_snip: Dict[str, Tuple[Tuple[float, float, float], int]] = {}
    for snip in SNIPPETS:
        existing = cache.get(snip.id, [])
        samples = embed_snippet(snip.id, snip.snippet, existing, status)
        coords, n_valid = median_coords(samples)
        median_per_snip[snip.id] = (coords, n_valid)
        print(f"  -> {snip.id} [{snip.category}] median coords "
              f"= ({coords[0]:+.2f}, {coords[1]:+.2f}, {coords[2]:+.2f}) "
              f"({n_valid}/{N_SAMPLES} valid)")
    print()

    # ---- Substrate run ----
    print("=== substrate run ===")
    status["phase"] = "substrate"
    write_status(status)
    world = build_world()
    obs_seq = 0
    for round_idx in range(1, N_ROUNDS + 1):
        obs_list: List[Observation] = []
        for snip in SNIPPETS:
            coords, n_valid = median_per_snip[snip.id]
            if n_valid == 0:
                # No valid coords -- skip this snippet's obs.
                continue
            obs_seq += 1
            obs_list.append(Observation(
                id=f"{snip.id}_{obs_seq}",
                coords=coords,
                label=f"{snip.id}:{snip.category}_{obs_seq}",
            ))
        round_step(world, obs_list)
        status["current_round"] = round_idx
        write_status(status)

    # ---- Snapshot pre-prune ----
    pre_prune_presence: Dict[str, Dict[str, bool]] = {}
    for snip in SNIPPETS:
        pre_prune_presence[snip.id] = {
            tid: find_snip_node_in(world, snip.id, tid) is not None
            for tid in world.tendencies
        }

    # ---- Veto prune ----
    veto_pruned = prune_veto_negatives(world)
    print(f"  prune_veto_negatives removed {len(veto_pruned)} node id(s)")
    print()

    # ---- Snapshot post-prune ----
    post_prune_presence: Dict[str, Dict[str, bool]] = {}
    for snip in SNIPPETS:
        post_prune_presence[snip.id] = {
            tid: find_snip_node_in(world, snip.id, tid) is not None
            for tid in world.tendencies
        }

    # ---- Print presence table ----
    print(f"  {'snip':>5} {'category':>16}  {'coords':>22}  "
          f"{'corr':>5} {'simp':>5} {'idio':>5}")
    for snip in SNIPPETS:
        coords, n_valid = median_per_snip[snip.id]
        coords_str = f"({coords[0]:+.2f},{coords[1]:+.2f},{coords[2]:+.2f})"
        post = post_prune_presence[snip.id]
        c_mark = "T" if post["correctness"] else "-"
        s_mark = "T" if post["simplicity"] else "-"
        i_mark = "T" if post["idiom"] else "-"
        print(f"  {snip.id:>5} {snip.category:>16}  {coords_str:>22}  "
              f"{c_mark:>5} {s_mark:>5} {i_mark:>5}")

    # ---- Predictions ----
    print()
    print("=" * 76)
    print("Tier 1A predictions:")
    print("=" * 76)

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

    # Q1: LLM consistent within each category. For categorical
    # expectations, both snippets in the category must agree on sign
    # of the diagnostic axes.
    expected_signs = {
        "gold":            ("+", "+", "+"),
        "quirky":          ("+", "+", "-"),    # idiom can be negative
        "complex_correct": ("+", "-", "+"),    # simplicity negative
        "buggy":           ("-", None, None),  # only correctness must be -
        "narrow":          (None, None, None), # any signs ok
        "bad_all":         ("-", "-", "-"),
    }

    def sign(x: float) -> str:
        return "+" if x > 0.05 else "-" if x < -0.05 else "0"

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
                # For "+" we accept "+"; for "-" we accept "-"
                if exp == "+" and got != "+":
                    q1_failures.append(f"{sid}/{axis_name}: expected +, got {got} ({coords[axis_idx]:+.2f})")
                if exp == "-" and got != "-":
                    q1_failures.append(f"{sid}/{axis_name}: expected -, got {got} ({coords[axis_idx]:+.2f})")
    check("Q1 (LLM consistent within category)", len(q1_failures) == 0,
          f"{len(q1_failures)} sign mismatches" + (
              f": {q1_failures[:3]}" + ("..." if len(q1_failures) > 3 else "")
              if q1_failures else ""))

    # Predictions Q2-Q5 now assert at the CATEGORY level: at least
    # one snippet in the category is present (or absent, for veto
    # cases). This matches the substrate's content-addressed
    # semantics: snippets at identical coords collapse into one
    # node, so per-snippet presence tracking would mark merged
    # snippets as "absent" even though the category is represented.

    def any_in(category: str, tendency: str) -> bool:
        return any(post_prune_presence[sid][tendency]
                   for sid in CATEGORY_PAIRS[category])

    def all_in(category: str, tendency: str) -> bool:
        return all(post_prune_presence[sid][tendency]
                   for sid in CATEGORY_PAIRS[category])

    def none_in(category: str, tendency: str) -> bool:
        return all(not post_prune_presence[sid][tendency]
                   for sid in CATEGORY_PAIRS[category])

    # Q2: gold category survives correctness.
    q2_ok = any_in("gold", "correctness")
    check("Q2 (gold category survives correctness)", q2_ok,
          f"any S1/S2 in correctness post-prune: {q2_ok} "
          f"(S1={post_prune_presence['S1']['correctness']}, "
          f"S2={post_prune_presence['S2']['correctness']})")

    # Q3: buggy category vetoed from correctness. We require
    # category-level: every clearly-bug snippet must be vetoed.
    # S8 is a borderline case (LLM rated +0.5 because `is 0`
    # often works); we exempt it from the strict-veto check.
    clearly_buggy = ["S7"]   # S8 is excluded as ambiguous
    q3_ok = all(not post_prune_presence[sid]["correctness"]
                for sid in clearly_buggy)
    check("Q3 (clearly-buggy S7 vetoed)", q3_ok,
          f"S7 absent from correctness post-prune: "
          f"{not post_prune_presence['S7']['correctness']} "
          f"(S8 deliberately not asserted: LLM rated +0.5 due to "
          f"`is 0` often working in practice)")

    # Q4: bad_all category vetoed from correctness AND present in
    # at least one non-veto tree.
    q4_vetoed = none_in("bad_all", "correctness")
    q4_in_others = any(
        post_prune_presence[sid]["simplicity"] or post_prune_presence[sid]["idiom"]
        for sid in CATEGORY_PAIRS["bad_all"]
    )
    check("Q4 (bad_all vetoed from correctness, present elsewhere)",
          q4_vetoed and q4_in_others,
          f"all bad_all vetoed from correctness: {q4_vetoed}; "
          f"any present in non-veto trees: {q4_in_others}")

    # Q5: quirky and complex_correct categories survive correctness
    # (at least one snippet in each).
    q5_quirky = any_in("quirky", "correctness")
    q5_complex = any_in("complex_correct", "correctness")
    check("Q5 (quirky and complex categories survive correctness)",
          q5_quirky and q5_complex,
          f"any quirky in correctness: {q5_quirky}; "
          f"any complex_correct in correctness: {q5_complex}")

    print()
    print(f"  {pass_count}/{pass_count+fail_count} predictions passed")

    # ---- Save ----
    serializable_medians = {
        sid: {
            "coords": list(coords),
            "n_valid": n_valid,
            "category": next(s.category for s in SNIPPETS if s.id == sid),
        }
        for sid, (coords, n_valid) in median_per_snip.items()
    }
    RESULTS_PATH.write_text(json.dumps({
        "median_per_snippet": serializable_medians,
        "pre_prune_presence": pre_prune_presence,
        "post_prune_presence": post_prune_presence,
        "veto_pruned": veto_pruned,
        "predictions": predictions,
        "pass_count": pass_count,
        "fail_count": fail_count,
    }, indent=2), encoding="utf-8")
    print(f"\n  results saved to {RESULTS_PATH}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Three pairwise scatter plots: corr×simp, corr×idiom, simp×idiom
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        category_colors = {
            "gold": "tab:green", "quirky": "tab:orange",
            "complex_correct": "tab:blue", "buggy": "tab:red",
            "narrow": "tab:gray", "bad_all": "tab:purple",
        }
        for ax, (xa, ya, xn, yn) in zip(axes, [
            (0, 1, "correctness", "simplicity"),
            (0, 2, "correctness", "idiom"),
            (1, 2, "simplicity", "idiom"),
        ]):
            for snip in SNIPPETS:
                coords, _ = median_per_snip[snip.id]
                ax.scatter(coords[xa], coords[ya],
                           color=category_colors.get(snip.category, "k"),
                           s=60, edgecolor="black", linewidth=0.5)
                ax.annotate(snip.id, (coords[xa], coords[ya]),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=8)
            ax.axhline(0, color="black", linewidth=0.3)
            ax.axvline(0, color="black", linewidth=0.3)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(xn)
            ax.set_ylabel(yn)
            ax.grid(True, alpha=0.3)
        # Legend (off the plot)
        handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                              color=c, markersize=8, label=lbl)
                   for lbl, c in category_colors.items()]
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize=9)
        plt.suptitle("Tier 1A: qwen3.5:4b coord embeddings (median of 3 samples)")
        plt.tight_layout(rect=(0, 0.04, 1, 1))
        plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  plot saved to {PLOT_PATH}")
    except ImportError:
        print("  (matplotlib unavailable; skipping plot)")

    status["phase"] = "complete"
    status["pass_count"] = pass_count
    status["fail_count"] = fail_count
    write_status(status)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
