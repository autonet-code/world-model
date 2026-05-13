#!/usr/bin/env python3
"""Oracle grading + analysis for the substrate-vs-frontier reasoning contest.

Reads the JSONL produced by run_contest.py (one row per question, with
three contestant answers a1_no_context / a2_with_code / a3_substrate),
and asks an Opus oracle (via the Claude Max bridge -- NO API keys) to
blind-grade each row. Then aggregates per-contestant means, bootstrap
95% confidence intervals, per-category means, and paired t-tests, and
emits both:

  - oracle_grades.jsonl: one line per graded question.
  - reasoning_results.json: aggregate stats + success-criterion verdict.

The oracle never sees which contestant produced which answer: per
question we shuffle the order ('A', 'B', 'C') with a deterministic
seed = sha256(question_id), call Opus, then map grades back via the
remembered mapping.

Usage
-----

    python grade_contest.py \\
        --contest-results contest_results.jsonl \\
        --questions questions.jsonl \\
        --out-grades oracle_grades.jsonl \\
        --out-results reasoning_results.json \\
        --status status_grading.json \\
        --grade-cache grade_cache.jsonl \\
        --limit 30
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make autonet substrate + bridge provider importable. Mirror the import
# pattern from train_with_judges.py / run_contest.py exactly.
_AUTONET = Path(r"C:\code\autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))


# ---------------------------------------------------------------------------
# Imports from autonet
# ---------------------------------------------------------------------------

# Bridge provider -- Claude Max auth, NO API keys.
from atn.providers.bridge import BridgeProvider  # type: ignore  # noqa: E402


log = logging.getLogger("grade_contest")


# Contestant slot ids in the JSONL produced by run_contest.py.
CONTESTANTS: Tuple[str, str, str] = (
    "a1_no_context",
    "a2_with_code",
    "a3_substrate",
)


# ---------------------------------------------------------------------------
# Oracle prompts (system prompt is constant across calls so the bridge can
# prompt-cache it after the first request).
# ---------------------------------------------------------------------------

ORACLE_SYSTEM_PROMPT = """You are an impartial technical grader. You will be given:
  - A question about a software codebase
  - Notes describing what a strong answer should reference (the answer key the questioner has in mind, drawn from the actual code)
  - Three candidate answers, labeled A, B, and C in random order

Grade each candidate independently on a 1-5 scale across three axes:
  - correctness: does the answer state things that are actually true about the code?
  - completeness: does it cover the key points the question asks about?
  - accuracy_of_references: when it cites file/function/class/constant names, are they real?

Then provide a single overall score (1-5, integer or one decimal) per candidate.

Critical rules:
  - Grade ONLY against the question and notes. Do NOT favor any candidate based on style, length, hedging, or because it sounds confident.
  - If a candidate cites code references, only credit them if they appear in the notes or are clearly consistent with the codebase as described in the notes.
  - "I don't know" is better than fabrication. Penalize fabrication harshly.
  - Stay terse in your justification -- one sentence per axis is enough.

Respond with ONLY this JSON (no markdown, no preamble):
{
  "A": {"correctness": <1-5>, "completeness": <1-5>, "accuracy_of_references": <1-5>, "overall": <1-5>, "justification": "..."},
  "B": {...same...},
  "C": {...same...}
}"""


def build_oracle_user_prompt(
    *,
    category: str,
    question: str,
    notes: str,
    expected_modules: Any,
    anon_a_answer: str,
    anon_b_answer: str,
    anon_c_answer: str,
) -> str:
    """Build the per-question user prompt for the oracle."""
    if isinstance(expected_modules, list):
        expected_modules_str = ", ".join(str(m) for m in expected_modules)
    else:
        expected_modules_str = str(expected_modules)

    return (
        f"QUESTION (category={category}):\n"
        f"{question}\n\n"
        f"NOTES (what a strong answer should reference):\n"
        f"{notes}\n\n"
        f"EXPECTED MODULES: {expected_modules_str}\n\n"
        f"ANSWER A:\n"
        f"{anon_a_answer}\n\n"
        f"ANSWER B:\n"
        f"{anon_b_answer}\n\n"
        f"ANSWER C:\n"
        f"{anon_c_answer}\n\n"
        f"Grade now."
    )


# ---------------------------------------------------------------------------
# Robust JSON parsing -- tolerant of fences/preamble/postamble.
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_oracle_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse the oracle's response into the {A: {...}, B: {...}, C: {...}}
    structure. Tolerant of: code fences, preamble before/after the JSON
    object, `{...}` embedded inside other text. Returns None on failure.

    Mirrors parse_judge_json from train_with_judges.py.
    """
    if not text:
        return None

    candidate = text.strip()

    # Strip markdown fences if present.
    fence_match = _FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
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

    # Validate shape: must contain A, B, C each with the four numeric fields.
    out: Dict[str, Any] = {}
    for label in ("A", "B", "C"):
        block = parsed.get(label)
        if not isinstance(block, dict):
            return None
        try:
            cleaned: Dict[str, Any] = {
                "correctness": float(block.get("correctness")),
                "completeness": float(block.get("completeness")),
                "accuracy_of_references": float(block.get("accuracy_of_references")),
                "overall": float(block.get("overall")),
                "justification": str(block.get("justification", "")),
            }
        except (TypeError, ValueError):
            return None
        # Bound-check 1..5 (allow halves via float).
        for k in ("correctness", "completeness", "accuracy_of_references", "overall"):
            v = cleaned[k]
            if not (1.0 <= v <= 5.0):
                # Don't reject -- clamp. Fabricated 0/6 happens; clip and keep.
                cleaned[k] = max(1.0, min(5.0, v))
        out[label] = cleaned

    return out


# ---------------------------------------------------------------------------
# Caching: oracle responses keyed by sha256(qid + "|" + a + "|" + b + "|" + c)
# ---------------------------------------------------------------------------


def cache_key(question_id: str, anon_a: str, anon_b: str, anon_c: str) -> str:
    payload = (
        question_id + "|" + (anon_a or "") + "|" + (anon_b or "") + "|" + (anon_c or "")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_grade_cache(path: Path) -> Dict[str, Dict[str, Any]]:
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
        log.warning("failed to load grade cache %s: %s", path, e)
    return out


def append_grade_cache(path: Path, entry: Dict[str, Any]) -> None:
    """Append a single entry to the JSONL cache file."""
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("failed to append grade cache %s: %s", path, e)


# ---------------------------------------------------------------------------
# Anonymization: deterministic per-question shuffle of the contestant slots.
# ---------------------------------------------------------------------------


def anonymize(question_id: str) -> Dict[str, str]:
    """Return a {anon_label: contestant_id} mapping that's deterministic in
    question_id. The mapping is built by shuffling the three contestant ids
    using a Random seeded with sha256(question_id) and pairing them with
    ('A', 'B', 'C') in order.
    """
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:8]
    rng = random.Random(int(digest, 16))
    contestants = list(CONTESTANTS)
    rng.shuffle(contestants)
    return {"A": contestants[0], "B": contestants[1], "C": contestants[2]}


# ---------------------------------------------------------------------------
# Question loaders
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("skipping malformed JSONL line: %s", e)
                continue
    return out


def index_questions_by_id(questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index questions.jsonl by question id so we can look up `notes`
    and `expected_modules` even if they're not echoed in contest_results.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for q in questions:
        qid = q.get("id") or q.get("question_id")
        if qid:
            out[qid] = q
    return out


# ---------------------------------------------------------------------------
# Oracle bridge call
# ---------------------------------------------------------------------------


async def call_oracle(
    provider: BridgeProvider,
    *,
    user_prompt: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Call the Opus oracle. Returns (parsed_or_None, raw_response_text)."""
    try:
        result = await provider.send(
            messages=[{"role": "user", "content": user_prompt}],
            system=ORACLE_SYSTEM_PROMPT,
            model="opus",
        )
    except Exception as e:
        log.warning("bridge.send() failed: %s", e)
        return None, ""

    text = result.text or ""
    parsed = parse_oracle_json(text)
    return parsed, text


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float], ddof: int = 1) -> float:
    """Sample std dev with Bessel's correction by default. Returns 0.0 if
    fewer than (ddof + 1) samples (no variance to report)."""
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = _mean(xs)
    s2 = sum((x - m) ** 2 for x in xs) / (n - ddof)
    return math.sqrt(s2)


def bootstrap_ci(
    xs: List[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0xCAFE,
) -> Tuple[float, float]:
    """Bootstrap a (1 - alpha) CI for the mean of xs.

    Algorithm:
        1. Draw n_resamples samples WITH replacement, each of size len(xs).
        2. Take the mean of each resample -> bootstrap distribution.
        3. CI = (alpha/2 percentile, 1 - alpha/2 percentile).

    Returns (lower, upper). If xs is empty, returns (0.0, 0.0).
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(xs[0]), float(xs[0])

    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(n_resamples):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()

    # Percentile interpolation -- nearest rank for simplicity.
    lo_idx = int(math.floor((alpha / 2.0) * n_resamples))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_resamples)) - 1
    lo_idx = max(0, min(n_resamples - 1, lo_idx))
    hi_idx = max(0, min(n_resamples - 1, hi_idx))
    return float(means[lo_idx]), float(means[hi_idx])


# Try to use scipy for the paired t-test if available; otherwise a hand-
# rolled implementation kicks in. Both yield equivalent results within
# float precision for paired samples.
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover -- scipy may not be installed
    _scipy_stats = None
    _HAVE_SCIPY = False


def _student_t_sf(t: float, df: float) -> float:
    """Survival function (one-sided P(T > t)) of Student's t with df.

    Uses the regularized incomplete beta relation:
        P(|T| >= |t|) = I_{df/(df + t^2)}(df/2, 1/2)
    where I is the regularized incomplete beta. We then halve to get
    the one-sided survival, but for a two-sided p we just return the
    full thing. Caller is expected to interpret accordingly. We provide
    a two-sided helper below.

    This avoids a SciPy dependency for the fallback path.
    """
    # Two-sided p-value from |t|, df.
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    # Regularized incomplete beta I_x(a, b) with a=df/2, b=1/2.
    return _betai(df / 2.0, 0.5, x)


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b). Numerical Recipes-style
    continued fraction. Adequate for p-value computation in our regime
    (df >= 1, |t| not pathological)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def paired_t_test(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Two-sided paired t-test. Returns (mean_diff = mean(x_i - y_i), p_value).

    Formula:
        d_i = x_i - y_i
        t   = mean(d) / (std(d, ddof=1) / sqrt(n))
        df  = n - 1
        p   = 2 * P(T > |t|) for T ~ Student-t(df)

    Uses scipy.stats.ttest_rel when available; otherwise computes t and
    converts to p via the regularized incomplete beta function above.
    """
    if len(xs) != len(ys):
        raise ValueError(f"paired_t_test: unequal lengths {len(xs)} vs {len(ys)}")
    n = len(xs)
    if n < 2:
        return 0.0, 1.0
    diffs = [a - b for a, b in zip(xs, ys)]
    mean_diff = _mean(diffs)

    if _HAVE_SCIPY:
        try:
            res = _scipy_stats.ttest_rel(xs, ys)
            return float(mean_diff), float(res.pvalue)
        except Exception as e:
            log.warning("scipy ttest_rel failed (%s); falling back to hand-rolled", e)

    sd = _std(diffs, ddof=1)
    if sd == 0.0:
        # All diffs identical. p=0 if mean != 0, else p=1.
        return float(mean_diff), 0.0 if mean_diff != 0.0 else 1.0
    t = mean_diff / (sd / math.sqrt(n))
    df = n - 1
    p = _student_t_sf(abs(t), df)
    p = max(0.0, min(1.0, p))
    return float(mean_diff), float(p)


def interpret_diff(label_x: str, label_y: str, mean_diff: float, p_value: float) -> str:
    """One-line interpretation for the paired_comparisons block."""
    direction = "above" if mean_diff > 0 else ("below" if mean_diff < 0 else "tied with")
    if p_value < 0.05:
        sig = "significantly"
    elif p_value < 0.10:
        sig = "marginally"
    else:
        sig = "not significantly"
    short_x = label_x.split("_")[0]
    short_y = label_y.split("_")[0]
    return f"{short_x} {sig} {direction} {short_y}"


# ---------------------------------------------------------------------------
# Status writer (mirror run_contest.py's StatusWriter shape)
# ---------------------------------------------------------------------------


class StatusWriter:
    def __init__(self, path: Path, n_questions: int) -> None:
        self._path = path
        self._state: Dict[str, Any] = {
            "started_at": time.time(),
            "phase": "grading",
            "n_questions_total": n_questions,
            "current_question": 0,
            "oracle_calls_made": 0,
            "oracle_cache_hits": 0,
            "oracle_parse_failures": 0,
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
# Main grading + analysis pipeline
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    started_at = time.time()

    contest_path = Path(args.contest_results)
    if not contest_path.exists():
        print(
            f"  ERROR: contest results not found: {contest_path}",
            file=sys.stderr,
        )
        return 2

    questions_path = Path(args.questions)
    if not questions_path.exists():
        # Not strictly required: contest_results already echoes notes,
        # but we prefer the canonical questions.jsonl when available.
        log.warning("questions file not found at %s; using notes from contest_results", questions_path)
        questions_index: Dict[str, Dict[str, Any]] = {}
    else:
        questions_index = index_questions_by_id(load_jsonl(questions_path))

    rows = load_jsonl(contest_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    n_total = len(rows)
    print(f"  loaded {n_total} contest rows from {contest_path}")

    # Output paths.
    out_grades_path = Path(args.out_grades)
    out_results_path = Path(args.out_results)
    grade_cache_path = Path(args.grade_cache)
    status_path = Path(args.status)

    # Truncate grades file at start so re-runs don't accumulate. Cache
    # lives across runs.
    out_grades_path.write_text("", encoding="utf-8")

    cache = load_grade_cache(grade_cache_path)
    if cache:
        print(f"  loaded {len(cache)} grade cache entries from {grade_cache_path}")
    else:
        print(f"  no existing grade cache (will write to {grade_cache_path})")

    status = StatusWriter(status_path, n_questions=n_total)

    print(
        f"\n  This will make up to ~{n_total} bridge calls to Opus "
        f"against your Claude Max subscription "
        f"(cached grades reused, null-answer rows skipped).\n"
    )

    provider = BridgeProvider(model="opus")

    # Per-row grading loop.
    graded_rows: List[Dict[str, Any]] = []
    skipped_null = 0

    try:
        for i, row in enumerate(rows, start=1):
            qid = row.get("question_id") or row.get("id") or f"q{i:02d}"
            category = row.get("category", "")
            question = row.get("question", "") or ""

            # Pull notes / expected_modules from canonical questions index
            # if available; otherwise fall back to whatever the contest row
            # echoed.
            qrec = questions_index.get(qid, {})
            notes = qrec.get("notes") or row.get("notes", "") or ""
            expected_modules = (
                qrec.get("expected_modules")
                if qrec.get("expected_modules") is not None
                else row.get("expected_modules", [])
            )

            status.update(current_question=i)
            status.write()

            # Pull answers.
            answers: Dict[str, Optional[str]] = {}
            for cid in CONTESTANTS:
                blk = row.get(cid) or {}
                ans = blk.get("answer") if isinstance(blk, dict) else None
                answers[cid] = ans if isinstance(ans, str) and ans.strip() else None

            if any(v is None for v in answers.values()):
                missing = [k for k, v in answers.items() if v is None]
                log.warning(
                    "[%s/%s] %s: skipping (missing answers: %s)",
                    i, n_total, qid, ", ".join(missing),
                )
                skipped_null += 1
                continue

            # Anonymize.
            mapping = anonymize(qid)  # {"A": cid, "B": cid, "C": cid}
            anon_a = answers[mapping["A"]] or ""
            anon_b = answers[mapping["B"]] or ""
            anon_c = answers[mapping["C"]] or ""

            user_prompt = build_oracle_user_prompt(
                category=category,
                question=question,
                notes=notes,
                expected_modules=expected_modules,
                anon_a_answer=anon_a,
                anon_b_answer=anon_b,
                anon_c_answer=anon_c,
            )

            key = cache_key(qid, anon_a, anon_b, anon_c)
            cached = cache.get(key)

            parsed: Optional[Dict[str, Any]] = None
            raw_text: str = ""
            if cached and isinstance(cached.get("parsed"), dict):
                parsed = cached["parsed"]
                raw_text = cached.get("response_text", "")
                status.bump("oracle_cache_hits")
            else:
                parsed, raw_text = await call_oracle(provider, user_prompt=user_prompt)
                status.bump("oracle_calls_made")
                if parsed is None:
                    status.bump("oracle_parse_failures")
                    log.warning(
                        "[%s/%s] %s: oracle parse failed; raw=%r (skipping)",
                        i, n_total, qid,
                        raw_text[:200] if raw_text else "<empty>",
                    )
                    status.write()
                    continue
                # Cache the successful parse.
                entry = {
                    "key": key,
                    "question_id": qid,
                    "anonymization": mapping,
                    "response_text": raw_text,
                    "parsed": parsed,
                    "model": "opus",
                }
                cache[key] = entry
                append_grade_cache(grade_cache_path, entry)

            # Map anonymized grades back to contestant grades.
            grades_by_contestant: Dict[str, Dict[str, Any]] = {}
            for anon_label, cid in mapping.items():
                grades_by_contestant[cid] = parsed[anon_label]

            out_row: Dict[str, Any] = {
                "question_id": qid,
                "category": category,
                "anonymization": mapping,
                "grades": grades_by_contestant,
                "raw_oracle_response": raw_text,
            }

            with out_grades_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            graded_rows.append(out_row)
            status.write()

            print(
                f"  [{i}/{n_total}] {qid} ({category}) "
                f"a1={grades_by_contestant['a1_no_context']['overall']} "
                f"a2={grades_by_contestant['a2_with_code']['overall']} "
                f"a3={grades_by_contestant['a3_substrate']['overall']}"
            )

    except Exception as e:
        log.exception("grading loop failed")
        status.update(phase="failed")
        status.write()
        raise
    finally:
        try:
            await provider.close()
        except Exception as e:
            log.warning("provider.close() failed: %s", e)

    # ---------- Analysis ----------
    status.update(phase="analyzing")
    status.write()

    n_graded = len(graded_rows)
    print(f"\n  graded {n_graded}/{n_total} rows ({skipped_null} skipped for null answers)")

    results = build_aggregate_results(
        graded_rows=graded_rows,
        n_total=n_total,
        n_graded=n_graded,
    )

    with out_results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    status.update(phase="complete")
    status.write()

    elapsed = time.time() - started_at
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  questions:        {n_total}")
    print(f"  graded:           {n_graded}")
    print(f"  null-skipped:     {skipped_null}")
    print(f"  oracle calls:     {status._state['oracle_calls_made']}")
    print(f"  cache hits:       {status._state['oracle_cache_hits']}")
    print(f"  parse failures:   {status._state['oracle_parse_failures']}")
    print(f"  elapsed:          {elapsed:.1f}s")
    print(f"  grades:           {out_grades_path}")
    print(f"  results:          {out_results_path}")
    print(f"  status:           {status_path}")
    print(f"  cache:            {grade_cache_path}")

    return 0


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------


def build_aggregate_results(
    *,
    graded_rows: List[Dict[str, Any]],
    n_total: int,
    n_graded: int,
) -> Dict[str, Any]:
    """Build the reasoning_results.json payload from graded_rows."""

    # Per-contestant flat lists (parallel to graded_rows order so paired
    # tests work).
    flat: Dict[str, Dict[str, List[float]]] = {
        cid: {
            "overall": [],
            "correctness": [],
            "completeness": [],
            "accuracy_of_references": [],
        }
        for cid in CONTESTANTS
    }
    for row in graded_rows:
        grades = row.get("grades", {})
        for cid in CONTESTANTS:
            g = grades.get(cid, {})
            for axis in ("overall", "correctness", "completeness", "accuracy_of_references"):
                v = g.get(axis)
                if v is None:
                    # Should not happen given validation in parse_oracle_json,
                    # but guard anyway.
                    continue
                flat[cid][axis].append(float(v))

    # by_contestant: means + bootstrap CI on overall.
    by_contestant: Dict[str, Dict[str, Any]] = {}
    for cid in CONTESTANTS:
        overall = flat[cid]["overall"]
        ci_lo, ci_hi = bootstrap_ci(overall)
        by_contestant[cid] = {
            "mean_overall": _mean(overall),
            "mean_correctness": _mean(flat[cid]["correctness"]),
            "mean_completeness": _mean(flat[cid]["completeness"]),
            "mean_accuracy": _mean(flat[cid]["accuracy_of_references"]),
            "ci_overall_lower": ci_lo,
            "ci_overall_upper": ci_hi,
        }

    # by_category: per-category mean overall per contestant.
    cat_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in graded_rows:
        cat = row.get("category", "") or "uncategorized"
        cat_buckets.setdefault(cat, []).append(row)

    by_category: Dict[str, Dict[str, Any]] = {}
    for cat, rows_in_cat in cat_buckets.items():
        a1_vals: List[float] = []
        a2_vals: List[float] = []
        a3_vals: List[float] = []
        for row in rows_in_cat:
            grades = row.get("grades", {})
            a1_vals.append(float(grades["a1_no_context"]["overall"]))
            a2_vals.append(float(grades["a2_with_code"]["overall"]))
            a3_vals.append(float(grades["a3_substrate"]["overall"]))
        by_category[cat] = {
            "n": len(rows_in_cat),
            "a1_mean": _mean(a1_vals),
            "a2_mean": _mean(a2_vals),
            "a3_mean": _mean(a3_vals),
        }

    # Paired comparisons.
    a1_overall = flat["a1_no_context"]["overall"]
    a2_overall = flat["a2_with_code"]["overall"]
    a3_overall = flat["a3_substrate"]["overall"]

    a3_vs_a1_diff, a3_vs_a1_p = paired_t_test(a3_overall, a1_overall)
    a3_vs_a2_diff, a3_vs_a2_p = paired_t_test(a3_overall, a2_overall)
    a2_vs_a1_diff, a2_vs_a1_p = paired_t_test(a2_overall, a1_overall)

    paired = {
        "a3_vs_a1": {
            "mean_diff": a3_vs_a1_diff,
            "p_value": a3_vs_a1_p,
            "interpretation": interpret_diff("a3", "a1", a3_vs_a1_diff, a3_vs_a1_p),
        },
        "a3_vs_a2": {
            "mean_diff": a3_vs_a2_diff,
            "p_value": a3_vs_a2_p,
            "interpretation": interpret_diff("a3", "a2", a3_vs_a2_diff, a3_vs_a2_p),
        },
        "a2_vs_a1": {
            "mean_diff": a2_vs_a1_diff,
            "p_value": a2_vs_a1_p,
            "interpretation": interpret_diff("a2", "a1", a2_vs_a1_diff, a2_vs_a1_p),
        },
    }

    # Success criterion: a3 within 0.5 of a2 AND a3 significantly above a1.
    a3_minus_a2 = a3_vs_a2_diff           # a3 - a2 (signed)
    a3_minus_a1 = a3_vs_a1_diff           # a3 - a1 (signed)
    within_half_of_a2 = a3_minus_a2 >= -0.5
    sig_above_a1 = (a3_vs_a1_p < 0.05) and (a3_minus_a1 > 0)
    passed = bool(within_half_of_a2 and sig_above_a1)

    if passed:
        verdict = (
            "Substrate (a3) lands within 0.5 of frontier-with-code (a2) "
            "AND is significantly above no-context (a1) -- success criterion met."
        )
    else:
        # Try to be specific about WHY it failed.
        bits: List[str] = []
        if not within_half_of_a2:
            bits.append(
                f"a3 trails a2 by {-a3_minus_a2:.2f} (>0.5 gap)"
            )
        if not sig_above_a1:
            if a3_minus_a1 <= 0:
                bits.append(
                    f"a3 is at or below a1 (diff={a3_minus_a1:+.2f})"
                )
            else:
                bits.append(
                    f"a3 above a1 by {a3_minus_a1:.2f} but p={a3_vs_a1_p:.3g} not <0.05"
                )
        verdict = (
            "Success criterion NOT met: "
            + "; ".join(bits) + "."
        )

    success = {
        "claim": "a3 within 0.5 of a2 AND a3 significantly above a1",
        "a3_minus_a2": a3_minus_a2,
        "a3_minus_a1": a3_minus_a1,
        "a3_a1_p_value": a3_vs_a1_p,
        "passed": passed,
        "verdict": verdict,
    }

    return {
        "n_questions": n_total,
        "n_graded": n_graded,
        "by_contestant": by_contestant,
        "by_category": by_category,
        "paired_comparisons": paired,
        "success_criterion": success,
        "stats_engine": "scipy.stats.ttest_rel" if _HAVE_SCIPY else "hand_rolled_paired_t",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Oracle (Opus via Claude Max bridge) blind-grades the three "
            "contestant answers per question in contest_results.jsonl, "
            "then aggregates means, bootstrap CIs, and paired t-tests. "
            "NO API keys -- uses the Claude Max bridge."
        )
    )
    parser.add_argument(
        "--contest-results",
        default="contest_results.jsonl",
        help="JSONL produced by run_contest.py (default: contest_results.jsonl).",
    )
    parser.add_argument(
        "--questions",
        default="questions.jsonl",
        help=(
            "Original questions JSONL, used as the canonical source for "
            "`notes` and `expected_modules` (default: questions.jsonl)."
        ),
    )
    parser.add_argument(
        "--out-grades",
        default="oracle_grades.jsonl",
        help="Output JSONL: one graded row per question.",
    )
    parser.add_argument(
        "--out-results",
        default="reasoning_results.json",
        help="Output JSON: aggregate stats + success-criterion verdict.",
    )
    parser.add_argument(
        "--status",
        default="status_grading.json",
        help="Live status file, updated after each question.",
    )
    parser.add_argument(
        "--grade-cache",
        default="grade_cache.jsonl",
        help=(
            "JSONL cache for oracle responses (keyed by sha256 of question_id "
            "+ the three anonymized answers)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Grade only the first N rows (default: 30; 0 = all).",
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
