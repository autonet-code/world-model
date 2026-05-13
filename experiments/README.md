# experiments

Runnable validation experiments accumulated over the substrate's
development arc. Findings docs and specs live next to the engine in
`docs/substrate-validation/`; this folder holds the **runnable code,
result JSONs, plots, and synthetic LLM caches** needed to re-run or
audit them.

Many scripts reference `C:\code\autonet` and `C:\code\world-model`
on `sys.path` — these paths are not normalized, so the scripts run
on the original author's machine. Treat them as historical record +
re-runnable on a configured machine, not as portable.

## Layout

### `vl-jepa-era/`

Pre-substrate experiments from the earlier (1024-dim coord, no
post-and-coparent) architecture.

  - `tier0/` — first tier validation: a hand-built graph + question
    set against the early world-model.
  - `lindblad/` — the original continuous-Lindblad kernel
    exploration arc (Step A Hamiltonian, Step B predictions, Step C
    comparison, Step D parameters, Stage 3 substrate trace,
    novelty refactor). The `equilibrate_continuous` kernel started
    here and was upstreamed into the engine.
  - `reasoning/` — the reasoning contest scaffolding from the
    pre-substrate era (`run_experiment.py`, `run_contest.py`,
    `grade_contest.py`, `train_with_judges.py`,
    `extract_sessions.py`). Conversation-derived inputs and result
    JSONs are intentionally excluded from this folder.

### `phase2-substrate-tiers/`

Tier 0–3B on the post-and-coparent architecture.

  - **Tier 0**: substrate seed time on a synthetic 100-event corpus.
  - **Tier 1A/1B/1C**: LLM-as-judge contests on code snippets,
    different model/prompt combos (haiku vs qwen, free-form vs
    binary-flag).
  - **Tier 1G**: gated arena variant.
  - **Tier 2**: at-scale substrate-native consensus (3/6 by-letter,
    validated by intent).
  - **Tier 3A**: LLM-as-embedder swap-in.
  - **Tier 3B**: 4-root vs 6-root charter (verdict separation, mint
    rank, attention dilution).
  - **A1 reshaped**: predictions-as-actions variant.
  - **dim_sweep.py**: PCA dim measurement (1024 / 512 / 256 / 128 /
    64) on the categorical-separation H4 statistic — picks 64-dim.
  - **ablate_magnitude_collapse.py**: 16-cell ablation grid that
    found the Lindblad-writeback magnitude-collapse bug.
  - **verify_scoped_tier3b.py / verify_integrated_tier3b.py**: smoke
    tests that scoped equilibrate and the integrated engine match
    pre-refactor Tier 3B numbers.

Findings docs are in `docs/substrate-validation/` (Tier 0–3A) and
`docs/substrate-validation/tier3b/` (Tier 3B).

### `phase5-toolz-rename/`

Substrate-vs-RAG value-prop validation on a memorization-defeated
corpus.

  - **Corpus**: toolz functoolz + itertoolz + dicttoolz, identifiers
    renamed to defeat training-memory pattern-match (`compose` →
    `fold_funcs`, `pipe` → `thread_value`, etc.). 44 train / 11
    test, rich docstrings + impl on train, sparse one-line +
    examples on test.
  - **Two tendencies**: `correctness`, `simplicity`.
  - **Four contestants**: `haiku-alone`, `haiku+RAG`,
    `haiku+substrate`, `qwen+substrate`.
  - **Grader**: `doctest_harness.grade_implementation` — runs the
    contestant's code against the function's doctests, scores 0..1.
  - **Result**: `haiku+substrate` beat `haiku+RAG` by +0.14.

Plan: `docs/substrate-validation/phase5-phase6/PHASE5_PLAN.md`.

### `phase6-tendency-scaling/`

Does substrate accuracy scale with tendency count?

  - **Corpus**: Phase 5's corpus expanded with `more_itertools`
    (116 entries, 93 train / 23 test, all renamed).
  - **Ten tendencies**: correctness, simplicity, robustness, purity,
    laziness, composability, type_flexibility, error_clarity,
    efficiency, documentation_fidelity.
  - **Pipeline**:
      1. `build_corpus.py` — toolz + more_itertools rename + split.
      2. `run_judges.py` — sonnet returns 10-axis sub-claims per
         function, cached to `judge_cache.jsonl`. Smoke check +
         mid-run coverage guard.
      3. `build_substrate.py --n N` — builds substrate at
         N ∈ {2,4,6,8,10}; truncates the 10-axis cache to active
         subset at sprout time.
      4. `run_contest.py --n N` — `haiku+RAG` vs `haiku+substrate`
         on the 23 test problems, doctest-graded.
      5. `aggregate.json` — final curve.
  - **Result** (aggregate.json):

    | N  | RAG   | substrate | delta  |
    |----|-------|-----------|--------|
    |  2 | 0.448 | 0.729     | +0.280 |
    |  4 | 0.448 | 0.685     | +0.237 |
    |  6 | 0.448 | 0.685     | +0.237 |
    |  8 | 0.468 | 0.671     | +0.203 |
    | 10 | 0.448 | 0.743     | +0.295 |

    Substrate beats RAG at every N. n_test = 23 means ±0.05 is
    within noise; the +0.20 to +0.30 band is directional, not
    statistically airtight. Curve is plateau-with-spike, not
    cleanly monotonic.

Plan: `docs/substrate-validation/phase5-phase6/PHASE6_PLAN.md`.

### `specs/`

Standalone spec documents that don't fit a single phase:

  - `LINDBLAD_SCAFFOLD.md` — the original Lindblad exploration spec.
  - `POST_AUTONET_FINDINGS.md` — handover notes from the
    "consolidate into autonet" phase, including the O(N²)
    equilibrate blocker that motivated scoped equilibrate.
  - `REASONING_EXPERIMENT_PLAN.md` — pre-substrate reasoning
    contest plan (precursor to Phase 5).

## Excluded from this folder

- Conversation-derived JSONL caches and result files from the
  pre-substrate reasoning experiments (root-level
  `judge_cache.jsonl`, `smoke_cache.jsonl`, `grade_cache_v3.jsonl`,
  `oracle_grades_v3.jsonl`, `contest_results*.jsonl`,
  `work_units_*.jsonl`). These contain extracts from real Claude
  sessions and are kept local-only.
- `__pycache__/` and other build artifacts.
