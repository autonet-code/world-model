# Phase 6: tendency-count scaling

## Question

Does graded inference accuracy grow as the substrate's tendency
count grows? Curve at N ∈ {2, 4, 6, 8, 10}.

The toolz-renamed corpus from Phase 5, expanded with more_itertools,
serves as the testbed. The 10 tendencies are toolz-domain (no
alignment axes — those don't engage on a functional library).

## 10 tendencies

| # | name | what it grades |
|---|------|----------------|
| 1 | correctness          | output matches contract |
| 2 | simplicity           | minimal branching, terse |
| 3 | robustness           | handles edge cases (empty, single, mixed types) |
| 4 | purity               | no side effects, no input mutation |
| 5 | laziness             | streaming-friendly, doesn't materialize unnecessarily |
| 6 | composability        | return types pipe cleanly into other functions |
| 7 | type_flexibility     | accepts reasonable input types (list, tuple, gen, dict) |
| 8 | error_clarity        | failures are informative, not silent wrong answers |
| 9 | efficiency           | algorithmic complexity appropriate for task |
| 10| documentation_fidelity | implementation matches what docstring claims |

Subsets per N (cumulative):
  - N=2:  axes 1-2
  - N=4:  axes 1-4
  - N=6:  axes 1-6
  - N=8:  axes 1-8
  - N=10: axes 1-10

## Method

1. Build expanded corpus (toolz + more_itertools, renamed).
2. Run smoke check: 1 sonnet call returns all 10 axes.
3. For each train function: sonnet emits sub-claims on all 10 axes,
   cached. Single judge cache used at every N.
4. For each N: rebuild substrate using only axes 1..N. Same corpus,
   same observations, judge output truncated to active subset.
5. For each held-out test problem (26): two contestants per N:
   - haiku + substrate (production probe)
   - haiku + RAG (top-k by embedding similarity)
6. Score via doctest harness. Compute (substrate - RAG) delta per
   problem, then mean delta per N.
7. Plot delta-vs-N.

## Contestants

Only haiku + substrate and haiku + RAG, since the question is about
the architecture's scaling, not LLM-size viability (Phase 5 covered
the qwen-vs-haiku-alone story).

## Observability — embedded throughout

- `phase6/status.json` — live state (phase, current N, current
  function, sub-claim counts, parse fails, elapsed).
- `phase6/judge_log.jsonl` — append-only one row per sonnet call:
  function name, raw response, parsed result, timing. Mid-run
  readable; grows with progress.
- `phase6/substrate_N{2,4,6,8,10}.json` — snapshot at each substrate
  build: node count, work-item count, axis density distribution.
- `phase6/contest_progress.jsonl` — one row per (N, problem,
  contestant): contestant impl + parse_ok + score. Appended as the
  run proceeds. Partial results survive crashes.

## Fail-fast guards

- Pre-flight smoke check: one sonnet call must return all 10 axes
  with ≥1 sub-claim each. Halt with clear message otherwise.
- Mid-run: if 3 consecutive functions return <50% axis coverage,
  halt and surface raw responses for inspection. Sonnet quality
  drift is a real risk and silent failure is the worst outcome.

## Cost estimate

  - Corpus build + smoke: 30 min
  - Judge call phase: ~130 sonnet calls × ~12s = ~26 min wall
  - 5 substrate builds: ~5 min each (mostly equilibrate)
  - 5 contest runs: ~26 problems × 2 contestants × ~12s = ~10 min each
  - Total: ~2-3 hours wall time
