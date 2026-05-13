# Tier 1γ spec: single-axis substrate (correctness only)

## What this tests

Whether the LLM-substrate loop produces correct verdicts when the
substrate has only one root (correctness, veto-shaped). Same 12
snippets as Tier 1A. The LLM emits a 1-D correctness score; the
substrate vetoes buggy snippets.

## Why this is the right next experiment

Tier 1A's failures could come from two suspects:
  (a) LLM calibration — qwen3.5:4b is generous on stylistic axes.
  (b) Substrate composition — three-root co-parenting introduces
      sub-child accretion that cancels veto contributions for
      mid-magnitude coords.

Tier 1γ isolates them: drop axes b and c, keep only correctness.
qwen is already decisive on correctness sign in Tier 1A's data
(S7=-1, S11=-1, S12=-1, gold=+1). With one axis:
  - There's nothing for co-parenting to bridge (only one tree).
  - There's no sub-child accretion across tendencies.
  - The veto's signed contribution is unambiguous.

If γ passes 5/5, three-root composition is the muddler — and the
fix is either substrate-side (dedup sub-children with same
observation_id) or test-design-side (fewer rounds, stricter
floors). If γ fails, the bottleneck is somewhere else (LLM
single-axis calibration on subtle bugs like S8, or a substrate
issue that surfaces even single-axis).

## Setup

### Substrate

Just one tendency:
  correctness: anchor=(1.0,), axis=(1.0,), bandwidth=1.5,
               veto_shaped=True, veto_score_floor=-0.5,
               novelty_gamma_pro=1.0, novelty_gamma_con=1.5

1-D coordinate space. A snippet's coord is `(c,)` where c is the
correctness score from the LLM.

### LLM call

Same shape as Tier 1A but a simpler prompt asking only for
correctness. Reuse the cache from Tier 1A by extracting the
correctness field from each cached parsed sample. Median across
the same 3 samples per snippet. NO new LLM calls needed.

### Substrate run

N_ROUNDS = 1 (consistent with Tier 1A's lesson). All 12 snippets
fire as observations once. Then prune_veto_negatives.

### Predictions

Three predictions. Smaller surface than Tier 1A because we're
testing one axis only.

  **G1 (gold survives correctness):** S1, S2 (median c=+1.0)
  remain in correctness's tree post-prune.

  **G2 (clearly buggy vetoed):** S7 (median c=-1.0), S11 (-1.0),
  S12 (-1.0) all absent from correctness's tree post-prune.

  **G3 (subtle/quirky/complex survive):** S3, S4 (quirky, c=+1.0),
  S5, S6 (complex, c=+1.0), S8 (subtle bug, c=+0.5), S9, S10
  (narrow, c=+1.0) all present in correctness's tree post-prune.

S8 is the interesting borderline case: qwen rated it +0.5 not
clearly negative (because `is 0` often "works"), so the substrate
should treat it as PRO of correctness — present, not vetoed.
That's the LLM's call, and we honor it.

## Comparison shape

After γ runs, compute the "two-axis-vs-one-axis" delta:

  | snippet | Tier 1A correctness-presence | Tier 1γ correctness-presence |
  | S1      | T                            | T                            |
  | S7      | T (was supposed to be F)     | F (vetoed)                   |
  | S11     | T (was supposed to be F)     | ?                            |
  | etc.

If S7 was already correctly vetoed in Tier 1A and γ matches that,
the lesson is "Tier 1A was already right on this case." If S11
was-not-vetoed in Tier 1A but IS-vetoed in γ, that's strong
evidence the three-root composition is what cancelled S11's
veto contribution.

## Outputs

  - `tier1g_status.json`
  - `tier1g_results.json` — per-snippet (correctness coord, post-
    prune presence, was-it-also-vetoed-in-Tier-1A flag)
  - PASS/FAIL per prediction at end
  - Optional: brief textual comparison block in the run output

## Implementation notes

- Reuse `tier1a_snippets.py` directly.
- Reuse `tier1a_llm_cache.jsonl` directly — extract `parsed["correctness"]`
  from each sample to derive 1-D coords. NO new LLM calls.
- New file `run_tier1g.py` modeled on `run_tier1a.py` but with
  single-tendency substrate and three predictions.
- Use the same intrinsic_score_in_tendency-aware veto-prune.

## Estimated effort

- Spec: done (this doc).
- Runner: ~30 minutes (it's a strict subset of run_tier1a.py).
- Run: <1 minute (no LLM calls).
- Diagnose: 15 minutes.

Total: ~45 minutes from spec-OK to verdict.

## Success / failure / unknown

  - **G1, G2, G3 all pass:** the LLM-substrate loop works on a
    single decisive axis. Three-root composition is what added the
    noise in Tier 1A. Next experiment is more focused: test what
    breaks when a second root is added.
  - **G2 fails (some buggy not vetoed):** even single-axis the
    veto isn't reliable. Likely substrate-side: a sub-child of the
    buggy node's CON child is cancelling the contribution. Need
    to investigate sprout deduplication.
  - **G3 fails (something gets vetoed that shouldn't):** the floor
    is too tight, or the substrate is mis-classifying mid-magnitude
    PRO observations as CON. Configuration issue.
  - **G1 fails:** something is fundamentally wrong with the
    single-tendency setup. This shouldn't happen — Tier 0 used
    a similar shape and passed.
