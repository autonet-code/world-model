# Tier 1B vs Tier 1A: haiku vs qwen3.5:4b as embedder

## Headline result

  | metric                    | Tier 1A (qwen3.5:4b) | Tier 1B (haiku-4-5) |
  |---------------------------|----------------------|---------------------|
  | predictions passed        | **4/5**              | **3/5**             |
  | total runtime             | ~25 min (3K tokens)  | ~3 min (cache-read) |
  | per-call avg               | ~30s                 | ~5s                 |
  | sample-to-sample variance | low                  | low (verbose rats)  |

Haiku is faster, more verbose in rationales, and more nuanced —
which sounds better but turns out to be slightly worse for the
*substrate-as-judge* design. The substrate uses thresholds and
vetoes; it works better when the LLM commits to decisive negative
scores than when it gives graded "mostly correct, slight flaw"
ratings.

## Three concise framings of what changed

### Framing 1 — by snippet

  | snippet | qwen correctness | haiku correctness | substrate verdict |
  |---------|------------------|-------------------|-------------------|
  | S7 (off-by-one) | -1.0 | -1.0 | both veto ✓ |
  | S8 (`is 0`)     | +0.5 | -0.8 | qwen survived (PRO), haiku vetoed |
  | S11 (bubble OBO)| -1.0 | +0.8 | qwen vetoed, haiku survived (PRO) |
  | S12 (factorial) | -1.0 | -1.0 | both veto ✓ |
  | S3, S4 (quirky) | idiom +0.5 | idiom -0.5/-0.8 | both survive correctness |
  | S5, S6 (complex)| simplicity +1 | simplicity +1 | both survive (no improvement) |

Net: haiku catches S8 better but misses S11. They trade. On the
non-correctness axes, haiku improves on idiom (gives decisive
negative for non-pythonic code) but is identical on simplicity
(both refuse to call nested comprehensions "complex").

### Framing 2 — by capability

  - **Correctness sign on clear bugs**: both decisive. Tie.
  - **Correctness sign on subtle bugs**: haiku catches S8
    (`is 0`); qwen calls it +0.5. Haiku wins.
  - **Correctness magnitude calibration**: qwen is binary
    (-1.0 / +1.0 mostly); haiku is graded (-1.0, -0.8, +0.8,
    +0.5). For a substrate with veto thresholds, **qwen's
    binary stance is more useful** because graded ratings can
    slip through the veto by landing PRO at +0.8 instead of
    CON at -0.5.
  - **Idiom calibration**: haiku reaches the negative end
    cleanly (-0.3 to -0.8); qwen tops out at +0.5. Haiku wins.
  - **Simplicity calibration**: both reluctant to give negative
    scores for "complex but working" code. Tie at the unhelpful
    ceiling.

### Framing 3 — what we learned about model size vs training norms

  Going up from a 4B local model to a frontier-trained Haiku:

  - **fixed**: idiom calibration (the model is willing to grade
    pythonic-vs-not on a real scale).
  - **fixed**: subtle-bug detection (S8 `is 0` caught).
  - **NOT fixed**: simplicity calibration. Both models call
    nested comprehensions "+1.0 simple" because they're both
    trained to be helpful and not nitpicky about working code.
    This is a *training norm*, not a *model size* issue.
  - **introduced**: graded correctness scores. Haiku will say
    "correctness=+0.8, has a bug but mostly works" where qwen
    just says -1.0. The substrate doesn't reward this nuance
    -- it reads PRO and survives the veto.

## Two implications

### Implication 1 — the substrate is a binary classifier in disguise

The veto-prune mechanism asks one question per direct child of a
veto root: "is this signed contribution below the floor?" Yes ->
prune. No -> survive. **The interesting cognitive work is:
which side of the floor does this end up on?** Continuous
gradation between -0.5 and -1.0 doesn't matter once you cross.
Continuous gradation between +0.5 and +0.8 doesn't matter at all.

If we're building this substrate to run on small / cheap models,
we should embrace this. Don't ask the LLM "rate this 0-1"; ask
"is this bad? (-1 if clearly so, +1 otherwise)." Decisive
classifications produce decisive verdicts.

### Implication 2 — Option C is the right architectural posture

Recall Option C from the prior turn: treat positive scores as
"no concerns flagged" rather than as a continuous quality
signal. With Tier 1B's data:
  - haiku's S11=+0.8 ("mostly correct, with a bug") is the
    failure mode of "I notice a problem but won't commit."
  - qwen's S8=+0.5 ("works for small ints") was the same shape
    -- LLM hedging on a borderline case, substrate honored it.

Both qwen and haiku show this. **Hedging is a property of the
embedder role**, not of model size. The architecture should
either commit to the binary posture (force the LLM into
flag/no-flag) or accept that the substrate is fuzzy on
borderlines.

## Recommended pivot

**Reshape the prompt to be binary** for the next iteration:

  Instead of "score correctness on -1..+1", ask: "does this
  snippet have a clear correctness flaw? Return correctness=-1
  if YES, +1 if NO, 0 only if the snippet is too small to tell."

Same for simplicity and idiom: "does this snippet have a clear
simplicity / idiom flaw?"

This forces the LLM to commit. It probably loses some of haiku's
nuance, but it gains substrate determinism. The architecture
then becomes:

  LLM flags concerns (binary per axis) ->
  substrate composes (multi-axis veto with co-parenting) ->
  prune produces verdict (which work items have unconcerned
  correctness)

Which is exactly Option C from the previous turn, now with
empirical support from both qwen and haiku.

## What's NOT worth chasing

  - **Better prompt engineering for graded scores**: hedging is
    structural to LLMs, not a prompt issue. Few-shotting graded
    examples might mask the issue without fixing it; the next
    snippet that doesn't fit the few-shots will hedge again.
  - **Larger model than haiku**: sonnet / opus would probably
    grade more confidently, but the cost goes up and we're
    already past the "small-first" budget. The architectural
    fix (binary prompt, Option C) does the same job for less.

## Estimated effort for the binary-prompt version

  - Rewrite SYSTEM in run_tier1a.py / run_tier1b.py to
    binary classification per axis
  - Re-run on the same 12 snippets (cached if possible)
  - Re-evaluate predictions

  ~30 min if we test only on qwen, ~45 min if we run both qwen
  and haiku for comparison. The cache only helps if we keep the
  prompt structurally identical, which a binary rewrite breaks
  -- expect fresh LLM time.

## Outputs

  - `tier1b_results.json` -- haiku verdicts
  - `tier1b_plot.png` -- 3-D scatter of haiku coords
  - `tier1b_llm_cache.jsonl` -- 36 cached calls, cache_read
    confirmed (Anthropic's 1-hour TTL kept the prompt warm)
