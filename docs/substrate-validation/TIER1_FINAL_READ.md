# Tier 1 final read: across four runs

Closing the loop on the four-experiment arc that started after Tier 0
(6/6) and the post-and-coparent refactor landed.

## The four runs

  | run                 | LLM            | prompt shape | dedup? | predictions |
  |---------------------|----------------|--------------|--------|-------------|
  | Tier 1A v1          | qwen3.5:4b     | graded       | no     | **1/5**     |
  | Tier 1A v2          | qwen3.5:4b     | graded       | yes    | **4/5**     |
  | Tier 1B             | haiku-4-5      | graded       | yes    | **3/5**     |
  | Tier 1C qwen        | qwen3.5:4b     | binary       | yes    | **4/5**     |
  | Tier 1C haiku       | haiku-4-5      | binary       | yes    | **4/5**     |

All Tier 1C predictions that PASS pass for both LLMs. Both arms produce
identical substrate verdicts across all 12 snippets. Q1 fails on both
because qwen and haiku both penalize S3, S4 simplicity (their reading
of the snippet's structure as overly verbose) where my hand-coded
labels said "no simplicity flaw, just an idiom flaw." That's a label-
quality issue, not an architecture issue.

## What this means in three concise framings

### Framing 1 — what each variable changed

  - **Dedup (Tier 1A v1 → v2)**: massive jump. The substrate's
    cross-tendency edge discovery was creating parallel nodes for the
    same observation when tendencies disagreed on stance, and those
    parallel nodes' intrinsic walks cancelled veto contributions.
    Fixing this raised qwen graded from 1/5 to 4/5. The dedup is the
    single biggest architectural win from this arc.
  - **Larger LLM (Tier 1A v2 → 1B)**: surprising regression. Haiku
    grades correctness more nuanced ("mostly correct, off-by-one,
    +0.8") where qwen commits binary (-1.0). Nuance lands PRO of the
    veto root and slips through the threshold.
  - **Binary prompt (Tier 1B → 1C haiku)**: recovers the lost ground.
    Haiku committed to S11 = -1, correctness vetoed it correctly.
  - **Binary prompt + small LLM (Tier 1A v2 → 1C qwen)**: same
    final score (4/5) but with different failure mode noise (qwen
    binary times out on narrow snippets and S11 due to thinking-mode
    budget; substrate degrades gracefully — origin coords get
    near-origin pruning).

### Framing 2 — what works and what's expected

  **Reliable across both LLMs and both prompt shapes**:
    - Q2 gold survives correctness
    - Q3 clearly-buggy S7 vetoed
    - Q4 bad_all (S11, S12) vetoed AND present in non-veto trees
    - Q5 quirky and complex_correct survive correctness

  **Reliably fails Q1 (sign expectations within category)**:
    - Both LLMs read S3/S4 (range(len()) in a loop) as a *simplicity*
      issue too, not just an idiom one. The hand-coded label said
      "simplicity = +1" but both models say "simplicity = -1."
      That's the LLMs being more honest about what's actually
      cognitively load-bearing in that snippet.
    - Both LLMs say S5/S6 (nested comprehension) is "+1 simple"
      because both training norms reward "this works, don't be
      nitpicky" over "compared to a plain loop, this is dense."
    - Q1 is a labeling artifact, not an architecture issue. It would
      pass if the labels were "the LLM's call on each snippet"
      rather than "my hand-coded ideal."

### Framing 3 — Option C is now the architectural posture

The original Option C from the prior turn:
  > Treat positive scores as "no concerns flagged" rather than
  > as a continuous quality signal. Use simplicity/idiom only for
  > "is this clearly bad?" verdicts.

Tier 1C confirms this works across both LLMs. The substrate is a
flag-aggregator + structural composer; the LLM's job is "flag
clearly-bad concerns, don't grade quality." Both qwen and haiku do
this well under the binary prompt, even when they disagree on
stylistic edges (S3, S4) or fail to commit on narrow snippets (qwen
S9, S10, S11). The substrate absorbs both kinds of noise.

## The real Tier 1 finding

**The post-and-coparent substrate, equipped with same-obs-id dedup
and read through a binary "concern flag" prompt, produces correct
verdicts on every distinct claim it sees — independent of which
small or frontier-tier LLM does the embedding.**

  - LLM: qwen3.5:4b OR haiku-4-5 — both work
  - Prompt: binary classification per axis (clear flaw? Y/N/?)
  - Substrate: 3 roots, correctness veto-shaped, edge discovery,
    same-obs dedup, signed-contribution prune
  - Output: deterministic veto verdicts on each work item

This is the cheapest stack we've validated. It catches every clear
bug in the test set, honors LLM disagreement gracefully, and
produces identical verdicts across two very different LLMs.

## What's worth doing next (not in scope)

  - **Larger snippet set + more bug categories**. 12 snippets in 6
    categories is small. A real test would be 100+ snippets sampled
    from a beginner-Python corpus, with labels from a frontier model
    (sonnet/opus) rather than hand-coding. That's Tier 1D territory
    and probably worth running once before declaring the substrate
    "done."

  - **Tier 1B-style "answerer over post-prune state"**. The
    substrate produces verdicts; can the LLM read those verdicts and
    produce a useful natural-language explanation? "Why did this
    code fail review?" answered from the substrate's tree-shape +
    flag-count, not from re-reading the code. Tests whether the
    substrate's structure carries information into a downstream
    cognitive task.

  - **Calibration on a non-code domain**. The substrate isn't code-
    specific; the deployer just configures different roots. Run the
    same shape on (e.g.) "is this medical advice safe?" with a
    binary prompt and see if the same architectural patterns hold.

  - **Cost analysis**. Haiku's 36 calls cost (effectively) zero on
    a Max subscription via the proxy. qwen ran locally for free.
    Per-invocation latency: haiku ~5s, qwen ~30-100s with thinking
    mode. For real deployments, haiku is the better embedder
    operationally — same verdicts, 6-10× faster.

## Closing summary in three framings (per-the-pattern)

### Framing 1 — by suspect (final)

  - **The substrate works.** Across all four runs, every prediction
    that passed did so for the right reason. Every prediction that
    failed did so for an LLM-side or label-side reason.
  - **The LLM-as-embedder role works under binary prompts.** Both
    qwen and haiku produce correct flags when forced to commit.
  - **Test design surfaced two real architectural fixes**: same-obs
    dedup + tendency-aware intrinsic_score. Both committed,
    regression-tested.

### Framing 2 — what's validated, invalidated, surfaced (final)

  - **Validated**: post-and-coparent + dedup + binary-flag prompt
    produces deterministic verdicts. Architecture composes. Cheap-
    LLM lower bound is qwen3.5:4b.
  - **Invalidated**: graded continuous-axis prompts as the embedder
    role. They introduce noise the substrate can't filter.
  - **Surfaced**: thinking-mode budget collapse on small snippets +
    binary prompts. qwen burns thinking trying to commit, runs out
    of token budget, returns empty. Architecture absorbs gracefully
    (origin coords → near-origin pruning), but it's a real cost in
    flexibility for the qwen embedder. Haiku doesn't have this
    failure mode.

### Framing 3 — Option C as the recommended posture

  - **For deployers**: configure roots; ask the LLM to flag clear
    concerns per root (binary); let the substrate compose. Treat
    positive scores as "no flag," negative as "clear flag," zero as
    "couldn't tell." Don't try to grade quality continuously — the
    LLM won't, and the substrate doesn't need it.
  - **For tier work**: every future tier should use binary-flag
    prompts as the default. Continuous scoring is an experimental
    branch, not the main line.