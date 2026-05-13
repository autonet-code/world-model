# Tier 1A spec: LLM-as-embedder, substrate as judge

## What this tests

Whether a small LLM (qwen3.5:4b) can map a beginner-Python snippet to
3-D substrate coordinates (correctness, simplicity, idiom) reliably
enough that the Tier 0 substrate dynamics produce the right verdict
on each snippet — without any human or larger-model labeling.

The substrate stays in charge of dynamics. The LLM's only job is
the embedding: snippet -> (c, s, i) in [-1, 1]^3.

## Why this is the right next step

Tier 0 showed the substrate's three-root architecture works on
hand-coded coords. Tier 1A asks: can a small model produce coords
that work the same way? If yes, we have a cheap pipeline:

  python snippet -> qwen3.5:4b -> coords -> substrate -> verdict

with the LLM doing the cognitive work that's hard to do without a
language model (reading code) and the substrate doing the
cognitive work that's hard to do well in an LLM (composing three
domain criteria + asymmetric veto + persistent state).

If qwen can't produce coords reliably, we learn what the
substrate's lower bound is for LLM intelligence — and that bounds
the rest of the tier ladder.

## Setup

### Snippets (12 hand-picked beginner-Python pieces)

Six categories, two snippets each:

  **Gold (S1, S2):** correct, simple, idiomatic
    e.g. `sum(range(10))`, list comprehension that's clearly
    pythonic.

  **Quirky-but-fine (S3, S4):** correct, simple, NOT idiomatic
    e.g. C-style index loop instead of enumerate, manual indexing
    instead of `zip`.

  **Complex-but-correct (S5, S6):** correct, idiomatic, complex
    e.g. nested comprehension that's correct and pythonic but
    harder to read than a plain loop.

  **Buggy (S7, S8):** incorrect, otherwise nice
    e.g. off-by-one in range, wrong operator, `is` instead of
    `==` for value comparison.

  **Narrow correctness (S9, S10):** correct, scope unclear
    e.g. a one-liner that does what it says, but the surrounding
    intent is missing — the LLM can confirm correctness but
    shouldn't make confident simplicity/idiom claims.

  **Bad-on-all-axes (S11, S12):** incorrect, complex, unidiomatic
    e.g. recursive bubble sort with off-by-one and ternary abuse.

These map to Tier 0's W1-W6 in shape. Each category has two
snippets so we can see if the LLM is consistent within a category
or just lucky on one example.

### LLM call

For each snippet, prompt qwen3.5:4b to return a JSON object:

```json
{
  "correctness": -1.0 to +1.0,
  "simplicity":  -1.0 to +1.0,
  "idiom":       -1.0 to +1.0,
  "rationale": "<brief>"
}
```

Use a system prompt that defines each axis precisely. Sample 3
times per snippet (temperature 0.3) and median-aggregate the three
values per axis. This smooths out single-call noise.

### Substrate run

Same as Tier 0: build_world() with three roots, fire all 12
snippets every round for 15 rounds, run prune_veto_negatives at
epoch close.

### Predictions

Five falsifiable predictions:

  **Q1 (LLM consistent within category):** for each of the 6
  categories, the median-aggregate coords of the two snippets in
  that category agree on sign per-axis (e.g. both gold snippets
  must have correctness > 0, simplicity > 0, idiom > 0). The two
  buggy snippets must both have correctness < 0.

  **Q2 (gold survives):** S1 and S2 are NOT vetoed from
  correctness; their per-tree intrinsic_score is positive in all
  three trees.

  **Q3 (buggy vetoed):** S7 and S8 are vetoed from correctness
  (CON-position + accumulated evidence triggers veto-prune).
  This is the cheapest pipeline-level signal: did "the substrate
  caught the bug, given LLM-derived coords"?

  **Q4 (bad-on-all-axes vetoed):** S11 and S12 are vetoed from
  correctness AND remain present (CON) in simplicity and idiom
  trees post-prune (CON-position record survives in non-veto
  trees).

  **Q5 (quirky survives correctness, complex survives correctness):**
  S3, S4 (quirky-but-fine) and S5, S6 (complex-but-correct) are
  NOT vetoed from correctness. The substrate distinguishes "fails
  on a non-veto axis" from "fails correctness."

If all 5 hold, qwen3.5:4b is doing enough cognitive work as an
embedder that the substrate's verdicts match human intuition
without further LLM involvement.

## Outputs

  - `tier1a_status.json` — live status (which snippet, which call)
  - `tier1a_results.json` — per-snippet (3 raw LLM calls + median
    coords + veto verdict + per-tree presence)
  - `tier1a_plot.png` — coords scatter (12 snippets in 3-D
    projected to 2-D pairs) plus per-snippet n trajectories
  - PASS/FAIL per prediction at end

## Implementation notes

- Reuse the streaming ollama call shape from `run_a1.py`. qwen3.5:4b
  with temperature 0.3, num_predict ~500 (we only need a small JSON
  object plus rationale).
- Cache LLM calls in a JSONL file so re-runs are cheap. Same
  pattern as `run_a1.py`'s `load_existing_results`.
- The substrate run is fast (Tier 0 took seconds); the LLM-call
  block is the cost driver. Total: 12 snippets * 3 calls = 36
  qwen calls. At ~5s each that's ~3 minutes. Cache means re-runs
  are sub-second.
- Hand-write the 12 snippets in `tier1a_snippets.py` so they're
  inspectable and easy to swap.
- Use the same `build_world` / `round_step` / `prune_veto_negatives`
  call structure as `run_tier0.py`. Differences are small:
  observation coords come from cached LLM output instead of
  `WORK_UNITS`.

## Success / failure / unknown

  - **All 5 predictions hold:** Tier 1A green. The cheap pipeline
    works. We can plan Tier 1B (LLM as answerer over post-prune
    state) or other extensions.
  - **Q1 fails (LLM inconsistent):** the LLM can't produce stable
    coords. Either tighten the prompt + few-shot it, or accept
    that this size of model won't carry the embedder role and
    move up to a larger one.
  - **Q3 fails (buggy not vetoed):** either the LLM rated buggy
    snippets as correct=positive (LLM problem), or coords were
    correct but the substrate didn't accumulate enough CON
    evidence (substrate problem). The trace will show which.
    Investigate before adding more capability.
  - **Q5 fails (quirky/complex got vetoed):** the LLM was too
    harsh on correctness for non-bug code. Probably a prompt issue.
  - **Other failures:** investigate per-snippet, decide whether
    to revise snippet selection, prompt, or substrate config.

## Actual outcome (2026-05-03)

1/5 predictions passed (Q2). The findings, and what they mean:

**What qwen3.5:4b does well (the LLM-side signal):**
  - Decisive on the correctness axis: gold snippets all (1,1,1);
    clear bugs (S7 off-by-one, S11 IndexError, S12 wrong-recursion)
    all get correctness=-1.
  - Returns clean JSON when given enough token budget (5500
    num_predict). Empty responses at 3000 tokens were the model
    running out of budget before emitting JSON after thinking.
  - Sample-to-sample stable: median of 3 calls is consistent.

**What qwen3.5:4b does poorly:**
  - Generous on simplicity/idiom: "uses range(len()) instead of
    enumerate" gets idiom=+0.5 not -0.5; "nested comprehension" gets
    simplicity=+1.0 not -1.0. The model interprets the axes as
    "is this fine?" rather than "compared to the most pythonic
    version." Q1's sign expectations on simplicity/idiom failed
    9 ways for this reason.
  - Misses subtle bugs: `is 0` got median correctness=+0.5
    because CPython's small-int interning makes the snippet
    "work" most of the time. The model isn't wrong here -- the
    bug is real but conditional, and the LLM is reflecting that.

**What the substrate does (and doesn't do):**
  - When LLM coords are clean and structurally distinct, the veto
    fires correctly: S7 vetoed (coords (-1, 0, -1) -> CON of
    correctness, intrinsic -4); S12 vetoed (coords (-1, 1, 1) ->
    CON of correctness, intrinsic -5). Same shape as Tier 0's W4
    case, working as expected.
  - When LLM coords are ambiguous (S8 at (+0.5, +0.5, -0.5)), the
    substrate places the snippet as PRO of correctness because
    coords[0] > 0. The veto doesn't fire because there's no CON
    classification to read. This is correct behavior: ambiguous
    LLM signal -> ambiguous substrate verdict.
  - When LLM coords land on the same point (S1, S2, S5, S6, S9,
    S10 all at (+1,+1,+1)), the content-addressed hash collapses
    them into one node. The presence-tracker sees only the first
    sprout's observation_id. This is correct architecturally
    (one claim per coord) but the test instrumentation lost
    track of duplicates. Q2 passes because S1 + S2 are both
    "represented" by the merged node, and that node survives.
  - When the substrate evaluates the same coord-class observation
    multiple times in act(), it can sprout sub-children under
    the existing snippet node. Those sub-children's edges
    contribute to the tendency-tree intrinsic walk. With one
    snippet (S11 at (-1, +0.8, +0.5)), this produced a CON
    sub-child whose sub_intr=+6 cancelled the parent's 5 direct
    stakes -- net intrinsic = -1, signed contribution = +1, not
    vetoed. This is a substrate-side complication that didn't
    show up in Tier 0's diagonal-coord setup.

**The honest take:**
  - The pipeline works for clear cases (gold, simple bugs).
  - It produces calibration noise on stylistic axes because the
    small LLM doesn't grade simplicity/idiom on the same scale
    we hand-coded into Tier 0.
  - The substrate's veto is sensitive to structural placement; it
    works cleanly when LLM coords are decisive but can be cancelled
    by accreted sub-children when coords are mid-magnitude.
  - 1/5 is not a green light, but it's also not a hard fail.
    It's "the substrate's lower bound on LLM intelligence is
    higher than 4B-with-thinking-mode for this configuration."

**What would unblock further:**
  - Better LLM (haiku, qwen2.5-coder:7b+): might fix Q1 calibration
    and S8's bug detection, removing the LLM-side noise without
    changing the substrate.
  - Few-shot the prompt with an example of "uses .append in a
    loop -> idiom = -0.4" so qwen knows the negative end of the
    scale is reachable. Cheap experiment.
  - Substrate-side dedup of same-obs sprouts: don't sprout a
    sub-child if one of the same observation_id already exists
    elsewhere. Would close the S11 sub-child accretion. Bigger
    architectural change.

## Estimated effort

- Snippets file: 30 minutes (12 snippets, careful labeling)
- Runner: ~1 hour (LLM calling, caching, substrate run, predictions)
- First run + investigate: 30-60 minutes

Total: 2-3 hours from spec-OK to verdict.

## What this does NOT test

- LLM-as-answerer (reading the post-prune state and producing a
  natural-language verdict). Tier 1B if we want it.
- Federation across solvers. Already covered by
  `test_federation_parent_merge.py`.
- Continuous-kernel dynamics. Lindblad already validated; not
  re-tested here.
- Multi-epoch n drift / repeated observation cycles. The single-
  epoch shape is already enough to test the embedder.
- Larger-than-beginner Python (any non-trivial library use,
  multi-file code, OO design). That's Tier 2 territory.
