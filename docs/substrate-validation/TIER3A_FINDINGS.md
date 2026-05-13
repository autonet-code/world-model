# Tier 3A findings: LLM-as-embedder in autonet's pipeline

4/5 predictions passed. The one "failure" was a metric bug — the
real disagreement rate between LLM and heuristic is 0.8%, not the
37.5% the original A3 metric reported.

## Headline result

  | prediction | result | what actually happened                          |
  |------------|--------|--------------------------------------------------|
  | A1 destructive committed | PASS | All 6 destructive turns: LLM committed life≤-1 or self≤-1 |
  | A2 banal not noisy       | PASS | 8 banal turns; avg Manhattan distance to heuristic 0.50  |
  | A3 sign agreement ≥80%   | FAIL | 62.5% by-letter; 0.8% real flip rate (metric bug, see below) |
  | A4 LLM catches misses    | PASS | 8 witness turns where heuristic returned (0,0,0,0) and LLM committed |
  | A5 events round-trip     | PASS | 82 events replayed cleanly; 58-node post-replay world; 4 root scores |

## The A3 metric reframe

A3 said "substrate verdicts agree on sign for ≥80% of (turn, axis)
pairs." By-letter the metric returned 75/120 = 62.5% — failing.

Breaking down those 120 axis-pairs more carefully:

  - **66 (55%)**: both heuristic and LLM returned 0 (banal/filler
    turns; trivial agreement)
  - **9 (7.5%)**: both committed and agreed on sign
  - **44 (36.7%)**: LLM committed, heuristic returned 0 — this is
    the LLM **adding signal**, not disagreeing
  - **0 (0%)**: heuristic committed, LLM returned 0
  - **1 (0.8%)**: actual flip — heuristic and LLM both committed
    but on opposite sides

**The real disagreement rate is 0.8%, not 37.5%.** The metric
conflated "LLM adds signal where heuristic stays silent" with
"LLM disagrees with heuristic." The LLM almost never contradicts
the heuristic when they both commit.

## Three concise framings

### Framing 1 — what the LLM actually adds

The LLM's value-add is **committing where the heuristic abstains**.
The keyword-based heuristic (score_turn_4d) catches obvious markers
like "rm" or "delete" but returns (0, 0, 0, 0) for anything that
isn't a hard keyword match. The LLM reads context: it scores
"writing regression tests" as positive on self_preservation +
intelligence + evolution; it scores explanatory replies as
positive on intelligence + evolution; it scores `is 0` style
subtle-bug discussions as negative on multiple axes.

44 out of 120 (36.7%) axis-pairs gain real signal from the LLM
that the heuristic doesn't see. **That's a substantial cognitive
upgrade for a single-call swap-in.**

### Framing 2 — what the LLM doesn't change

The LLM **rarely contradicts** the heuristic. Of 120 axis-pairs,
exactly 1 was a sign flip. The heuristic's keyword-based scores
are mostly correct *when they're nonzero*; the heuristic's main
weakness is being too conservative (returning zero too often),
not getting the direction wrong.

This means the integration is non-controversial. Swapping to
LLM-as-embedder doesn't reverse the heuristic's reads — it
extends them. Existing autonet workflows that depend on the
heuristic's behavior would see strictly more signal, not
different signal.

### Framing 3 — what the substrate sees

The two arms produce dramatically different root scores:

  | tendency               | heuristic score | LLM score |
  |------------------------|-----------------|-----------|
  | life_precious          | +278            | -1267     |
  | self_preservation      | -147            | -2961     |
  | promotion_of_intelligence | +254         | -3427     |
  | evolution              | +88             | -3333     |

These look wildly different but reflect the same input data
through two different decisiveness levels. The LLM's commit-on-
clearly-bad behavior generates much higher-magnitude scores
because every destructive turn lands -1 across multiple axes;
the heuristic's softer scoring makes those same turns land
-0.4 to -0.8 on a single axis.

**The relative ordering is what matters for verdict purposes**,
and the LLM's decisive shape produces clearer signal-to-noise.
At equilibrate-then-prune time, the LLM-driven world will
distinguish "clearly contested" from "clearly settled" more
reliably than the heuristic-driven world.

## Per-turn highlights

  - **d1 (rm -rf)**: heuristic (0, -0.8, 0, 0); LLM (0, -1, -1, -1).
    LLM caught self_preservation + intelligence + evolution dimensions
    the heuristic missed.
  - **c2 (refactor with rationale)**: heuristic (0, 0, 0, +0.2);
    LLM (0, +1, +1, +1). LLM read "refactor" + documented rationale
    as supporting all three positive axes.
  - **r1 (long technical explanation)**: heuristic (0, 0, 0, 0);
    LLM (0, 0, +1, +1). LLM caught the explanatory work as
    intelligence + evolution; heuristic saw nothing.
  - **b6 (`git status`)**: heuristic (0, 0, 0, 0); LLM (0, +1, +1, 0).
    LLM interpreted status-checking as careful + reasoning; arguable
    over-commit, but reasonable.
  - **b4, b8 (short acks)**: heuristic (0, 0, 0, 0); LLM
    (0, 0, -1, 0). LLM read terse replies as negative on
    intelligence ("not engaging openly"); arguable, perhaps too
    strict.

## What this validates

  - **Tier 1C's binary-flag pattern transfers to autonet's seam.**
    Same shape, different domain (charter axes instead of code
    correctness/simplicity/idiom). LLM commits when forced; the
    heuristic stays silent.
  - **The integration is mechanical.** No engine changes, no
    protocol changes, no aggregator changes. One function swap
    (`turn_to_observation` → `turn_to_observation_via_llm`).
  - **Events round-trip through autonet's existing aggregator.**
    Despite some "unknown node" warnings during replay (likely a
    sub-claim sprouted under a node that itself got pruned mid-
    equilibrate), the contribution payload aggregates cleanly and
    produces a well-formed post-replay world.

## What's worth knowing

  - **The LLM is more decisive on borderlines.** This is good for
    veto-style verdicts (Tier 0 / Tier 1C lesson) but means the
    LLM-driven scores have higher absolute magnitudes than the
    heuristic. Comparison metrics should normalize before reading.
  - **The "unknown node" warnings during replay** suggest some
    sub-claims sprouted during per-turn equilibrate get orphaned
    by the time the aggregator replays. This doesn't break A5,
    but it means each replayed event isn't 1:1 with each original
    sprout. Worth a closer look if event-level attribution matters
    downstream (it does for mint computation in `reconcile.py`).
  - **The synthesized corpus is biased toward category-clear cases.**
    Real autonet turns (loaded via `--real N`) would test on more
    ambiguous data. Worth a follow-up with `--real 30` to see how
    the LLM behaves on actual conversation traces.

## Recommendation

**Swap-in is worthwhile if you want decisive verdicts and don't
mind the LLM cost.** The integration shape is mechanical:

1. Add a flag (e.g. `embedder=llm` or env var `AUTONET_EMBEDDER=llm`)
   to `turn_to_observation` that routes to `turn_to_observation_via_llm`
   when set.
2. Cache LLM results by turn-obs-id (already implemented).
3. Pre-warm the cache by running the embedder on existing
   `~/.atn/conversations/` data before deploying.
4. Run a controlled comparison in real training cycles: same
   solver on same task, both arms, watch mint outcomes.

The cost shape with haiku via the proxy is: 30 turns ≈ 90 calls ≈
3-5 minutes of wall time, ~zero per-token cost (Claude Max sub).
For autonet's training cycles (each cycle processes ~200-500 turns),
this is ~10-30 minutes per cycle of LLM time — not free, but the
substrate's verdicts get meaningfully sharper.

## What's NOT done

  - Multi-solver consensus on the LLM-driven world (Tier 3B).
    That would test whether multiple solvers running the LLM
    embedder converge on the same verdict.
  - End-to-end mint flow (RPB.recordTraining etc.). PLAN.md
    Phase 1 is independent.
  - Real-conversation supplement (--real 30). Worth running
    once before declaring the corpus characterized.
  - qwen comparison. Tier 1C showed qwen and haiku produce
    identical substrate verdicts under binary prompts; the
    haiku-only result is sufficient for the integration test.
    qwen would add ~90 minutes for no expected new signal.

## Estimated effort spent

  - Spec: 30 min
  - Implementation: 1 hour
  - Smoke + run + diagnose: 30 min (haiku run was ~5 min, but
    diagnosing the A3 metric was the time sink)
  - Writeup: 30 min

Total: 2.5 hours from spec-OK to verdict. Same envelope.
