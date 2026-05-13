# Tier 0 spec: synthetic three-root substrate (no LLM)

## What this tests

Whether the post-and-coparent substrate, configured with three
deployer-domain roots (correctness, simplicity, idiom), produces the
expected score and pruning dynamics under hand-crafted observations.

No LLM. Pure substrate dynamics. We drive observations targeting
specific (root, polarity) combinations, run the discrete kernel, and
check that scores, n trajectories, and veto-pruning behave as
predicted.

## Why this is the right first Tier 0

We have two cheap things going for us:

  - The reshaped-A1 harness already showed us the n dynamics shape
    we want (resist-then-yield-decisively under the Lindblad branch,
    discrete approximation under the classic branch).
  - The post-and-coparent refactor just landed. Its primitives need
    a multi-root exercise to confirm they compose: cross-tendency
    edge discovery, intrinsic_score across parents, correctness-as-
    veto pruning.

A synthetic harness lets us isolate "does the substrate express the
cognitive shape we want" before adding LLM noise. Pass Tier 0 →
green light to build the LLM-in-the-loop tier on top. Fail Tier 0 →
the substrate config (roots, bandwidths, rate constants) needs work
before LLMs can ride it.

## The substrate

Three roots in a 3-D coordinate space. Axis convention:
  - dim 0 = correctness axis
  - dim 1 = simplicity axis
  - dim 2 = idiom axis

```
  correctness:  anchor=(1, 0, 0), axis=(1, 0, 0), veto_shaped=True,
                novelty_gamma_con=1.5  (CON evidence settles fast)
  simplicity:   anchor=(0, 1, 0), axis=(0, 1, 0), veto_shaped=False
  idiom:        anchor=(0, 0, 1), axis=(0, 0, 1), veto_shaped=False
```

Bandwidth = 0.7 on each (gives `bandwidth*1.5 = 1.05` for edge
discovery — close to enough to bridge orthogonal-but-related claims).

`veto_score_floor = -1.0` on correctness.

## Observation language

Each observation has 3-D coords. The sign of `coords[i]` is the
position-on-axis-i:
  - `coords[0] > 0` = supports correctness  (PRO of correctness root)
  - `coords[0] < 0` = contradicts correctness (CON)
  - `coords[1] > 0` = supports simplicity, etc.

A "code is correct, simple, but unidiomatic" observation maps to
coords `(+1, +1, -1)`. A "code is incorrect" observation alone maps
to `(-1, 0, 0)`.

Magnitude is unit; we don't tune it because the post-only model
treats every post the same.

## Hand-crafted observation set

Six work-unit-shaped observations, drip-fed across rounds:

| W   | coords           | label                              | what it represents                          |
|-----|------------------|------------------------------------|---------------------------------------------|
| W1  | (+1, +1, +1)     | "clean, correct, idiomatic"        | gold standard work item                     |
| W2  | (+1, +1, -1)     | "correct, simple, but quirky"      | passes correctness + simplicity, not idiom  |
| W3  | (+1, -1, +1)     | "correct, idiomatic, complex"      | passes correctness + idiom, not simplicity  |
| W4  | (-1, +1, +1)     | "buggy but otherwise nice code"    | should trigger correctness veto             |
| W5  | (+1, 0, 0)       | "definitely correct, scope unknown"| narrow correctness post                     |
| W6  | (-1, -1, -1)     | "objectively bad on all axes"      | should be deeply pruned                     |

Schedule: under the post-only refactor, `apply_stakes` wipes prior
tendency stakes at the start of each round, so post counts are
round-fresh. To accumulate signal across all six work units in
parallel, we fire ALL six observations EVERY round for 15 rounds.
Predictions are tested at specific round indices (e.g. P1 at
round 5) plus after `prune_veto_negatives` at epoch close.

The substrate's `act` decides PRO/CON child placement based on the
probe's stance against each tendency's frame; we pre-sprout nothing.
We do NOT run `prune_settled_negatives` here -- it's too aggressive
when scores are 0 between rounds and would erase informative
structure. The veto-prune is the only structural mechanism we test.

## Predictions

Six falsifiable predictions covering the architecture's surface:

  **P1 (co-parenting forms):** W1's substrate-side node acquires
  parent edges in all three tendency trees by round 5 (anchor
  (1,1,1) is within bandwidth*1.5 of every root). Concrete: the
  parent_tendency set is exactly `{correctness, simplicity, idiom}`.

  **P2 (n decays under PRO):** W1's node `n` falls below 0.3 by
  round 5. Sustained PRO evidence is settling.

  **P3 (correctness veto fires on W4):** W4 = (-1,+1,+1) gets
  classified as CON of correctness root by the probe (its
  coords[0]=-1 sets the CON polarity). After
  `prune_veto_negatives` runs at epoch close, W4 is removed from
  correctness's tree. The veto mechanism reads CON-position +
  accumulated intrinsic_score (i.e. signed contribution to the
  root's score) against `veto_score_floor`; once W4 has enough
  accumulated evidence to cross the floor, it's vetoed.

  **P4 (non-veto roots don't auto-prune):** W3 sits as CON of
  simplicity (coords[1]=-1) but simplicity isn't veto-shaped, so
  W3 stays in simplicity's tree post-prune. Simplicity tracks the
  CON-classification as informative metadata, not as a kill signal.

  **P5 (W4 survives outside correctness):** W4 is removed from
  correctness's tree but remains present in simplicity's and idiom's
  trees post-prune. The veto removes the work item from the
  correctness tree without erasing it from the world.

  **P6 (W6 vetoed from correctness):** W6 = (-1,-1,-1) gets
  classified as CON in all three trees. After `prune_veto_negatives`,
  W6 is removed from correctness's tree (other roots keep their
  CON-position record).

## Outputs

  - `tier0_status.json` — live status (round, last update)
  - `tier0_results.json` — full trace + per-prediction PASS/FAIL
  - `tier0_plot.png` — per-W per-round (n, intrinsic_score in each
    of the three roots, parent-tendency count)
  - Prints PASS/FAIL per prediction at end of run

## Implementation notes

- Implement in `phase2/run_tier0.py` next to the existing
  reshaped-A1 runner. Same shape: `build_world`, `round_step`,
  schedule fn, predictions, plot.
- `act` posts unit-magnitude intents; we don't pre-sprout. The
  substrate sprouts whatever the probe decides on first observation.
- Pass `world` into all `_ensure_obs_child` -> `sprout_child` calls
  (already wired up in the engine).
- Run `equilibrate(world, max_rounds=2)` per round so we see the
  per-round dynamics rather than full convergence. Each round still
  triggers update_novelty once.
- At the end, run `prune_veto_negatives(world)` then
  `prune_settled_negatives(world)` and capture which ids were
  removed by which.

## Success / failure / unknown

  - **All 6 predictions hold:** the post-and-coparent substrate
    expresses the deployer's three-root cognitive structure. Green
    light to build LLM-in-the-loop on top.
  - **1-2 predictions fail in obvious ways:** identify which
    primitive is misbehaving (probe ranking, edge discovery, prune
    threshold), targeted fix, retest.
  - **Multiple predictions fail or behavior is unintelligible:**
    either rate constants need tuning, or our intuitions about
    three-root composition are off, or the probe's claim-ranking
    logic is sending observations to the wrong parent (the same
    issue we saw in reshaped-A1). Investigate before adding LLMs.

## Estimated time

30-45 minutes to implement (similar shape to run_a1_reshaped.py with
three roots instead of one). Run is seconds. Total: maybe 1 hour
from spec-OK to verdict.

## What this does NOT test

- LLM reasoning over the substrate. Tier 1+ territory.
- Continuous-kernel dynamics. The Lindblad kernel is already
  validated separately; adding it here would conflate concerns.
- Federation merge across solvers. Already covered by
  `test_federation_parent_merge.py`; no need to re-test here.
- Multi-epoch dynamics (n re-growth in quiet rounds, multi-prune
  passes). The 5-round quiet tail at the end is a minimal nod to
  this; deeper multi-epoch shape is Tier 1+ territory.
