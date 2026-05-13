# Reshaped A1: substrate n-tracking test

## What this tests

Whether the substrate's persistent novelty `n`, produced by the
dynamics from a designed observation schedule, tracks the confidence
pattern we'd expect.

No renderer, no LLM, no prompt translation. Just: observations in,
graph state out, compare `n` values to predictions.

## Why it's the right next test

C1 (the renderer-facing test) has a long causal chain. If it fails,
we won't know which link broke. Reshaped A1 isolates the substrate-
dynamics link first. Pass A1 → green light for C1. Fail A1 → fix the
dynamics before involving an LLM.

## Test design

A single tendency (single root) with multiple direct sub-claim
children, each occupying a distinct coordinate region. We drive
observations into specific regions on specific schedules, then read
out per-node `n` and compare against predictions.

### The substrate

One tendency `r` with anchor at the origin in 4-D coordinate space.
Four sub-claim regions, each represented by an initial PRO sub-claim
sprouted at a distinct coordinate:

  region A: anchor (1, 0, 0, 0)
  region B: anchor (0, 1, 0, 0)
  region C: anchor (0, 0, 1, 0)
  region D: anchor (0, 0, 0, 1)

Each region's PRO sub-claim starts at n=1.0 (fresh).

### The observation schedule

Drip 30 rounds of observations targeting different regions:

  rounds 1-10:  10 PRO observations targeting region A
                (sustained confirmation; expected: A's n decays toward
                the rate-determined steady state)

  rounds 1-10:  10 PRO observations targeting region B
                (in parallel with A)
                (expected: B's n decays similarly)

  rounds 11-20: 10 CON observations targeting region A
                (sudden contradiction after settling)
                (expected: A's n regrows; rises back above B's)

  rounds 1-5:   5 PRO observations targeting region C (early settling),
                then no observations rounds 6-30
                (expected: n_C drops rounds 1-5, then drifts UPWARD
                via the +epsilon(1-n) term over the long quiet period)

  rounds 1-3:   3 CON observations on region D, then nothing
                (brief contradiction; expected: n_D stays near 1.0
                throughout — never settled)

### Predictions to verify

  P1. After rounds 1-10:  n_A meaningfully decreased from 1.0 (concrete
      threshold: n_A < 0.6 — the substrate's sub-claim re-staking gives
      a per-round rate of about 0.057, so 10 rounds should drop n by
      roughly half).
  P2. After rounds 1-10:  n_B similarly decreased; n_A and n_B should
      be close to each other (within 0.1) since they receive symmetric
      treatment.
  P3. After rounds 11-20:  n_A > n_B (region A was re-surprised by CON;
      region B remained settled). Concrete: n_A - n_B > 0.1.
  P4. n_C trajectory is NON-MONOTONIC: drops during rounds 1-5
      (PRO confirmation) then drifts back upward during rounds 6-30
      (no observations, only +epsilon(1-n) drift). Concrete:
      n_C at round 5 < n_C at round 30.
  P5. n_D stays above 0.7 throughout (CON-only, never settled).
  P6. Trajectory of n_A is non-monotonic across rounds 1-20:
      drops during 1-10 (PRO), rises during 11-20 (CON).
      The resist-then-yield-decisively kernel signature.

### Outputs

  - `a1_reshaped_status.json` — live status (which round, etc.)
  - `a1_reshaped_results.json` — full trace: per-round per-region
    (n, score) values
  - `a1_reshaped_plot.png` — n trajectories per region across rounds
  - Print PASS/FAIL for each prediction

### Implementation notes

- Use the substrate's `equilibrate` (the discrete kernel) for this
  test. The continuous kernel changes score dynamics but doesn't
  change `update_novelty`. We test n-tracking alone first; testing
  whether continuous-kernel scores produce different n is a follow-on.
- Each "round" = add observation, run `act + apply_stakes + update_novelty`.
  We don't run full `equilibrate` to fixed point; we want to control
  the round granularity precisely.
- Use a coordinate-distance threshold to decide which region each
  observation lands in. The substrate's existing locality logic
  handles this via the polarity probe.
- Sub-claims should be sprouted upfront so the regions exist before
  observations arrive. Otherwise the first observation creates the
  region.

### Success / failure / unknown

  - **All 6 predictions hold:** substrate n captures intended cognitive
    structure. C1 is green-lit.
  - **1-2 predictions fail in obvious ways:** identify which dynamic
    is misbehaving, propose targeted fix, retest.
  - **Multiple predictions fail or behavior is unintelligible:** the
    `update_novelty` rate constants might need tuning, or the
    underlying formula might be wrong, OR our intuitions about
    what should happen might be off. Investigate before C1.

## Estimated time

15 minutes to implement. Run is seconds (no LLM). Plot inspection
takes 5 minutes. Total: maybe 25 minutes from spec-OK to verdict.
