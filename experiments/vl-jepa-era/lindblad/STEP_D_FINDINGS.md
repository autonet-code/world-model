# Step D — bounded direction + confidence-modulated gamma

## What was changed

Replaced unbounded omega/zeta with bounded direction/tension:

- omega = kappa * (signed_weight / total_weight) ∈ [-1, 1]
- zeta = lambda * 2*sqrt(p*c) ∈ [0, 1]
- Cross-coupling J normalized by sqrt(W_a * W_b * (W_a + W_b))
- gamma_eff = gamma_base / (1 + W / W_scale) — heavy roots resist obs

Side-by-side run on the same three scenarios from Step C.

## Honest results

### Scenario 1 — 10 PRO observations
Continuous still saturates to ~1.0 by step 5. Step D didn't change this
because direction was already saturated at +1 from step 1.

### Scenario 2 — alternating PRO/CON
Continuous still climbs monotonically to 0.95.
Reason: substrate's PRO sub-claim absorbs while CON doesn't (the
`_con_positioned` mechanism), so capacities asymmetrically favor PRO.
Direction stays positive throughout, omega stays positive, alpha climbs.

### Scenario 3 — 5 PRO then 5 CON (the cleanest result)
Now showing real dynamics: continuous tracks the direction reversal.
Root a goes from 0.96 (after PRO phase) to 0.73 (after CON phase).
Root b mirrors it to 0.58 via J coupling — entanglement-driven shift
without any direct CON observations on b.

## What worked

- The Step B predictions (beats, tilted steady states, entanglement,
  reversibility) were validated for hand-built configs in Step B and
  remain valid.
- Cross-root coupling propagates correctly through bound J.
- Direction reversal in S3 is now responsive (compared to Step C
  where continuous got stuck at 0.95).

## What didn't work

The W_scale knob doesn't meaningfully damp gamma in practice because
the substrate's per-sub-claim capacity values are tiny (~0.01-0.05),
making W small even with many sub-claims. Confidence modulation requires
W to grow faster than the dissipator's gamma, which doesn't happen at
realistic substrate scales.

This isn't fixable by tweaking W_scale alone. The substrate's capacity
mechanism — capped at 1.0 per node, decayed each round — fundamentally
limits how much W can grow.

## What this means

The Step D fix is **structurally correct** (direction in [-1,1] is the
right concept). But the empirical bridge from substrate state to W
needs more thought. Two interpretations:

  (a) The substrate's capacity mechanism is the wrong measure of
      confidence. Confidence should be a *running count* of supporting
      observations, not a recent-window capacity. We'd need a separate
      "evidence accumulator" per sub-claim that doesn't decay, only
      grows.

  (b) The continuous-time bridge isn't the right architecture for the
      substrate. We've spent five stages trying to map a discrete-stake
      graph onto a continuous-time quantum master equation, and each
      step has revealed a structural mismatch. Maybe the substrate
      really is fundamentally discrete and we should embrace that
      rather than fight it.

I lean (b). The substrate's discrete update rule was driven by a
clear cognitive intuition (agents stake on claims; scores accumulate;
graph grows). The Lindblad picture is elegant but the substrate keeps
not fitting it. Each fix introduces one more parameter we have to
calibrate to make the bridge meaningful, while the underlying
discrete substrate continues to do what it was always going to do.

## Recommendation

Stop bridging. The Step A/B/C/D math is correct, the kernel works,
but the substrate is a discrete-time stake-counter and the bridge
will never be more than a loose translation.

Two genuinely useful artifacts to keep:

  1. The Lindblad kernel (`lindblad_kernel.py`). Useful as a reference
     simulator for any future quantum-cognitive design choice.
  2. The Step A/B math as a **specification**, not an implementation.
     If we ever build a substrate from scratch with continuous-time
     dynamics in mind, this is the formal scaffold.

What we lose: the "substrate is secretly Lindblad evolution in disguise"
narrative. That story is wrong.

What we keep: knowing the substrate isn't a quantum-cognitive system
at the per-root level, and shouldn't be marketed as one. The substrate
*is* a graph-based debate registry with discrete stake mechanics. That's
what it does well; that's what its claims should rest on.

## What to do instead

Three options I think are worth real time:

  - **Sharpen the discrete substrate's actual claims.** What is it
    *good at* that classical Bayesian or symbolic systems aren't?
    Run experiments that test the genuinely distinctive features:
    federation via content addressing, smooth promotion, locate
    primitive's behavior under heterogeneous queries. Stop making
    quantum-cognitive claims; characterize what's actually there.

  - **Look at quantum-implementation pathways from a different angle.**
    The substrate's *state* is encodable in Hilbert space (every
    finite classical state is). The substrate's *update rule* might
    have specific operations that admit quantum speedup (Grover-style
    search on locate, amplitude amplification on contention search).
    These are tractable questions; Lindblad evolution wasn't the
    right one.

  - **Accept the discrete framework and explore its formal structure.**
    The substrate is closer to a Polya urn or a discrete-time Markov
    chain on stake configurations. There may be a clean formal theory
    here we haven't named. Different from quantum-cognitive but
    potentially just as interesting.

Which we pick depends on what the substrate's eventual purpose is.

## Files produced this stage

  - `lindblad/step_d_params.py` — bounded parameter formulas
  - `equilibrate_continuous` updated with mode="stepD" support
  - `step_c_compare.py` updated to use stepD and smooth_promotion
  - This document
