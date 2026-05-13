# Stage 3 — empirical findings

## Setup

Single-tendency, single-root substrate. 20 observations dripped one at
a time, equilibrating after each. PRO observations align with the
tendency's polarity axis; CON anti-aligned. Map net_score → α via
sigmoid.

Observation schedule: 10 PRO, 5 CON, 5 PRO.

## Result

The substrate's α(t) trajectory is **not** Lindblad-shaped at this
granularity.

| Model | RMSE on 21 points |
|---|---|
| Lindblad amplitude damping | 0.040 |
| Linear counter, sigmoid(k·(n_PRO − n_CON)) | 0.064 |

Lindblad fits better than the naive linear counter, but the residual
pattern is wrong: residuals grow monotonically during steady-direction
stretches (substrate keeps lagging behind the exponential's diminishing
returns), and flip sign on direction reversal.

## Why

The substrate's update rule is **not** continuous-time amplitude
damping. It's a discrete iteration over staked nodes:

  - Each PRO observation is added to `world.observations`.
  - Each round of equilibrate, `tendency.act` iterates over ALL
    current observations and stakes `mag = max(0.05, min(1.0, novelty)) × discount`
    on the relevant PRO/CON child.
  - Net score = sum of root's children's net_scores; PRO subtree
    increments, CON subtree decrements.

Because identical observations re-stake on the SAME child each round,
the net_score grows **linearly** in the number of accumulated
observations of a given polarity, with slope = `mag` per observation.
In the test setup `mag ≈ 0.0573`.

So at small α (near 0.5), `dα/dn ≈ k·sigmoid'(0) = k/4`, which looks
like the early part of an amplitude-damping curve. As α grows, the
sigmoid still pinches but the underlying counter doesn't — the
substrate doesn't know about asymptotes. This is why Lindblad's
diminishing-returns prediction diverges from the substrate's
sustained linear climb.

## Direction reversal anomaly

When a CON observation is introduced after 10 PRO, net_score drops
by ~0.5 in one step (much more than the per-obs `mag` of ~0.06). This
is a **frame effect**, not Lindblad-modelable:

  - Once the tendency's `_con_positioned` set marks an observation as
    CON-positioned, subsequent rounds re-stake it at fixed magnitude
    0.5 (rather than the per-obs `mag`).
  - The PRO observations from earlier rounds may have their staking
    mode change because `frame.absorb` doesn't run for CON-positioned
    obs, altering how subsequent PRO obs are evaluated.

This is graph-topological history-dependence. Lindblad is Markovian;
the substrate is not.

## What this means for the import

**Lindblad is the wrong formalism for the whole substrate.** Importing
`evolve(rho, H, jump_ops, t)` as a kinetics replacement for `equilibrate`
would change the substrate's dynamics in ways that break:

  - History-dependent reframing (the `_con_positioned` mechanism)
  - Discrete graph growth (sprout_child during act)
  - Cross-tendency staking rules

These aren't bugs we can paper over; they're load-bearing pieces of
the substrate's design.

**Lindblad IS the right formalism for one specific sub-problem**:
modeling how a single observation's contribution to a single root
ought to relax over (continuous) time when there are no graph-
topological changes. That's a narrow use case but a real one:

  - Slow-decay extension: instead of `mag` being applied as one
    discrete jump per round, it could decay continuously between
    observations via amplitude damping. Roots would settle smoother
    trajectories without the linear-counter overshoot.
  - Observation absorption: when an obs is processed, the resulting
    score change could be derived from a Lindblad pulse of finite
    duration rather than a fixed magnitude, with γ proportional to
    `novelty × outcome_strength × locality_weight`.

Both are kinetics tweaks that preserve the substrate's discrete
graph structure while replacing the magnitude calculation with a
principled rate-based one.

## Falsifiable predictions from Stage 1 — revisited

The original predictions:

  1. **Exponential relaxation** of α toward γ_+/(γ_+ + γ_-): NOT
     observed. The substrate produces linear-in-n trajectories, not
     exponential-in-time.

  2. **Coherent oscillations under unbalanced stakes**: untestable
     in current substrate (no native dt; equilibrate runs to fixed
     point each step).

  3. **Order-dependent query results**: not tested in Stage 3.

  4. **Bounded total mint per coherence-window**: not tested.

The first prediction is decisively negative. The substrate is not a
Lindblad system at the per-root level under its current update rule.

## What's still useful

The Lindblad kernel itself (Stage 2) is a working open-system simulator
with 8/8 tests passing. It can be used:

  - As a **test bed** for proposed kinetics changes (run the kernel,
    compare to the substrate's discrete behavior, decide whether the
    substrate's approximation is acceptable).
  - As a **theoretical reference** for what continuous-time per-root
    dynamics should look like, to check whether substrate updates are
    over- or under-relaxed.
  - As a **specification** for the proposed amplitude-damping
    extension — if we want roots to relax smoothly between
    observations, the kernel tells us how.

It is NOT a replacement for `equilibrate`.

## Recommendation

Stop the Lindblad-as-replacement plan. Don't proceed to Stage 4.

The substrate's claim to "quantum-cognitive structure" rests on
Stage 1 mathematics, and that mathematics describes a system the
substrate is not. Calling the substrate quantum-cognitive at the
single-root level is a mis-characterization.

The substrate is a **discrete-time stake-accumulating graph** with
sigmoid-mapped net scores, history-dependent reframing, and
discrete-event growth/decay. That's interesting on its own terms.
Borrowing the quantum-cognitive vocabulary obscures more than it
clarifies.

What may still be true:

  - The substrate may have **emergent Lindblad-like behavior** in
    specific limits (high observation density, equal frame, no growth).
    Worth checking once we've identified the right limit.
  - The substrate may have a **different formal equivalence** —
    perhaps a Polya urn or a discrete Markov chain on stake
    configurations. Worth searching.
  - The QQ-equality analog (parameter-free invariant) is still
    worth hunting, but should be derived from the substrate's actual
    update rule, not from Lindblad.

## Files produced

  - `lindblad/stage3_substrate_trace.py` — the trace harness.
  - `lindblad/stage3_results.json` — α trajectory + fit.
  - `lindblad/stage3_alpha_trajectory.png` — plot of substrate vs fit.
  - `LINDBLAD_SCAFFOLD.md` — Stage 1 math (still correct as a math
    document, but not a description of the substrate).
  - `lindblad/lindblad_kernel.py` — working open-system simulator,
    keep as a reference tool.
  - `lindblad/test_lindblad_kernel.py` — 8/8 passing.
