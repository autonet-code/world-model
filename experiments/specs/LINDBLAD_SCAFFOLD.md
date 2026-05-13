# Lindblad scaffold for the substrate

The substrate's per-root score α ∈ [0,1] under accumulating observations
is a closed-form open quantum system. This document writes it down
explicitly so it can be implemented and falsified.

## Hilbert space

**Single root.** A root has a PRO/CON polarity axis. The associated
Hilbert space is

    H_root = span{|PRO⟩, |CON⟩}    (dim = 2)

A density operator on H_root is a 2×2 matrix

    ρ = ⎛ p     c   ⎞
        ⎝ c*    1-p ⎠

with p ∈ [0,1] and |c|² ≤ p(1-p). The substrate's score for the root is

    α = ⟨PRO|ρ|PRO⟩ = p

The off-diagonal element c is *coherence* — the part of the state
that hasn't yet been forced into a classical PRO/CON outcome. The
substrate currently has no explicit coherence variable; this is the
new degree of freedom Lindblad introduces.

**N roots.** Joint Hilbert space is the tensor product

    H = ⊗ᵢ H_root_i              (dim = 2^N)

For the four-root charter, dim = 16. A general state is a 16×16 density
matrix; in practice we will track only marginals unless we explicitly
model entanglement between roots.

**Sub-claims do NOT enlarge the qubit register.** A sub-claim is a
contributor to the parent's score, not a separate qubit. Sub-claims
factor into the Hamiltonian (next section). This keeps the register
small and matches the existing substrate's view that nodes accumulate
into root scores.

## Hamiltonian H

The Hamiltonian generates unitary evolution — the part of dynamics
that doesn't involve observation. In substrate terms, this is *stake
propagation through the tree without new observations arriving*.

For a single root, the natural Hamiltonian on H_root is

    H_root = ω · σ_z + ζ · σ_x

where σ_z and σ_x are Pauli matrices in the {|PRO⟩, |CON⟩} basis, and:

  - **ω** is the *bias* term, set by the *net* signed stake at this
    root: ω = κ · Σⱼ sⱼ, where sⱼ ∈ {+1 for PRO sub-claim, -1 for CON}
    weighted by stake magnitude. κ is a scaling constant.

  - **ζ** is the *mixing* term, set by the *unresolved* stake at this
    root — staked-but-not-yet-influencing-direction. This generates
    coherent oscillation between PRO and CON: a root with mixed PRO/CON
    sub-claims won't simply settle on the majority, it will oscillate
    in superposition until observations decohere it.

For sub-claim cross-influence (the locality rule via CoordinateLocator),
add interaction terms between roots whose sub-claims share neighborhoods:

    H = Σᵢ H_root_i + Σᵢⱼ Jᵢⱼ · (σ_z_i ⊗ σ_z_j)

where Jᵢⱼ ∝ exp(-d_ij² / (2 · bandwidth²)) and d_ij is the coordinate
distance between the most-influential sub-claims of root i and root j.
Jᵢⱼ = 0 when the sub-claims are far apart, so the interaction
respects the locality rule already in the substrate.

In matrix form for one root:

    H_root = ⎛ ω   ζ  ⎞
             ⎝ ζ  -ω  ⎠

Eigenvalues ±√(ω² + ζ²); eigenvectors mix PRO/CON at angle determined
by ζ/ω. With ζ=0, eigenvectors *are* PRO and CON (no mixing). With
ω=0, eigenvectors are equal superpositions (maximum mixing).

## Lindblad operators Lⱼ

Each observation is a Lindblad jump operator. An observation that
nudges the root *toward* PRO with strength γ is a transition (amplitude-
damping) operator, not a projector:

    L_PRO = √γ · |PRO⟩⟨CON| = √γ · ⎛ 0  1 ⎞
                                    ⎝ 0  0 ⎠

Applied as a Lindblad superoperator, this drives population from CON
to PRO at rate γ and damps coherence at rate γ/2.

(Note: a *projector* form like √γ · |PRO⟩⟨PRO| is also a valid
Lindblad operator but it implements a non-demolition measurement —
it kills coherence without driving population. It's the wrong fit
for "an observation pulls toward PRO." We want amplitude damping.)

An observation toward CON is symmetric:

    L_CON = √γ · |CON⟩⟨PRO|

Each leaf-arrival in the substrate becomes an L_PRO or L_CON event
depending on whether the observation supported PRO or CON. The rate γⱼ
encodes:

  - stake magnitude (bigger stakes → bigger γ)
  - outcome certainty (a leaf with high outcome.signal_strength
    contributes more decisively)
  - locality weight (an observation in this root's coordinate
    neighborhood contributes more than a distant one)

A reasonable parameterization:

    γⱼ = stake_magnitude · outcome_strength · locality_weight

Which is exactly the product the substrate already computes when it
applies an observation; we're just *renaming* it as a Lindblad rate.

## The master equation

Putting it together:

    dρ/dt = -i[H, ρ] + Σⱼ γⱼ · D[Lⱼ](ρ)

where the dissipator D[L](ρ) is the standard Lindblad form:

    D[L](ρ) = L ρ L† - ½(L†L ρ + ρ L†L)

For L = √γ · |PRO⟩⟨PRO|, this dissipator has matrix elements that
push ρ toward the steady state determined by the balance of L_PRO and
L_CON jumps.

## The two limits

**γⱼ → ∞ (strong observation):** the dissipator dominates. Coherence
c decays instantly. ρ becomes diagonal:

    ρ_steady = ⎛ p_PRO    0    ⎞
                ⎝ 0     p_CON ⎠

with p_PRO = γ_PRO_total / (γ_PRO_total + γ_CON_total), where the
γ-totals are the cumulative observation rates toward each pole. This
recovers a *classical* root: pure PRO/CON mixture, no superposition,
score = p_PRO. The substrate's current behavior under heavy
observation pressure matches this.

**γⱼ → 0 (no observations):** dissipator vanishes. Pure unitary
evolution under H. ρ oscillates between PRO and CON at frequency
2√(ω² + ζ²) — Rabi oscillations driven by the stake imbalance. The
root never settles; coherence is preserved. This is the "novelty
hasn't been consumed yet" regime: the substrate has a strong stake
pattern but no observations have grounded it.

The substrate today is *implicitly* operating in the γ → ∞ limit
(immediate score updates, no preserved coherence). Lindblad lets us
sit at finite γ where the dynamics are richer.

## Closed form: one root + one channel

For a single root with observation channel L = √γ · |PRO⟩⟨PRO| and
no Hamiltonian (H = 0), starting from a maximally-mixed state
ρ_0 = ½ I:

    p(t) = ½ + ½(1 - exp(-γt))    →    α(t) = 1 - ½exp(-γt)
    c(t) = c_0 · exp(-γt/2)

Score relaxes toward 1 at rate γ (so a half-life of ln(2)/γ).
Coherence decays at rate γ/2. After time t = 5/γ, score is 0.997 of
its asymptotic value — practically settled.

For two channels L_PRO = √γ_+ · |PRO⟩⟨PRO| and L_CON = √γ_- · |CON⟩⟨CON|:

    α_steady = γ_+ / (γ_+ + γ_-)

with relaxation rate (γ_+ + γ_-) and coherence decay rate (γ_+ + γ_-)/2.
This is the directly-falsifiable prediction: in a substrate where a
root receives streams of PRO and CON observations at rates γ_+ and
γ_-, the score should asymptote to γ_+/(γ_+ + γ_-) and the trajectory
should be exponential with the predicted time constant.

When the Hamiltonian is non-zero, the trajectory acquires oscillatory
components on top of the exponential decay — quantum beats during
relaxation. If the substrate exhibits oscillation in α before settling
(it currently doesn't because equilibrate is overdamped), this is
evidence of Lindblad-style dynamics in the underlying mechanism.

## Multi-root marginals

For N roots tracked as a joint state, the marginal state of root i is

    ρᵢ = Trⱼ≠ᵢ ρ

obtained by partial trace over the other roots. Score of root i is
αᵢ = ⟨PRO|ρᵢ|PRO⟩.

If the cross-root interaction Jᵢⱼ = 0 (no locality coupling), the
joint state factorizes ρ = ⊗ᵢ ρᵢ and each root evolves independently.
The substrate currently behaves this way at the root level; cross-tendency
influence happens via the locality rule, which under Lindblad becomes
Jᵢⱼ ≠ 0.

When Jᵢⱼ ≠ 0, roots can become *entangled*: the joint state cannot
be written as a tensor product, and individual root dynamics depend
on the others' state. This is qualitatively new behavior — the
classical substrate cannot represent root-root entanglement. Whether
this matters empirically is a Stage 3 question.

## What changes in the substrate's interfaces

At the API level, almost nothing:

  - `score(node)` returns α (the diagonal of ρ) as before.
  - `add_observation(world, obs)` becomes "apply Lⱼ for time dt" instead of
    "directly update score."
  - `equilibrate(world)` becomes "evolve under H + Σⱼ γⱼ D[Lⱼ] for time t."

What's new:

  - Each root carries a 2×2 ρ instead of a scalar score.
  - Coherence c is exposed as a side-channel signal: high coherence =
    "this root is in flux," low coherence = "this root has decohered."
  - Cross-root Jᵢⱼ surfaces a quantitative measure of substrate
    entanglement, computable from the substrate's coordinate space and
    bandwidth.

## What stays unchanged

  - Tree topology, parent/child semantics, PRO/CON polarity.
  - Stake mechanics, content-addressing, sprout_child.
  - Locator (Keyword/Coordinate/Chain), render primitive.
  - Mint formula, novelty decay, pruning.
  - Federation via content-addressed events.
  - The four charter tendencies.
  - Outcome attribution and the persistent ledger.

The Lindblad import is purely a *kinetics* upgrade. It replaces the
"how does score change over time" mechanism with a closed-form
quantum-cognitive process while leaving topology, ledger, and
federation untouched.

## Falsifiable predictions

If the substrate is genuinely a Lindblad system at the per-root level:

  1. **Exponential relaxation.** A root receiving observations at
     rates γ_+, γ_- should have α(t) approaching γ_+/(γ_++γ_-) on
     timescale 1/(γ_++γ_-). Measurable from logs of any substrate run.

  2. **Coherent oscillations under unbalanced stakes.** A root with
     mixed PRO/CON stake but no observations should oscillate (visible
     if we run the substrate in a "no-observation" regime — currently
     hard to do, but possible in tests).

  3. **Order-dependent query results.** Two non-commuting locate
     queries on the same root region should yield different posterior
     states depending on order. Equivalent to QQ-equality if we can
     find an analog measurement.

  4. **Bounded total mint per coherence-window.** The total mint
     emitted per unit time should be bounded by a function of the
     decoherence rate Σⱼ γⱼ. A conservation law candidate.

Any of these, if measurable in the existing or extended substrate,
constitutes empirical support for the formalism.

## Open questions for Stage 2 and beyond

  - How small can the time-step dt be without exploding the cost?
    (Lindblad evolution is O(d³) per step with d the Hilbert dimension —
    each term in the master equation is a matrix-matrix product on
    d×d density matrices. For a 16×16 register (4 roots) that's
    trivially cheap. For 64×64 (6 roots) still fine. For 1024×1024
    (10 roots) we need to track marginals only, not the joint state,
    or the substrate becomes intractable.)

  - When does keeping the joint state matter vs. tracking marginals?
    (Marginals are always cheap; joint state is necessary only when
    Jᵢⱼ creates entanglement we care about.)

  - Does the existing equilibrate's behavior empirically match Lindblad
    evolution with reasonable γ choices? (Stage 3.)

  - Is the parameter-free invariant analog to QQ-equality available?
    (Stage 5.)
