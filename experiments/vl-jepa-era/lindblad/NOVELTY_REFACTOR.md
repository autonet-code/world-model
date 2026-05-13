# Novelty refactor: continuous two-way surprise

## What this changes

Novelty is currently a **per-evaluation measurement** computed by the
probe each time `tendency.evaluate(obs)` runs. Once an observation is
absorbed into the frame's `integrated` set, future evaluations of
similar observations get short-circuited at near-zero novelty. The
ratchet is one-way; once integrated, never un-integrated.

The refactor: novelty becomes **persistent per-node state**. Each
node carries a scalar `n ∈ [0, 1]` that:

  - decays under PRO observations (predicted state, less surprise),
  - regrows under CON observations (un-predicted state, re-surprise),
  - drifts upward slowly with no observations (entropy of certainty).

The probe still computes a per-evaluation novelty for the staking
decision; the per-node state is a *separate* signal that captures
the substrate's confidence in this region of its model.

## Semantic check: does it still capture surprise?

Surprise has three operational meanings the refactor must preserve:

  1. **Initial surprise.** First time we encounter a phenomenon in a
     region, novelty is high.
     → New node sprouts with n = 1.0. ✓

  2. **Confirmation reduces surprise.** Subsequent PRO observations
     consistent with our model in this region make us less surprised.
     → dn/dt has a -PRO term. ✓

  3. **Re-surprise on contradiction.** A model we thought was reliable
     gets challenged; our certainty crumbles; we're un-settled.
     → dn/dt has a +CON term, restoring novelty. ✓

  4. **Decay of certainty over time.** "I haven't checked this in a
     while; I might be wrong by now."
     → small +ε drift term. ✓

All four match the cognitive concept. The refactor is a *strict
generalization* of the current behavior: cases 1 and 2 are what
the substrate already does; cases 3 and 4 are new.

## The dynamics

For each node i with current novelty n_i ∈ [0, 1]:

    dn_i/dt = -γ_pro · n_i · ρ_pro_i(t)
             + γ_con · (1 - n_i) · ρ_con_i(t)
             + ε · (1 - n_i)

where:

  - ρ_pro_i(t) = rate at which PRO observations are arriving in
    node i's coordinate region. Locality kernel × observation rate.
  - ρ_con_i(t) = rate of CON observations in the region.
  - γ_pro, γ_con = base coupling rates (γ_pro typically larger so
    confirmation reduces uncertainty faster than contradiction
    restores it; otherwise the system is too jumpy).
  - ε = background drift rate (time-decay of certainty).

The form is deliberately the same shape as the Lindblad amplitude-
damping equation we derived in earlier stages:

    dα/dt = γ · (1 - α) · (CON drive) - γ · α · (PRO drive)

with `n` here playing the role of "1 - α" — population in the
"surprised" / coherent state. That's not coincidence; novelty IS
coherence in this picture.

## Steady-state

Setting dn/dt = 0:

    γ_con (1 - n) ρ_con + ε (1 - n) = γ_pro · n · ρ_pro

    n* = (γ_con · ρ_con + ε) / (γ_pro · ρ_pro + γ_con · ρ_con + ε)

Sanity check the limits:

  - **All PRO, no CON, no drift** (ρ_con = 0, ε = 0): n* = 0. Fully
    confirmed region has no novelty. ✓
  - **All CON, no PRO** (ρ_pro = 0): n* = 1. Region under sustained
    contradiction stays maximally surprising. ✓
  - **No observations** (ρ_pro = ρ_con = 0): n* = ε / ε = 1. With
    just drift, novelty climbs toward 1 — "I haven't seen this in a
    while; I'm uncertain." ✓
  - **Balanced PRO and CON** (ρ_pro = ρ_con = ρ, ε small): n* ≈
    γ_con / (γ_pro + γ_con). With γ_pro > γ_con (asymmetric: easier
    to confirm than challenge) this lands < 0.5 — region is mostly
    settled but still has residual surprise. Reasonable.

## Initial condition

A freshly sprouted node has no history. Set n_i(0) = 1 — we are
maximally surprised by this newly-distinguished claim. As observations
arrive in its region, n decays toward whatever the steady-state
balance dictates.

## Coupling to mint

Mint currently fires from the score-change × novelty product. Under
the refactor, mint uses the **per-node persistent novelty** rather
than the per-evaluation measurement. So:

    mint_event_strength_for_node_i = max(0, Δscore_i) · n_i · I(score_i > 0)

When n_i is high, score changes mint strongly. When n_i is low (region
has been heavily confirmed), even large score changes mint weakly —
because the substrate isn't surprised, so the network shouldn't reward
that "discovery" much.

Crucially: as n_i regrows due to CON pressure, future score changes
in this region mint MORE again. The substrate can re-discover
something that had been settled, and the network rewards that
re-discovery proportional to how much certainty was lost. This is
genuinely new substrate behavior.

## Coupling to capacity

Capacity is currently smooth-promotion: nodes earn outbound staking
voice based on accumulated PRO support. Under the refactor, capacity
should *also* be modulated by current novelty:

    effective_voice_i = capacity_i · (1 - n_i^k)

with k > 1 making well-confirmed nodes (low n) speak more, and
recently-shaken nodes (high n) speak less. A node whose novelty just
spiked from CON pressure should wait until things settle before
making strong sub-claims about neighbors.

This isn't strictly necessary for the refactor — capacity logic could
stay as-is — but it's a natural consequence of "novelty = how
unsettled this region is."

## Coupling to pruning

The existing pruning mechanism (StabilityTracker + is_decayed) marks
nodes that have been stable across many rounds. Under the refactor,
"stable" means n stays low for a long time. So the pruning condition
becomes:

    is_decayed(node_i) := n_i has stayed below threshold for window rounds

This is just a re-expression of the existing logic in terms of the
new novelty variable.

## Implementation surface

Three real changes to the substrate code:

  1. **Add `n: float` to Node** (or to a parallel per-node novelty
     dict on Tendency). Default 1.0 on sprout.

  2. **Add `update_novelty(world, dt)` method on Tendency** that
     walks all nodes and updates n per the dynamics above. Called
     once per round of equilibrate (or as often as we like; the
     dynamics are continuous so dt can be the round step).

  3. **Mint formula uses `n_i` from the node** instead of querying
     the probe. Capacity update optionally multiplies by (1 - n^k).

That's it. The probe stays unchanged. The frame's `integrated` set
stays unchanged (it still serves the per-evaluation novelty
measurement, which is a different thing — that measurement is for
deciding stance and stake direction, not for tracking surprise over
time).

So the refactor adds a parallel persistent-state layer; the probe
remains as the per-evaluation surprise *measurement*; mint now reads
from the persistent state.

## Self-check on the math

Conservation: nothing should be conserved exactly — n is a per-node
non-conservative state. ✓ (Confirmed by the rate equation having
no symmetric form.)

Boundedness: with γ_pro, γ_con, ε ≥ 0 and ρ_pro, ρ_con ≥ 0, the RHS
of dn/dt is non-positive when n = 1 (the +ε(1-n) term vanishes, and
the +γ_con(1-n)ρ_con term vanishes, leaving only -γ_pro n ρ_pro ≤ 0)
and non-negative when n = 0 (the -γ_pro n ρ_pro term vanishes,
leaving γ_con (1)ρ_con + ε(1) ≥ 0). So n stays in [0, 1] as long
as it starts there. ✓

Uniqueness of steady state: the RHS is linear in n with a negative
coefficient (γ_pro ρ_pro + γ_con ρ_con + ε ≥ 0), so the steady state
is unique and globally attracting whenever the coefficient is positive.
Marginal case: if all rates are zero (no observations, no drift), n
stays at its initial value indefinitely. Acceptable. ✓

Continuity in parameters: n* is a smooth function of γ, ρ, ε. ✓

## What this gives us

Two things:

  1. **A more honest cognitive model.** Surprise that can be restored
     by counter-evidence is closer to how cognition actually works.
     Settling is provisional; new evidence can un-settle.

  2. **A genuine quantum-cognitive correspondence.** With n behaving
     this way, mapping it to ρ_offdiagonal (coherence) is no longer
     a stretch. Both decay under confirmation and regrow under
     dissonance; both drift toward maximally uncertain in absence
     of observations; both are bounded in [0, 1]; both are
     non-conserved.

The Step A Hamiltonian + Step B predictions become testable against
substrate runs: track n_i(t) for several nodes during a substrate
session, check whether trajectories match the predicted decay and
regrowth rates. If they do, we have empirical evidence for the
quantum-cognitive correspondence. If they don't, we have a clean,
identifiable place where the substrate diverges from quantum
dynamics — and we can investigate that specific divergence rather
than chase the bridge through more layers.

## What's NOT changed

  - Tree topology
  - Stake mechanics (PRO/CON children, score propagation)
  - Content-addressed merge / federation
  - Locate primitive
  - Render primitive
  - The probe and the per-evaluation `composite` novelty (still used
    for staking decisions during `tendency.act`)
  - Frame's integrated set (still used by the probe)

## Two novelty signals, clearly

After the refactor the substrate has two novelty signals that must not
be confused:

  - **Per-evaluation novelty** (probe.composite, unchanged): how
    surprising IS THIS OBSERVATION relative to my current frame, right
    now. Used for staking decisions inside tendency.act — "this obs
    is novel and contradicts X, so I stake CON on X."

  - **Per-node persistent novelty** (the new n_i, refactored): how
    settled IS THIS REGION OF MY MODEL across time. Used for mint,
    pruning, capacity modulation, and the quantum-coherence
    correspondence.

Same word, different timescales. The probe answers an instantaneous
question; n_i answers a longitudinal one. Both are correct uses of
the word "novelty"; they capture different aspects of surprise.

## Open questions

  - **What are the right rate constants?** γ_pro, γ_con, ε all need
    values. Initial guess: γ_pro = 1.0, γ_con = 0.5, ε = 0.01 in
    units of "per round." Tunable; the steady-state formula tells
    us how the system responds to changes.

  - **Per-node novelty vs per-region?** Two nodes in the same
    coordinate neighborhood share information about that region.
    Should they share novelty? Probably not — each node is its own
    claim with its own surprise level. But adjacent nodes' novelty
    trajectories should be correlated through the locality kernel.

  - **How does sub-claim sprouting interact with novelty?** When a
    new sub-claim sprouts under a parent, does the parent's novelty
    change? My read: yes — sprouting a sub-claim is itself an event
    that should slightly bump the parent's novelty (the parent has
    been refined; some uncertainty was resolved by distinguishing a
    new sub-region; some new uncertainty introduced about how the
    sub-region relates back). Net effect probably small.

  - **Is it worth promoting novelty to a 2-vector** with separate
    "surprise from PRO confusion" and "surprise from CON contradiction"
    components? Probably not — adds complexity without clear payoff.

## Files this would touch

  - `c:\code\world-model\world_model\models\tree.py` — add `n: float`
    field to Node, default 1.0.
  - `c:\code\world-model\world_model\generalized\tendency.py` — add
    `update_novelty(world, dt)` method; sprout_child sets n=1.0;
    optionally modulate capacity by (1 - n^k).
  - `c:\code\world-model\world_model\generalized\equilibrate.py` —
    call `tendency.update_novelty(world, dt=1.0)` each round.
  - `c:\code\autonet\nodes\common\world_model_substrate\reconcile.py` —
    use `node.n` instead of probe-call when computing mint.
  - `c:\code\world-model\world_model\generalized\decay.py` —
    is_decayed uses persistent novelty state.

Substrate experiment files in `D:\videos\SF\manifesting\...\substrate_experiment\`
don't need changes; they consume the substrate's exposed API.
