# Step C — bridge findings

## What was built

`equilibrate_continuous(world, t_total, dt, ...)` reads a substrate
world, derives Hamiltonian parameters (omega, zeta, J) per the Step A
formulas, evolves the joint root state under a Lindblad master
equation, and writes resulting alpha values back as net_score
adjustments.

The classical `equilibrate` and the new `equilibrate_continuous` were
run side-by-side on three scenarios. Results in `step_c_results.json`
and visualized in `step_c_comparison.png`.

## Findings

### Scenario 1 — 10 PRO observations on a single root

| step | classical alpha | continuous alpha |
|------|-----------------|------------------|
|   0  |          0.500  |          0.500  |
|   5  |          0.571  |          0.960  |
|  10  |          0.639  |          0.997  |

Classical climbs linearly (the discrete-counter behavior identified
in Stage 3). Continuous saturates to ~1.0 within a few steps because
omega grows rapidly with each new sub-claim and the Lindblad evolution
pulls strongly toward PRO.

### Scenario 2 — alternating PRO/CON

| step | kind | classical alpha | continuous alpha |
|------|------|-----------------|------------------|
|   0  |   -  |          0.500  |          0.500  |
|   2  | CON  |          0.391  |          0.672  |
|   4  | CON  |          0.381  |          0.797  |
|   6  | CON  |          0.371  |          0.875  |
|   8  | CON  |          0.361  |          0.924  |
|  10  | CON  |          0.352  |          0.953  |

Classical drifts toward CON because each CON observation acquires a
permanent +0.5 stake via the `_con_positioned` mechanism, and the
absorption asymmetry means PRO and CON contributions don't cancel.

Continuous monotonically climbs to PRO. Reason: each PRO observation
sprouts a PRO sub-claim, each CON sprouts a CON sub-claim. As both
counts grow, omega = sum(stake * cap) grows linearly, but PRO stake
magnitudes happen to be larger here (capacity differences), so omega
> 0 always. The Lindblad evolution then pulls toward PRO with strength
proportional to omega, and as omega grows, the equilibrium alpha
saturates at 1.

Neither answer is what we want. A balanced PRO/CON sequence should
produce alpha approximately 0.5, not drift to 0.35 (classical) or
saturate at 0.95 (continuous).

### Scenario 3 — two roots with shared coordinates

| step | kind | classical alpha | continuous alpha (root a / b) |
|------|------|-----------------|-------------------------------|
|   0  |   -  |          0.500  |          0.500 / 0.500  |
|   5  | PRO  |          0.599  |          0.960 / 0.959  |
|  10  | CON  |          0.402  |          0.954 / 0.951  |

Roots a and b track each other in the continuous model — confirms
J coupling is working, the two roots are entangled (or rather, their
marginal alphas are highly correlated through the Ising coupling).

But continuous gets stuck near 0.95 even after 5 CON observations
arrive. The first 5 PRO observations sprouted strong PRO sub-claims
that drive omega ≈ 5. The 5 CON observations sprout CON sub-claims
with smaller magnitudes (because of capacity differences and how the
substrate sprouts CON children only via a different path). omega
stays positive and large; CON jump operators on the observations
can't overcome it.

## Diagnosis

The Step A Hamiltonian formula says `omega = kappa · sum(s_i · cap_i)`.
This is **unbounded**: as more sub-claims accumulate, omega grows
linearly. A root that has acquired 50 PRO sub-claims has omega ≈ 50,
and the time scale of relaxation is 1/omega ≈ 0.02 — observations at
rate gamma ~ 0.5 can't move it.

The substrate represents accumulated argumentation in its graph, but
ω in the Hamiltonian should not encode the *magnitude* of evidence —
it should encode the *direction*. A root with 50 PRO sub-claims and
0 CON sub-claims has the same direction as a root with 5 PRO and 0
CON; it's just more confident. Confidence should affect *coherence*
(how easy it is to disturb the root), not the *Hamiltonian frequency*
(how fast it oscillates).

The fix: normalize omega by the total stake magnitude, so omega
represents direction:

    omega = kappa · sum(s_i · cap_i) / (sum(|s_i| · cap_i) + epsilon)

This bounds omega to roughly [-kappa, kappa] regardless of how many
sub-claims accumulate. The total accumulated weight then enters
through a separate "stiffness" parameter — perhaps zeta should
*decrease* as confidence accumulates (heavily-evidenced roots have
less tension), or the dissipator's gamma should be *modulated* by
how strongly the observation aligns with existing structure.

## What's still right

The Step A predictions (Step B testing) all hold for **hand-built**
sub-claim configurations with bounded parameters:
  - Damped quantum beats ✓
  - Tilted steady states ✓
  - Cross-root entanglement ✓
  - Reversibility ✓

The bridge's failure is in the **mapping from substrate state to
Hamiltonian parameters**, not in the Hamiltonian dynamics themselves.
We have the right formalism; we have the wrong parameter map.

## What to do next

Step D: rebuild the parameter mapping with normalization. Three
candidate formulations:

  **(D1) Normalize omega:** omega measures direction in [-1, 1];
  separate "confidence" parameter kappa_eff modulates dissipator rates
  such that confident roots resist observations.

  **(D2) Cap omega and decay over time:** omega_eff = tanh(omega/scale)
  with scale chosen so reasonable substrate configurations land in
  [-1, 1].

  **(D3) Make sub-claims decay:** subclaim capacity decays over time
  unless reinforced, so omega doesn't accumulate forever. This already
  exists in the substrate as `capacity_decay` but isn't applied to the
  raw stake weight.

Option D1 is the cleanest physically — it separates "which way" from
"how much." Test it as the next iteration before moving on to anything
else.

## Files produced

  - `lindblad/equilibrate_continuous.py` — the bridge
  - `lindblad/step_c_compare.py` — three-scenario comparison runner
  - `lindblad/step_c_results.json` — per-step alpha for both kernels
  - `lindblad/step_c_comparison.png` — plot

## Honest assessment

The Step A math is correct as math. The Lindblad kernel is correct
numerically. The Step B predictions hold for hand-built configurations.
The Step C bridge needs a parameter-normalization fix before
substrate state can be cleanly evolved by the continuous model.
That's a clean, identifiable issue with a clean fix path. Not a
disqualifying problem.
