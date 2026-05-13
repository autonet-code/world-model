# Step A — Hamiltonian from sub-claim configuration

The substrate's classical update rule is wrong as a description of
what we actually want it to compute. Strip away the discrete-iteration
scaffolding and there's a continuous-time generative process that the
substrate is approximating. This document derives the Hamiltonian
H_root for that process, given the static state of a root and its
sub-claim subtree at some point in time.

The dynamics, in full, are

    dρ_root/dt = -i [H_root, ρ_root] + Σⱼ γⱼ D[Lⱼ](ρ_root)

H_root is what generates the unitary part — the part that runs
between observations, encoding how the existing graph structure pulls
the root's score around. Observations are the Lⱼ jump operators,
applied at discrete moments. Between jumps, ρ_root evolves under
H_root.

This document specifies H_root.

## The basis

A root has a polarity axis. Project everything onto the {|PRO⟩, |CON⟩}
basis. ρ_root is 2×2. Score α = ⟨PRO|ρ_root|PRO⟩.

## Three contributions to H_root

A root's dynamics are driven by three things, all of which sit in the
existing substrate but are currently encoded as scalar adjustments
rather than as Hamiltonian terms:

1. **Net stake imbalance.** Sub-claims are PRO or CON. Their net signed
   contribution biases the root toward one pole. This becomes a σ_z
   term — a static energy splitting that pulls toward whichever pole
   has more weight.

2. **Unresolved cross-pole stake.** When a root has both PRO and CON
   sub-claims with comparable weight, neither dominates. The classical
   substrate represents this as "net score near zero." But that loses
   information: a root with PRO=CON=0 (no stakes) and a root with
   PRO=CON=10 (maximum staked but balanced) are not the same. The
   second has tension; the first does not. This becomes a σ_x term
   — a transverse field that drives oscillation between the two
   poles. Roots with high tension oscillate; roots with no tension
   sit still.

3. **Cross-root coupling via shared sub-claim coordinates.** Two roots
   whose sub-claims occupy the same coordinate neighborhood influence
   each other (the locality rule). This becomes σ_z ⊗ σ_z coupling
   — Ising-type interactions between roots, decaying with coordinate
   distance.

We write each term separately, then assemble.

## Notation

For a single root, let:

  - S = {s₁, s₂, ..., s_n} = signed stakes on direct sub-claims of
    the root (s_i > 0 for PRO, s_i < 0 for CON; magnitude = stake).
  - For each sub-claim i, let cap_i = its capacity (how settled it is;
    cap_i ∈ [0, 1], with 0 meaning fully unsettled and 1 meaning fully
    settled).
  - Let coord_i = sub-claim i's anchor coordinate.

For two roots at indices a and b:

  - For a sub-claim i of root a and sub-claim j of root b, let
    d(i, j) = ‖coord_i − coord_j‖ in the substrate's coordinate space.

## Term 1 — net stake bias (σ_z)

The simplest term. Each sub-claim i with stake s_i contributes a
σ_z term proportional to s_i × cap_i:

    H_z = -ω · σ_z

with

    ω = κ · Σᵢ s_i · cap_i

κ is a unit-conversion constant (probably ~1 in dimensionless units).
The product s_i · cap_i is "signed stake weighted by how settled this
sub-claim is" — a sub-claim that's fully settled at PRO contributes
+s_i to ω; an unsettled one contributes less.

The sign convention. In the {|PRO⟩, |CON⟩} basis, σ_z = diag(+1, -1).
With H_z = -ω σ_z, the energy of |PRO⟩ is -ω and |CON⟩ is +ω. With
ω > 0 (PRO-staked sub-claims), |PRO⟩ has *lower* energy and is
the preferred state under damping — which is what we want. The
minus sign is what couples "PRO-staked sub-claims pull the root
toward PRO" correctly.

Sanity check:

  - All-PRO sub-claims, all settled: ω > 0. |PRO⟩ at energy -ω
    (lower), so steady state under damping settles at α = 1. ✓
  - All-CON: ω < 0. |PRO⟩ at energy -ω = positive, so |CON⟩ is
    lower-energy. Steady state at α = 0. ✓
  - Empty: ω = 0. Hamiltonian vanishes, α = 0.5 under symmetric
    decoherence. ✓
  - Mixed PRO/CON canceling out: ω = 0, but Term 2 (next) takes over.

## Term 2 — unresolved cross-pole stake (σ_x)

When a root has stakes pulling in both directions, even if they cancel
in net, there's *latent tension*. The substrate currently has no way
to express this — once net_score = 0, the score is "settled at neutral."
But intuitively a root with PRO=CON=10 is *harder to settle* than a
root with no stakes at all: any new observation will swing it more
than it would swing the empty root.

We capture this with a transverse-field (σ_x) term whose magnitude
is proportional to the *unresolved magnitude* — the geometric mean
of total PRO and total CON stake (which is large only when both are
substantial):

    H_x = ζ · σ_x

with

    ζ = λ · sqrt(P · C)

where:

  - P = Σᵢ max(s_i, 0) · cap_i  (total settled-weighted PRO stake)
  - C = Σᵢ max(−s_i, 0) · cap_i  (total settled-weighted CON stake)
  - λ is another unit-conversion constant.

Geometric mean is the right choice: it's zero when either P or C is
zero (no tension when only one pole is staked), and grows symmetrically
when both grow. Arithmetic mean would have ζ > 0 even when one pole is
absent, which doesn't capture "tension."

Effect of σ_x on the dynamics: in the σ_z basis (PRO/CON), σ_x
mixes the two states. A root that starts in |PRO⟩ under H = ζ·σ_x
oscillates between PRO and CON at frequency 2ζ. Combined with H_z,
the eigenvalues become ±√(ω² + ζ²), and the eigenstates tilt away
from pure PRO/CON by an angle θ = ½ arctan(ζ/ω).

Sanity check:

  - All-PRO (P > 0, C = 0): ζ = 0. Pure σ_z dynamics, no oscillation.
  - All-CON: same. ✓
  - Balanced (P = C = high): ω = 0, ζ = high. Pure σ_x dynamics.
    Score oscillates rapidly. New observations matter a lot — small
    perturbations get amplified. This matches the "high-tension"
    intuition.
  - PRO dominant but CON nontrivial: ω > 0, ζ moderate. Eigenstates
    tilted toward PRO but with quantum-beat oscillations during
    relaxation. Predicted: α(t) approaches its steady value with
    damped oscillations on top of exponential decay — a falsifiable
    signature.

## Term 3 — cross-root coupling (σ_z^a ⊗ σ_z^b)

When sub-claim i of root a and sub-claim j of root b sit at nearby
coordinates, they influence each other. Currently this is the
"locality rule" — sub-claim staking from a's tree to b's tree is
bounded by coordinate distance.

We promote this from a substrate hack to a physical interaction term.
For two roots a and b, the cross-coupling is:

    H_int(a,b) = -J_ab · σ_z^a ⊗ σ_z^b

with

    J_ab = μ · Σᵢ Σⱼ s_i^a · s_j^b · cap_i · cap_j · K(d(i, j))

where K is the locality kernel — a function that's large at small
distance and small at large distance. The natural choice is

    K(d) = exp(-d² / (2 σ_loc²))

with σ_loc = the bandwidth from the existing substrate (for the
charter, that's 0.5 in coordinate units).

The minus sign in H_int matches the Ising convention. With H = -J σσ:
  - J > 0 → ferromagnetic: aligned states |PROPRO⟩, |CONCON⟩ at
    energy -J (lower), preferred under damping. Roots tend to agree.
  - J < 0 → antiferromagnetic: anti-aligned states preferred. Roots
    tend to disagree.

Sanity check:

  - No nearby sub-claims between a and b: J_ab = 0. Roots evolve
    independently. ✓
  - Both roots have nearby PRO sub-claims (s_i and s_j both positive):
    J_ab > 0. Ferromagnetic. Aligned PRO-PRO and CON-CON are lower-
    energy. Roots tend to settle in the same direction. ✓
  - Roots have opposing nearby sub-claims (s_i positive, s_j negative):
    J_ab < 0. Antiferromagnetic. Anti-aligned states preferred. Roots
    tend to settle in opposite directions. ✓ — "this sub-claim
    supports root a; the same coordinate region rejects root b."

## Full Hamiltonian for an N-root substrate

Pull it all together:

    H = Σₐ (-ωₐ σ_z^a + ζₐ σ_x^a) + Σ_{a<b} (-J_ab) (σ_z^a ⊗ σ_z^b)

This is a transverse-field Ising-like Hamiltonian on N qubits, with
parameters (ωₐ, ζₐ, J_ab) computed from the substrate's classical
state.

For the four-charter case (N=4), the joint Hilbert space is 16-D.
Writing H in matrix form requires building 16×16 matrices via
Kronecker products of σ_z, σ_x, and identity. Tractable.

## Observations as Lindblad jumps

Already covered in LINDBLAD_SCAFFOLD.md: an observation aligned with
root a's PRO axis is L = √γ · |PRO⟩⟨CON|^a ⊗ I^{rest}, applied as
a jump that increments PRO population at rate γ. CON-aligned obs
is the symmetric form. The rate γ encodes:

    γ = stake_mag × novelty × outcome_strength × locality_weight_to_root

This is exactly the product the substrate already computes for the
discrete `mag` value, just renamed as a rate.

## What's NOT in the Hamiltonian

The Hamiltonian is **static** between observation events. It encodes
the existing graph structure. Things that *change* the substrate
state — sprouting new sub-claims, pruning dead nodes, content-
addressed merges from federation — are events that **modify H**, not
terms within H.

So a full simulation step is:

  1. Compute H from current graph state.
  2. Compute Lⱼ for any incoming observations.
  3. Evolve ρ over a time interval Δt under H plus Lⱼ jumps.
  4. After Δt, recompute graph state: process any sprout-or-prune
     events that the equilibrated ρ implies. (This is where the
     classical-discrete part lives.)
  5. Recompute H from the new graph state.
  6. Repeat.

The continuous-time evolution lives in steps 1-3. The discrete graph
events live in step 4. Both are necessary.

## Free parameters

We've introduced three constants:

  - κ — couples stake to ω. Sets the energy scale of the bias term.
  - λ — couples sqrt(P·C) to ζ. Sets the energy scale of tension.
  - μ — couples cross-root products to J_ab. Sets the energy scale
    of cross-coupling.

These are dimensional constants. In a fully-dimensionless substrate
they could all be 1; if we want to match observed substrate timescales,
we'd fit them to data.

The locality bandwidth σ_loc is reused from the existing substrate.

## Falsifiable predictions specific to this Hamiltonian

1. **Damped quantum beats during settling.** A root with mixed PRO/CON
   stake (ω small, ζ large) under added observations should show
   oscillatory α(t) with envelope decaying at rate Σⱼ γⱼ. The current
   classical substrate cannot produce this; if we observe it in a
   continuous-time implementation, that's evidence the formalism is
   capturing a real phenomenon.

2. **Tilted steady states.** Under finite ζ, the steady-state
   eigenstate is not pure |PRO⟩ or |CON⟩ but a tilted superposition.
   Score asymptote ≠ γ_+/(γ_+ + γ_-) — there's an additional
   correction from ζ. Specifically:

       α_steady(ω, ζ, γ) = γ_+ / (γ_+ + γ_-) + correction(ω, ζ, γ)

   We'd compute the correction and check that observed α matches.

3. **Charter entanglement signatures.** Two roots with overlapping
   sub-claim coordinates (J_ab ≠ 0) should exhibit correlated
   relaxation. Specifically, after applying an observation to root a,
   root b's α should also shift, with magnitude proportional to J_ab.
   In the classical substrate this happens via cross-staking; in the
   Lindblad picture it's automatic from the joint state. The
   prediction: the correlation strength matches J_ab quantitatively.

4. **Reversibility in the unitary regime.** With observations turned
   off (γ = 0), the substrate should be reversible — the joint state
   evolves under e^{-iHt} and can be evolved backward to recover any
   prior configuration. The classical substrate is irreversible
   (sprouting and pruning can't be undone). This is a prediction we
   can't test in the current code but defines the difference between
   the continuous and discrete pictures.

## Open questions

- **What's the right time unit?** ω, ζ, J_ab all have units of
  inverse time (we're setting ℏ = 1). The substrate currently doesn't
  have a natural time unit because each "round" of equilibrate
  collapses dynamics. We'd pick Δt = 1 per observation as the natural
  scale, then the constants κ, λ, μ are fit from the observed
  per-observation effect.

- **How does graph growth interact with H?** A new sub-claim sprout
  *adds* terms to H. Strictly speaking this is an open system whose
  Hilbert space dimension is growing. The clean way is to fix the
  Hilbert space (= number of roots, N) and let sub-claims modify the
  parameters of the existing H rather than enlarging the qubit
  register. (This was already the design choice in LINDBLAD_SCAFFOLD.md
  but worth re-stating.)

- **What about novelty decay?** Currently capacity decays with each
  round. In Hamiltonian language: cap_i contributes to H linearly,
  so a decaying cap_i means H itself drifts with time. That's
  technically a time-dependent Hamiltonian, ok but more complex to
  simulate than time-independent. We could absorb the decay into the
  Lindblad dissipator instead: a "novelty exhaustion" channel that
  damps sub-claim weights over time.

## Self-check: do the limits work?

- **Empty substrate** (no sub-claims): ω = ζ = J = 0. H = 0. Pure
  dissipator dynamics from observations alone. Each observation pulls
  the root toward its pole at rate γ. Score relaxes exponentially.
  This is the Lindblad scaffold's basic case from LINDBLAD_SCAFFOLD.md.
  Matches. ✓

- **Single PRO sub-claim, no observations**: ω > 0, ζ = 0, no jumps.
  Pure σ_z dynamics. Eigenstates are |PRO⟩ and |CON⟩. State that
  starts at |PRO⟩ stays there; state at |CON⟩ stays there. State at
  superposition oscillates in phase but populations don't change.
  This is the "no observations have arrived to break the symmetry"
  case — score is stuck. The substrate's behavior should match:
  without observations, scores don't move. ✓

- **Mixed sub-claims, no observations**: ω small, ζ large, no jumps.
  Coherent oscillation between PRO and CON at frequency 2ζ. Score
  oscillates without settling. Falsifiable: the substrate, run in
  the continuous limit with no observations, should oscillate
  whenever it has mixed stake. The classical substrate can't show
  this because equilibrate runs to fixed point; the continuous
  implementation should.

- **All-PRO sub-claims, observations arriving**: ω > 0, ζ = 0, plus
  damping toward PRO. Score relaxes monotonically toward 1 at rate γ
  (modulo a slow drift from ω). Same as the basic Lindblad case but
  with bias. Sanity-checked. ✓

- **Two roots with strong cross-coupling**: J ≠ 0, observations on
  root a alone. Root b's score should shift even without direct
  observation. The shift magnitude is proportional to J_ab. In the
  classical substrate this happens via cross-staking; in the Lindblad
  picture it's automatic. Predicted to match.

The math checks out. Two main sanity tests pending:

  - Numerical simulation on a hand-built 2-root substrate with
    cross-coupling (to verify the entanglement-driven score shift).
  - Comparison of damped-quantum-beat trajectory against substrate
    behavior on a high-tension root configuration.

Both are Step B work.

## What this gives us

If Step B confirms this Hamiltonian behaves as predicted on hand-built
test cases, we have:

1. A closed-form continuous-time generalization of the substrate's
   classical scoring rule.
2. A falsifiable theory: the substrate *should* show damped beats,
   tilted steady states, and entanglement-driven correlations. If a
   continuous implementation does, the formalism captures real
   phenomena. If not, we revise.
3. A genuine quantum-implementation path: the substrate as a
   transverse-field Ising model with Lindblad observations. Quantum
   hardware can run this natively.
4. The discrete classical algorithm as a *coarse-grained* approximation
   to the continuous evolution. Useful for cheap classical operation;
   not the underlying truth.
