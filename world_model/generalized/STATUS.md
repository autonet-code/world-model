# Generalized world model — status

A non-LLM, coordinate-space implementation of the formalization in
`docs/FORMALIZATION.md` and `docs/THEORY.md`.

## Current capability

Two demos work end-to-end:

  - `demo_two_tendencies.py`: 1D adversarial setting. Two tendencies on
    opposite sides of an axis. Balanced evidence ties them; tilted
    evidence produces a score gap. Confirms the architecture responds
    to evidence direction.

  - `demo_constraint_satisfaction.py`: 3-variable boolean SAT via
    coordinate-space encoding. 6/8 cases pass:
      - AND, NOT, OR, IMPLICATION (single-hop): all pass.
      - UNSAT detection (contradictory constraint produces zero gap):
        passes.
      - Underdetermined detection (unconstrained variable stays at
        zero gap): passes.
      - **Two-hop chained reasoning fails.** When `x->y; y->z; x=T`,
        x and z resolve correctly but y stays tied at +0.824 because
        c_imp1 supports y_T and c_imp2 (independently) supports y_F.

## Architecture (implemented)

  - **CoordinateFrame** (coordinate_frame.py): a non-LLM ReferenceFrame.
    sim = Gaussian kernel. stance = sign of projection onto polarity
    axis, gated by topical proximity. contains = exact-match in
    integrated set, with epsilon distance tolerance.

  - **CoordinateProbe**: NoveltyProbe that walks claim adjacency.
    Terminations: INTEGRATED / CONTRADICTS_ROOT / DISRUPTS / ORTHOGONAL
    / MAX_ITERATIONS.

  - **GeneralizedTendency** (tendency.py): owns a Tree + a Frame + a
    Probe + a budget. Acts by running the probe on observations and
    on other tendencies' nodes, then translating terminations into
    signed stakes.

  - **World** (world.py): coalition of tendencies + observation stream
    + stake graph. apply_stakes writes signed stakes to the right
    nodes; root_scores reads the equilibrium.

  - **equilibrate / equilibrate_with_growth** (equilibrate.py): rounds
    of (act, apply_stakes) until convergence, optionally with growth
    rounds between equilibrations.

  - **propose_growth** (grow.py): sprout PRO/CON children under nodes
    with mixed-sign stakes. **Currently fires almost never** because
    real contention manifests as tied root scores after equilibration,
    not mixed-sign stakes on individual nodes. Needs revision.

## Known gaps

1. **Two-hop chained reasoning** is the open problem. Single-pass
   constraint propagation produces ties when balanced evidence
   supports both sides of a variable. The fix needs growth to fire
   on equilibrium-tied tendencies, sprouting sub-claims that capture
   the conditional structure ('y=T conditional on x', 'y=F
   conditional on z').

2. **Growth rule is too crude.** `propose_growth` looks for nodes
   with both PRO and CON stakes, but real contention shows up as
   *tied root scores* after equilibrium, not raw stake mixing on
   individual nodes. A revised rule should:
     - Run after equilibration, not before.
     - Compare root scores between competing tendencies (e.g.
       y_T vs y_F) and fire when they're within a small gap.
     - Sprout sub-claims that encode conditional dependencies
       (which other variable's state distinguishes the cases).

3. **Frame absorption is permanent.** Once an obs is in
   `frame.integrated`, it's there forever and future probes return
   INTEGRATED at full sim. For time-windowed contexts (recent vs
   historical), we'll want a decay mechanism.

4. **No batched runtime.** Pure-Python per-observation loops. Fine for
   demos at this scale; won't scale.

## What's NOT carried forward from prior commits

The flat bipartite-graph classification work (iris/digits/letters
demos, contrast wiring, surprise-weighted input substitution, the
attempt at hierarchical cascade) is still in the repo but doesn't
exercise this architecture. It tested a degenerate shadow of the
real model — a single tree per class with no internal PRO/CON
structure. Those demos were valuable for finding what doesn't work;
the new generalized module is where forward progress happens.

## Next research threads (in rough order)

1. **Equilibrium-tied growth rule**: implement contention detection
   on root-score gaps and sub-claim sprouting that conditions on
   independently-resolved variables. Gate: pass case 7 of the SAT
   demo (two-hop reasoning).

2. **Imputation in the generalized model**: given partial-state
   observation (some variables fixed, others unknown), can the
   architecture predict the unknown? Maps to the existing imputation
   work but tests whether the new architecture handles it without
   hand-derived feature wiring.

3. **Larger SAT problems**: 5+ variables, more constraints. Test
   whether the architecture scales gracefully or whether the depth
   requirement explodes.

4. **Decentralization properties**: with the architecture working at
   3 variables, write up the genuinely decentralization-friendly
   properties (no central log, no gradients, statistics merge
   cleanly via running stats, lineage attestation). The
   intelligence-threshold doc's economic question can only be
   addressed once the architecture solves something nontrivial; SAT
   at scale is the soonest credible benchmark.

## File map

```
world_model/generalized/
├── __init__.py            -- public API
├── observation.py         -- Observation data class
├── coordinate_frame.py    -- CoordinateClaim, CoordinateFrame, CoordinateProbe
├── tendency.py            -- GeneralizedTendency
├── world.py               -- World coalition
├── equilibrate.py         -- equilibrate, equilibrate_with_growth
├── grow.py                -- propose_growth (needs rework)
└── STATUS.md              -- this file
```

```
demo_two_tendencies.py            -- 1D adversarial demo (passes)
demo_constraint_satisfaction.py   -- SAT demo (6/8 pass)
```
