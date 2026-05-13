# Lindblad: continuous-time score dynamics for the substrate

Research artifacts that informed `world-model/world_model/generalized/lindblad.py`
and `equilibrate_continuous`. The math, the failed bridges, and the
empirical findings that pointed at the right architectural call.

## What's here

### Math + math-as-spec

  - `STEP_A_HAMILTONIAN.md` -- derivation of H from sub-claim
    configuration, with self-checked sign conventions and limit cases.
  - `NOVELTY_REFACTOR.md` -- derivation of n as continuous two-way
    coherence variable. Now landed in world-model
    (commit on smooth-promotion adding Node.n + Tendency.update_novelty).

### Standalone numerical kernels

  - `lindblad_kernel.py` -- pure-numpy Lindblad master-equation
    integrator. RK4 with re-Hermitization. 8/8 unit tests passing
    against textbook two-level systems (Rabi, dephasing, amplitude
    damping, two-channel steady state, Ising-coupled marginals).
  - `test_lindblad_kernel.py` -- the 8 tests.
  - `step_b_predictions.py` -- the four falsifiable Step A predictions
    (damped beats, tilted steady states, cross-root entanglement,
    reversibility). All confirmed on hand-built configs.

### Empirical bridge attempts

  - `stage3_substrate_trace.py` -- first attempt to fit substrate
    alpha(t) to Lindblad evolution. Result: classical substrate is not
    Lindblad-shaped under its current update rule.
  - `step_c_compare.py` -- runs classical equilibrate vs continuous
    bridge on three scenarios. Reveals omega-unboundedness issue.
  - `step_d_params.py` -- bounded-direction-plus-confidence formulas.
    Helps but doesn't close the gap with classical.
  - `phase_1_with_novelty.py` -- the decisive test: rho_0 with
    novelty-as-coherence. Three arms (classical, continuous-no-novelty,
    continuous-with-novelty). Shows continuous-with-novelty is its
    own thing, not "classical's reflection."

### Findings docs

  - `STAGE3_FINDINGS.md` -- the substrate is not Lindblad-shaped under
    its discrete update rule.
  - `STEP_C_FINDINGS.md` -- omega unbounded, parameter map needs fix.
  - `STEP_D_FINDINGS.md` -- direction/confidence split helps but
    doesn't unify the kernels.

## Decision

After all of the above: continuous-with-coherence is the cognitive
shape we want (resist-then-yield-decisively under sudden contradiction).
The classical equilibrate is the lossy approximation.

Implemented in world-model on `feature/lindblad-equilibrate`:
  - `world_model/generalized/lindblad.py` (engine kernel)
  - `world_model/generalized/equilibrate.py::equilibrate_continuous`
  - `test_lindblad_equilibrate.py` (S3 documenting test, 3/3 passing)

Discrete `equilibrate` stays the default. `equilibrate_continuous`
is opt-in for callers wanting quantum-cognitive dynamics.

## Companion repos

See `D:/videos/SF/BRANCHES.md` for cross-repo branch tracking.
