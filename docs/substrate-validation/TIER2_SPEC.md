# Tier 2 spec: resist-then-yield-decisively at consensus scale

## What this tests

Whether the substrate's existing Lindblad kernel produces
substrate-native consensus when N novelty-bearing agents post
contradictory observations on the same claim. Specifically:

  - Does the resist-then-yield-decisively dynamic settle within
    bounded time?
  - Does settling time scale sub-linearly with N (coherent
    damping kicks in)?
  - Does the final population alpha track the input ratio with a
    "decisiveness gain" — tilts amplified into clear winners?
  - Does the substrate produce the same equilibrium under
    different decision-time structures (one big pulse vs. drip
    vs. asynchronous)?

## Why this is the right Tier 2

The substrate's Tier 0/1 work validated:
  - Architecture composes (3 roots, veto, dedup)
  - LLM-as-embedder gives correct verdicts on clear cases
  - Cheap-LLM lower bound is real

Tier 2 asks the next architectural question: **does the same
substrate carry consensus when agents disagree at scale?** Per
the prior turn's framing:

  - "Honest" is the wrong split. Agents either carry novelty
    potential (a real reading of the situation) or they don't.
  - Disagreement at scale isn't a failure mode — it's the
    Lindblad kernel doing its job.
  - The right question isn't "does it pick the right answer"
    (there is no external truth) but "how does it settle?"

The Lindblad kernel was built for exactly this shape (resist-
then-yield-decisively, tilted steady states, damped quantum
beats under high tension). This experiment runs that kernel at
deployment scale.

## Setup

### The substrate

Single tendency, single sub-claim under it. The sub-claim is the
"contested point" — agents post PRO or CON on it.

```python
correctness = GeneralizedTendency(
    id="correctness",
    thesis="The contested claim.",
    anchor=(1.0,),
    polarity_axis=(1.0,),
    bandwidth=0.7,
    veto_shaped=False,   # we're testing settling, not veto
    novelty_gamma_pro=1.0,
    novelty_gamma_con=1.0,   # symmetric; no asymmetric veto
    novelty_drift=0.01,
)
```

A single sub-claim sprouted at coords (1,) under the root.
Agents post PRO or CON on this sub-claim by emitting
observations.

### The agents

Agents are simulated. Each agent has:
  - id: unique identifier
  - position: PRO or CON (their stance on the contested claim)
  - cognitive_cost: per-post effort (proxy for "how hard they
    thought before posting"). Default 1.0 — uniform cost.

Each round, each active agent emits an observation reflecting
their position. The observation has coords matching the
sub-claim (so it gets posted on it) with sign matching position.

### The experimental conditions

**Axis 1 — N (population size)**:
  - N = 10, 100, 1000

**Axis 2 — split (PRO/CON ratio)**:
  - 50/50 (genuinely contested)
  - 60/40 (mild lean)
  - 80/20 (strong lean)

**Axis 3 — temporal structure**:
  - "pulse": all agents post in round 1, then quiet for 99 rounds
  - "drip": agents post one per round across 100 rounds
  - "async": each agent posts in a random round in [1, 100]

3 × 3 × 3 = 27 configurations. Run each once for the headline
experiment; repeat the most interesting (50/50 + drip + various
N) with multiple seeds for variance estimate.

### What gets measured

For each configuration, run for T_total = 100 rounds (or until
settled, whichever is later). Snapshot per round:

  - round_idx
  - per-node n (the contested sub-claim's n)
  - per-node net_score (the contested sub-claim's score)
  - alpha = sigmoid(net_score) (population in PRO state)
  - posts_this_round_pro, posts_this_round_con (drive)
  - posts_total_pro, posts_total_con (cumulative)

Derived metrics:

  - **settle_time**: first round where d(alpha)/dt < 0.001 for 5
    consecutive rounds (the kernel has equilibrated)
  - **final_alpha**: alpha at T_total
  - **transient_max_amplitude**: max |alpha(t) - final_alpha|
    during the run
  - **decisiveness_gain**: |final_alpha - 0.5| / |input_ratio - 0.5|
    where input_ratio = posts_total_pro / (posts_total_pro +
    posts_total_con). gain > 1 means tilts amplify; gain ≈ 1
    means linear; gain < 1 means dampening (probably wrong).

Run both kernels on each configuration:

  - **discrete kernel** (`equilibrate`): the cheap reference
  - **continuous kernel** (`equilibrate_continuous`): the
    Lindblad version, expected to show resist-then-yield-decisively

Compare the trajectories.

## Predictions

Six falsifiable predictions covering the architecture's claim:

  **C1 (settles within bounded time):** Every configuration
  reaches settle_time < 80 rounds (within the 100-round run).
  No oscillation forever.

  **C2 (settle time scales sub-linearly in N):** For fixed
  split (e.g. 50/50), settle_time(N=1000) < 10 × settle_time(N=10).
  Coherent damping kicks in faster at scale, not slower.

  **C3 (final alpha tracks input ratio):** For each config,
  final_alpha is on the same side of 0.5 as the input ratio
  (PRO majority → final_alpha > 0.5). No flips.

  **C4 (decisiveness gain on tilted inputs):** For 60/40 inputs,
  decisiveness_gain ≥ 1.5 in the continuous kernel (a 10%
  tilt amplifies into a ≥15% effective tilt). The kernel is
  doing its yield-decisively job.

  **C5 (50/50 lands near 0.5 with bounded jitter):** For genuine
  contests (50/50 input), final_alpha is in [0.4, 0.6]. The
  kernel doesn't artificially break ties.

  **C6 (continuous kernel diverges from discrete on tilted
  cases):** For 60/40 and 80/20, RMSE between continuous and
  discrete trajectories > 0.05. The Lindblad kernel produces
  the "decisive yield" shape that the discrete classical kernel
  lossily approximates.

## Outputs

  - `tier2_status.json` — live progress (which N/split/temporal
    config currently running)
  - `tier2_results.json` — full per-config trace (per-round n,
    score, alpha, posts) + derived metrics + predictions
    PASS/FAIL
  - `tier2_plots/` — one plot per (N, split) showing:
      - alpha(t) for each temporal structure × {discrete,
        continuous} kernel
      - vertical line at settle_time
      - horizontal line at final_alpha
  - PASS/FAIL print at end

## Implementation notes

### What we reuse

  - `world_model.generalized.equilibrate` (discrete kernel)
  - `world_model.generalized.equilibrate.equilibrate_continuous`
    (Lindblad kernel)
  - The substrate's natural post-and-coparent + dedup behavior

### What's new

  - `tier2_agents.py`: simulated agent class, schedule generators
    (pulse / drip / async), one_round_step
  - `run_tier2.py`: configuration matrix runner, metrics
    extraction, predictions, plotting

### Cost shape

  - 27 configurations × 2 kernels = 54 substrate runs
  - Each run: T_total = 100 rounds × O(N) work per round
  - For N=1000, that's 100K post operations per run
  - Substrate ops are sub-millisecond; total ~5-10 minutes for
    everything
  - No LLM calls. Pure substrate dynamics.

### Pre-flight checks

  - Confirm Lindblad kernel converges on a 1-tendency, 1-subclaim
    shape (the existing test_lindblad_equilibrate uses a
    different shape; quick smoke first)
  - Confirm alpha computation is stable for N up to 1000

## Success / failure / unknown

  - **All 6 predictions hold:** the substrate carries consensus.
    Decentralized inference / training story is architecturally
    grounded. We can build the deployment narrative.
  - **C1 fails (oscillates forever):** kernel parameters are
    wrong for this shape. Likely γ_pro/γ_con balance or ε too
    low. Tunable.
  - **C2 fails (settle time grows linearly with N):** the
    kernel's coherent component isn't damping fluctuations as
    expected. Investigate Lindblad coupling J.
  - **C3 fails (alpha flips):** the kernel is finding
    non-input-tracking equilibria. Indicates a Hamiltonian sign
    error or a drift term that's overpowering the input.
  - **C4 fails (decisiveness ≤ 1):** the kernel doesn't amplify
    tilts. The continuous-with-coherence claim is wrong; the
    Lindblad kernel reduces to classical and we should drop it
    or fix it.
  - **C6 fails (continuous ≈ discrete):** the kernel isn't
    actually doing anything different from the classical
    update. Same as C4 — investigate.

## Estimated effort

  - Spec: done (this doc)
  - Agent simulator: ~1 hour (small, just emits observations)
  - Runner: ~1 hour (configuration matrix, metrics, plots)
  - Run: ~10 minutes (no LLM calls)
  - Diagnose: 1-2 hours if predictions surprise

Total: 4-6 hours from spec-OK to verdict.

## What this does NOT test

  - Adversarial behavior. Sybil resistance + cognitive-cost
    equalizer arguments handle that conceptually; this
    experiment is about novelty-bearing disagreement, not
    attack.
  - Identity / provenance. Each "agent" is just a unique
    agent_id label; we trust the simulator. The gossip layer
    handles real provenance upstream.
  - Cross-tendency dynamics at scale. Tier 2 is single-tendency
    on purpose; multi-tendency consensus is Tier 3.
  - Network partitions / liveness under partial failure. Out
    of scope; we assume all posts arrive.
  - Production cost analysis. The N=1000 simulation is local;
    real-world costs depend on gossip + storage, not substrate
    compute.

## What this experiment buys

If predictions hold, **we have empirical grounding for
"substrate-native consensus"**: a substrate post-and-coparent
graph with the Lindblad equilibrate kernel produces deterministic,
well-behaved agreement under N-scale novelty-bearing disagreement.
That's the missing piece between "we have a substrate that
catches bugs" (Tier 0/1) and "decentralized training+inference
runs on this substrate" (the deployment story).
