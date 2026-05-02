# World Model

**A graph-equilibration AI substrate. No backprop, no gradients, no LLMs.**

A coalition of competing theses (tendencies) that stake on a fractal tree of claims. Observations arrive, equilibrium settles, mint is awarded for movement that survives. The same architecture handles boolean reasoning, alignment scoring, federated training, and decentralized inference — without a single tensor or gradient step.

---

## What it does today

Three deployed capabilities, each with a runnable demo:

### Boolean reasoning by single-pass equilibration

```bash
python demo_constraint_satisfaction.py    # 10/10 hand-built SAT cases
python demo_satlib.py 20                   # SATLIB uf20-91: 94.1% clause satisfaction
```

- 10/10 SAT cases pass via single-pass equilibration: AND, NOT, OR, IMPLICATION, two-hop and three-hop chained reasoning, UNSAT detection, mixed 4-variable constraints.
- SATLIB uf20-91 (20-variable random 3-SAT at the phase transition): **94.1% clause satisfaction**. Random baseline is 87.5%. Best instance: 97.8%. No backtracking — just one settling round per instance.

### Smooth promotion: every node earns standing

```bash
python demo_promotion_earns_keep.py       # 3.67-point asymmetric effect
```

There is no qualitative line between "claim" and "tendency." Every node has an outbound staking capacity proportional to the PRO stake it has accumulated. A freshly sprouted sub-claim starts silent; as it earns standing, its voice grows. The A/B test shows a 3.67-point shift in equilibrium delta attributable to smooth promotion alone, in scenarios where root staking can't reach.

### Pruning at epoch close

```bash
python demo_pruning.py                    # entropy reduction
```

Branches whose score never moved meaningfully **and** which never accumulated standing get discarded. Roots are never pruned. Idempotent and deterministic — same inputs, same pruned set across all verifiers.

---

## Architecture (generalized model)

Located in `world_model/generalized/`. This is what's used in production.

| Type | What it does |
|---|---|
| `GeneralizedTendency` | A thesis about the world. Owns a tree of sub-claims, a coordinate-space frame, a budget. |
| `World` | A coalition of tendencies + an observation stream + the cross-stake graph. |
| `CoordinateClaim` | A claim anchored at a point in coordinate space, with a polarity axis. |
| `CoordinateFrame` | Non-LLM ReferenceFrame. Sparse-friendly dimensional-overlap similarity; sign-agreement stance. |
| `CoordinateProbe` | NoveltyProbe with the formalized terminations (INTEGRATED / CONTRADICTS / DISRUPTS / ORTHOGONAL). |
| `equilibrate(world)` | Rounds of (act, apply_stakes) until stake intents stabilize. |
| `propose_growth(world)` | Sprout PRO/CON children under contended nodes. |
| `prune_settled_negatives(world, history)` | Discard low-score, low-novelty subtrees. |
| `ScoreHistory` | Append-only ring of per-node score snapshots; feeds pruning and reconciliation. |

The **content-addressed sub-claim ids** mean two solvers proposing the same claim under the same parent at the same coords produce the same node id. Federation converges naturally — no consensus protocol, no central log, no merge resolution.

---

## How novelty and mint relate

Novelty and mint are deliberately separated:

- **Novelty** is descriptive: the magnitude of score movement at a node during an epoch. Captures surprise regardless of direction or correctness.
- **Mint** is rewarded: only positive movement that ends with positive score qualifies. The agent who caused that upward survival mints; CON-contributors don't, even when correct.

This closes the obvious collusion attack: two agents can't extract mint by manufacturing a false PRO claim and "debunking" each other's claims, because the debunking leg never lands in mint territory.

The full reconciliation pipeline lives in autonet's substrate adapter (`c:\code\autonet\nodes\common\world_model_substrate\reconcile.py`), which uses this engine.

---

## Autonet integration

This engine is the substrate for [autonet](https://github.com/autonet-code/autonet)'s decentralized AI network, replacing the VL-JEPA training architecture that hit a capacity ceiling on real-world data.

Vertical slice (solver → aggregator → verifier → inference) runs end-to-end without any smart-contract changes. Per-agent mint distribution is computed at epoch boundaries; the substrate produces protocol-compatible event streams that the existing FedAvg-shaped aggregator path replaces with `aggregate_contributions`.

```bash
# In autonet:
pip install -e c:/code/world-model
python test_world_model_substrate_e2e.py        # vertical slice
python test_epoch_reconciliation.py             # mint distribution
python test_multi_solver_convergence.py         # federation
```

---

## Quick start (engine only)

```python
from world_model.generalized import (
    GeneralizedTendency, World, Observation, equilibrate,
)

# Two opposing tendencies on a 1D axis
A = GeneralizedTendency(
    id="left", thesis="left wins",
    anchor=(-1.0,), polarity_axis=(-1.0,),
    bandwidth=2.0,
)
B = GeneralizedTendency(
    id="right", thesis="right wins",
    anchor=(+1.0,), polarity_axis=(+1.0,),
    bandwidth=2.0,
)
world = World()
world.add_tendency(A)
world.add_tendency(B)

# Feed evidence on the left side
for x in [-1.2, -1.0, -0.8]:
    world.add_observation(Observation(coords=(x,)))
equilibrate(world)

print(world.root_scores())   # left tendency wins
```

For boolean reasoning, see `demo_constraint_satisfaction.py`. For multi-tendency federation, see autonet's substrate slice.

---

## Open research

Two engine-level threads, deferred until empirical data from real long-running streams shapes the design:

1. **Novelty decay on settled regions.** Once a node's score has been stable for N epochs, decay its potential-novelty so the engine concentrates probe budget on growth frontiers. Required for the depth-driven attention property to scale.
2. **Locality rule for sub-claim staking.** Sub-claims at depth d should stake mostly at depth d-1 to d+1, not back at the roots. The operational graph becomes the active frontier rather than the whole tree.

Both are necessary for the engine to scale beyond shallow worlds without re-evaluating settled structure each round. They're tracked but waiting on real-world signal about what "settled" looks like in operation.

---

## Origin: the personality model

This module started as a personality-modeling system: seven human drives (SURVIVAL, MEANING, AUTONOMY, COMFORT, CONNECTION, CURIOSITY, STATUS) competing in adversarial debates over observations about a person. That work is still in `world_model/models/`, `world_model/dynamics/`, `world_model/staking/`, etc., and the legacy README sections describing it are at `docs/legacy/`.

The generalized model in `world_model/generalized/` was built when it became clear that the same architecture handles any system — physics, alignment, reasoning — not just personality. The personality case is now one application of a general substrate. The seven tendencies become four (life, self-preservation, intelligence, evolution) when the substrate is used for charter alignment in autonet; arbitrary tendencies for other domains.

The formalization in `docs/FORMALIZATION.md` and `docs/THEORY.md` was domain-agnostic from the start; the LLM-backed concrete probes in `world_model/novelty/` (Wikidata, NLI, Hybrid) are still there. The new `CoordinateFrame` and `CoordinateProbe` in `world_model/generalized/coordinate_frame.py` are the no-LLM concrete implementations of those same abstractions.

---

## Status

- 14 / 14 substrate-integration tasks complete (see [autonet's task list](https://github.com/autonet-code/autonet)).
- 10 / 10 hand-built SAT cases pass.
- SATLIB 94.1% on 20-instance run, 6.6 percentage points above random.
- Multi-solver content-addressed convergence verified.
- Substrate vertical slice through autonet's solver / aggregator / verifier / inference paths runs end-to-end.

---

## Installation

```bash
pip install -e .
```

Python 3.11+. The generalized model has zero external dependencies. The personality-modeling and LLM-backed novelty modules require optional extras:

```bash
pip install -e ".[novelty]"        # LLM-backed novelty probes
pip install -e ".[api]"            # FastAPI service for personality model
pip install -e ".[all]"
```

---

## Documentation

- [`world_model/generalized/STATUS.md`](world_model/generalized/STATUS.md) — substrate state, what works, open threads.
- [`docs/FORMALIZATION.md`](docs/FORMALIZATION.md) — mathematical specification of novelty.
- [`docs/THEORY.md`](docs/THEORY.md) — theoretical foundation.
- [`docs/architecture.md`](docs/architecture.md) — original layered description (personality-era).
- [`docs/concepts.md`](docs/concepts.md) — terminology.

---

## License

MIT
