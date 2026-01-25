# Handover Notes

## What We Built

A system to model a person's worldview as an **adversarial equilibrium** of competing internal tendencies.

**Key insight**: Ideas are atomic, position is relational. The same observation means different things under different value lenses.

## Documentation

See `/docs` for comprehensive documentation:

- [docs/README.md](./docs/README.md) - Overview and quick start
- [docs/architecture.md](./docs/architecture.md) - System design
- [docs/concepts.md](./docs/concepts.md) - Core concepts
- [docs/training.md](./docs/training.md) - ML-style training guide
- [docs/api-reference.md](./docs/api-reference.md) - Module documentation
- [docs/development.md](./docs/development.md) - Contributing guide

## Quick Start

```python
from world_model import create_world_model, Arena

# Load observations
model = create_world_model("Andrei", "observations.json")

# Run adversarial debate
arena = Arena()
trees, result = arena.run_full_debate(
    observations=model.observations,
    agents=model.agents,
)

print(f"Winner: {result.winner}")  # Which tendency's claim won
```

## Current State

**Implemented:**
- 7 human tendencies as competing agents
- Adversarial debate: propose -> stake -> resolve
- Weight propagation: `net_score = direct + sum(pro) - sum(con)`
- ML-style training with convergence detection
- Validation with statistical significance testing
- JSON and Firestore persistence
- FastAPI service

**Validation Results (first run):**
- Accuracy: 27.3% (baseline: 7.1%)
- P-value: 0.001 (highly significant)
- Model predicts tendency ownership 4x better than chance

## The Spark (Why This Matters)

This isn't just a data structure. It's a hypothesis about how meaning works.

Binary distinction (pro/con) is the primitive. Everything is this/not-this. Compose recursively -> arbitrary complexity.

**The motivating error**: First extraction said Andrei "distrusts DAOs." But he built Jurisdiction with fractal DAO topology. Wrong because no mechanism to *contest* the claim. Pro/con structure surfaces tension, forces refinement.

**Adversarial dynamics**: Agents don't just categorize - they COMPETE. Propose claims, stake evidence, win or lose. The competition IS the life of the system.

**Agents as internal plurality**: A person isn't monolithic. They're a coalition of drives finding equilibrium. The structure of that equilibrium IS the personality.

Worth building because it might be *true*, not just useful.

## Key Files

```
life/
├── world_model/          # Core package
│   ├── models/           # Observation, Agent, Tree, Node
│   ├── dynamics/         # Arena (debates), Trainer (ML loop)
│   └── storage/          # JSON, Firestore persistence
├── api/                  # FastAPI service
├── docs/                 # Documentation
├── observations.json     # 165 observations about Andrei
├── test_arena.py         # Single debate test
└── test_training.py      # Full training test
```

## The Debate Model Foundation

Extends tokenized debates from `c:/code/werule_new`:
- Pro/con binary tree structure
- Weight propagation formula
- Arguments stake voting power

We extend it: multiple simultaneous trees, same observations appearing across them, internal tendencies as agents instead of external voters.
