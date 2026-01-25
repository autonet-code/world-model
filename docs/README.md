# World Model Documentation

A system for modeling a person's worldview as an **adversarial equilibrium** of competing internal tendencies.

## Quick Links

- [Architecture](./architecture.md) - System design and components
- [Core Concepts](./concepts.md) - Key ideas and terminology
- [Training Guide](./training.md) - ML-style training with convergence detection
- [API Reference](./api-reference.md) - Module documentation
- [Development](./development.md) - Contributing and extending
- [Future: Digital Twin](./future-digital-twin.md) - Vision for complete individual representation

## What Is This?

The World Model represents a person not as a static profile, but as a **dynamic equilibrium** of competing internal drives. Seven human tendencies (survival, status, meaning, connection, autonomy, comfort, curiosity) act as **agents** that:

1. **PROPOSE** claims about what matters ("Financial security is foundational")
2. **STAKE** observations to support their claims and undermine competitors
3. **WIN or LOSE** debates based on evidence
4. **GAIN or LOSE** influence based on outcomes

The equilibrium that emerges **IS** the personality.

## Key Insight

> Ideas are atomic. Position is relational.

The same observation means different things depending on what you're optimizing for:

- "Lives paycheck to paycheck at 42"
  - **SURVIVAL**: PRO (evidence of financial risk)
  - **MEANING**: PRO (sacrifice for purpose)
  - **AUTONOMY**: PRO (chose freedom over salary)
  - **COMFORT**: PRO (evidence of unsustainable intensity)

## Quick Start

```python
from world_model import (
    ObservationStore, AgentSet, Arena,
    WorldModel, create_world_model,
)

# Load a person's observations
model = create_world_model("Person", "observations.json")

# Run adversarial debate
arena = Arena()
trees, result = arena.run_full_debate(
    observations=model.observations,
    agents=model.agents,
    rounds=1,
)

# See who won
print(f"Winner: {result.winner}")
print(f"Final allocations: {model.agents}")
```

## ML-Style Training

```python
from world_model.dynamics import Trainer, TrainConfig

config = TrainConfig(
    max_epochs=5,
    convergence_threshold=0.01,
    validation_split=0.2,
)

trainer = Trainer(config)
history, result = trainer.train(
    observations=model.observations,
    agents=model.agents,
)

# Check validation accuracy
print(f"Accuracy: {history.validation_results[-1].accuracy:.1%}")
print(f"P-value: {history.validation_results[-1].p_value:.4f}")
```

## Project Structure

```
life/
├── world_model/
│   ├── models/           # Core data structures
│   │   ├── observation.py    # Atomic facts
│   │   ├── agent.py          # Human tendencies
│   │   └── tree.py           # Value hierarchies with weight propagation
│   ├── extraction/       # Extract observations from text
│   ├── staking/          # Legacy staking mechanisms
│   ├── dynamics/         # Adversarial competition
│   │   ├── arena.py          # Debate orchestration
│   │   └── trainer.py        # ML-style training loop
│   └── storage/          # Persistence (JSON, Firestore)
├── api/                  # FastAPI service
├── docs/                 # Documentation
└── test_*.py             # Test scripts
```

## Validation Results

First training run on 165 observations:

| Metric | Value |
|--------|-------|
| Accuracy | 27.3% |
| Baseline | 7.1% (1/7 random) |
| P-value | 0.001 |
| Significant | Yes |

The model predicts which tendency "owns" an observation nearly **4x better than chance**.
