# Architecture

## Overview

The World Model is organized in layers, from atomic data to emergent dynamics.

```
┌─────────────────────────────────────────────────────────────┐
│                    DYNAMICS LAYER                           │
│   Arena orchestrates adversarial competition                │
│   Trainer manages epochs, convergence, validation           │
├─────────────────────────────────────────────────────────────┤
│                    AGENTS LAYER                             │
│   7 tendencies compete: propose, stake, win/lose            │
├─────────────────────────────────────────────────────────────┤
│                    TREES LAYER                              │
│   Value hierarchies with weight propagation                 │
│   net_score = direct + sum(pro) - sum(con)                  │
├─────────────────────────────────────────────────────────────┤
│                    OBSERVATIONS LAYER                       │
│   Atomic facts (~280 bytes), no inherent polarity           │
└─────────────────────────────────────────────────────────────┘
```

## Layer 1: Observations

Atomic units of information. Sentence-sized, capped. No inherent polarity.

```python
@dataclass
class Observation:
    id: str
    content: str          # ~280 bytes max
    source_id: str        # Which document
    timestamp: datetime
    metadata: dict
```

Examples:
- "He lives paycheck to paycheck at age 42"
- "Delivered Etherlink solo in 3 months"
- "Uses ayahuasca for psychological calibration"

## Layer 2: Agents (Human Tendencies)

Seven generic drives that exist in every human:

| Tendency | Optimizes For |
|----------|---------------|
| SURVIVAL | Physical safety, resources, risk mitigation |
| STATUS | Social standing, achievement, recognition |
| MEANING | Significance, impact, legacy, purpose |
| CONNECTION | Relationships, belonging, community |
| AUTONOMY | Independence, self-determination, freedom |
| COMFORT | Ease, pleasure, avoiding pain |
| CURIOSITY | Knowledge, understanding, exploration |

Each agent has an **allocation** (0.0-1.0) representing influence. Allocations sum to 1.0.

```python
@dataclass
class Agent:
    tendency: Tendency
    allocation: float     # Starts at human average
    description: str
```

Default human-average allocations:
```python
DEFAULT_ALLOCATIONS = {
    Tendency.SURVIVAL: 0.18,
    Tendency.STATUS: 0.12,
    Tendency.MEANING: 0.10,
    Tendency.CONNECTION: 0.20,
    Tendency.AUTONOMY: 0.12,
    Tendency.COMFORT: 0.18,
    Tendency.CURIOSITY: 0.10,
}
```

## Layer 3: Trees (Value Hierarchies)

Binary tree structure where observations are positioned PRO or CON relative to a root claim.

```python
@dataclass
class Tree:
    id: str
    root_value: str       # The claim ("Financial security matters")
    root_node: Node

@dataclass
class Node:
    observation_id: str
    content: str
    position: Position    # ROOT, PRO, or CON
    stakes: dict[str, float]  # tendency -> weight
    pro_children: list[Node]
    con_children: list[Node]
```

### Weight Propagation

The core formula from the debate model:

```
net_score = direct_weight + sum(pro_children.score) - sum(con_children.score)
```

A node's strength isn't just its own stakes - it's adjusted by how well its sub-arguments hold up.

## Layer 4: Arena (Adversarial Dynamics)

Where "life" happens. The Arena orchestrates three phases:

### Phase 1: Proposal

Each agent proposes a **claim** (tree root) based on their tendency:

```
SURVIVAL: "Financial security is foundational to wellbeing"
MEANING: "Building infrastructure for posthumous continuity is the most significant work"
AUTONOMY: "True freedom comes from systems that can't be controlled by power structures"
```

### Phase 2: Adversarial Staking

Agents stake observations on ALL claims:

- **Support own claims**: Stake PRO on your tree
- **Undermine competitors**: Stake CON on their trees

The same observation gets staked multiple times with different positions:

```
"Lives paycheck to paycheck at 42"
  -> PRO on SURVIVAL's claim (evidence of financial risk)
  -> PRO on MEANING's claim (sacrifice for purpose)
  -> CON on COMFORT's claim (unsustainable)
```

### Phase 3: Resolution

1. Compute final scores for each claim (weight propagation)
2. Determine winner (highest score)
3. Reallocate influence based on scores

Winners gain allocation. Losers lose it. The equilibrium shifts.

```python
def _reallocate(self, agents, scores, learning_rate):
    # Normalize scores to target allocations
    # Blend current toward target
    # The person "learns" - becomes more oriented toward winning tendencies
```

## Layer 5: Trainer (ML-Style Training)

Manages multiple epochs with ML training patterns:

```python
@dataclass
class TrainConfig:
    max_epochs: int = 10
    min_epochs: int = 2
    convergence_threshold: float = 0.005
    patience: int = 3
    initial_lr: float = 0.15
    lr_decay: float = 0.9
    validation_split: float = 0.2
    min_allocation: float = 0.03
    max_allocation: float = 0.50
```

### Training Loop

1. Split observations into train/validation sets
2. For each epoch:
   - Run full debate on training observations
   - Check for convergence (allocation delta < threshold)
   - Decay learning rate
   - Run validation on held-out observations
3. Return history with metrics

### Validation

Statistical significance testing:

```python
class Validator:
    def validate(self, test_obs, claims, agents) -> ValidationResult:
        # For each observation, predict which claim/tendency owns it
        # Compare to random baseline (1/7 = 14.3%)
        # Compute p-value via binomial test
```

## Data Flow

```
Observations (JSON)
       │
       ▼
┌──────────────────┐
│ ObservationStore │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐
│     Arena        │────▶│    AgentSet      │
│                  │     │  (7 tendencies)  │
│  1. Proposal     │     └──────────────────┘
│  2. Staking      │
│  3. Resolution   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    TreeStore     │──── Claims with evidence trees
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  DebateResult    │──── Winner, scores, allocation changes
└──────────────────┘
```

## Storage

### JSON (Development)

```python
model = WorldModel(name="Person")
model.save("person.json")
model = WorldModel.load("person.json")
```

### Firestore (Production)

```python
adapter = FirestoreAdapter(db)
await adapter.save_world_model(model)
model = await adapter.load_world_model("person_id")
```

## API Layer

FastAPI service exposing:

- `GET /profile/{name}` - Load a world model
- `POST /profile/{name}/observations` - Add observations
- `POST /profile/{name}/debate` - Run adversarial debate
- `GET /profile/{name}/allocations` - Current agent allocations
