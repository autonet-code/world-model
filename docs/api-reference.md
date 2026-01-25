# API Reference

## Module Overview

```
world_model/
├── models/           # Core data structures
├── extraction/       # Observation extraction from text
├── staking/          # Legacy staking mechanisms
├── dynamics/         # Adversarial competition
└── storage/          # Persistence
```

---

## world_model.models

### Observation

```python
from world_model import Observation, ObservationStore

# Create observation
obs = Observation(
    content="Lives paycheck to paycheck at 42",
    source_id="conversation_001",
    timestamp=datetime.now(),
    metadata={"category": "financial"}
)

# Store
store = ObservationStore()
store.add(obs)

# Query
all_obs = store.all()
recent = store.recent(days=30)
by_source = store.by_source("conversation_001")
```

### Agent

```python
from world_model import Agent, AgentSet, Tendency, DEFAULT_ALLOCATIONS

# Single agent
agent = Agent(
    tendency=Tendency.MEANING,
    allocation=0.35,
    description="Optimizes for significance and legacy"
)

# Agent set with defaults
agents = AgentSet.with_defaults()

# Get specific agent
meaning_agent = agents.get(Tendency.MEANING)
print(meaning_agent.allocation)  # 0.10 (human average)

# Modify allocation
agents.get(Tendency.MEANING).allocation = 0.35
agents.normalize()  # Ensure sum = 1.0
```

**Tendency Enum**:
```python
class Tendency(Enum):
    SURVIVAL = "survival"
    STATUS = "status"
    MEANING = "meaning"
    CONNECTION = "connection"
    AUTONOMY = "autonomy"
    COMFORT = "comfort"
    CURIOSITY = "curiosity"
```

### Tree

```python
from world_model import Tree, TreeStore, Node, Position

# Create tree with root claim
tree = Tree(
    root_value="Financial security is foundational",
    description="SURVIVAL's claim"
)

# Add nodes
node = Node(
    observation_id="obs_001",
    content="Lives paycheck to paycheck",
    tree_id=tree.id,
)
node.add_stake("survival", 0.8)
tree.add_node(tree.root_node.id, node, Position.PRO)

# Query tree
all_nodes = tree.all_nodes()
depth = tree.depth()
score = tree.score  # Weight propagation

# Store multiple trees
trees = TreeStore()
trees.add(tree)
```

**Position Enum**:
```python
class Position(Enum):
    ROOT = "root"
    PRO = "pro"
    CON = "con"
```

---

## world_model.dynamics

### Arena

```python
from world_model import Arena

arena = Arena(work_dir="/tmp/arena")

# Full debate
trees, result = arena.run_full_debate(
    observations=obs_store,
    agents=agent_set,
    rounds=1,
    learning_rate=0.15,
    verbose=True,
)

# Or step by step
claims = arena.proposal_phase(observations, agents)
arena.staking_phase(observations, claims, agents)
result = arena.resolution_phase(claims, agents)
```

**DebateResult**:
```python
@dataclass
class DebateResult:
    claims: list[Claim]           # All proposed claims
    total_stakes: int             # Total staking decisions
    winner: Optional[Tendency]    # Highest-scoring tendency
    scores: dict[Tendency, float] # Score per tendency
    allocation_changes: dict[Tendency, float]  # Delta per tendency
```

### Trainer

```python
from world_model.dynamics import Trainer, TrainConfig

config = TrainConfig(
    max_epochs=5,
    convergence_threshold=0.01,
    validation_split=0.2,
)

trainer = Trainer(config)
history, result = trainer.train(
    observations=obs_store,
    agents=agent_set,
    logger=None,  # Optional logger
    verbose=True,
)
```

**TrainHistory**:
```python
@dataclass
class TrainHistory:
    epochs_run: int
    converged: bool
    best_epoch: int
    metrics: list[EpochMetrics]
    validation_results: list[ValidationResult]
```

### Loggers

```python
from world_model.dynamics import (
    ConsoleLogger,
    JSONLogger,
    TensorBoardLogger,
    WandbLogger,
)

# Console output
logger = ConsoleLogger()

# JSON file
logger = JSONLogger("training.json")

# TensorBoard
logger = TensorBoardLogger("runs/exp1")

# Weights & Biases
logger = WandbLogger(project="world-model", run_name="v1")

# Use in training
trainer.train(..., logger=logger)
```

---

## world_model.extraction

### ObservationExtractor

```python
from world_model import ObservationExtractor

extractor = ObservationExtractor()

# From markdown
observations = extractor.extract(
    content=markdown_text,
    source_id="doc_001"
)

# From file
observations = extractor.extract_from_file("summary.md")
```

---

## world_model.storage

### WorldModel

```python
from world_model import WorldModel, create_world_model

# Create from file
model = create_world_model("Person", "observations.json")

# Access components
print(model.name)
print(len(model.observations))
print(model.agents)
print(model.trees)

# Save/load JSON
model.save("person.json")
model = WorldModel.load("person.json")
```

### FirestoreAdapter

```python
from world_model.storage import FirestoreAdapter
from google.cloud import firestore

db = firestore.AsyncClient()
adapter = FirestoreAdapter(db)

# Save
await adapter.save_world_model(model)

# Load
model = await adapter.load_world_model("person_id")

# Update specific parts
await adapter.update_observations(model.name, new_observations)
await adapter.update_agents(model.name, model.agents)
```

---

## api (FastAPI)

### Endpoints

```python
# Start server
uvicorn api.main:app --reload
```

**GET /profile/{name}**
```bash
curl http://localhost:8000/profile/andrei
```

**POST /profile/{name}/observations**
```bash
curl -X POST http://localhost:8000/profile/andrei/observations \
  -H "Content-Type: application/json" \
  -d '{"content": "New observation here"}'
```

**POST /profile/{name}/debate**
```bash
curl -X POST http://localhost:8000/profile/andrei/debate \
  -H "Content-Type: application/json" \
  -d '{"rounds": 1, "learning_rate": 0.15}'
```

**GET /profile/{name}/allocations**
```bash
curl http://localhost:8000/profile/andrei/allocations
```

---

## Complete Example

```python
from world_model import (
    ObservationStore, Observation,
    AgentSet, Tendency,
    Arena, WorldModel,
)
from world_model.dynamics import Trainer, TrainConfig, JSONLogger

# 1. Create observations
store = ObservationStore()
store.add(Observation(
    content="Lives paycheck to paycheck at 42",
    source_id="bio"
))
store.add(Observation(
    content="Built governance framework solo",
    source_id="bio"
))

# 2. Create agents
agents = AgentSet.with_defaults()

# 3. Run single debate
arena = Arena()
trees, result = arena.run_full_debate(
    observations=store,
    agents=agents,
    rounds=1,
)
print(f"Winner: {result.winner}")

# 4. Run training with validation
config = TrainConfig(max_epochs=3, validation_split=0.2)
trainer = Trainer(config)
history, _ = trainer.train(store, agents, JSONLogger("log.json"))

print(f"Accuracy: {history.validation_results[-1].accuracy:.1%}")

# 5. Save
model = WorldModel(name="person", observations=store, agents=agents, trees=trees)
model.save("person_trained.json")
```
