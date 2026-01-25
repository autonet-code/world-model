# Development Guide

## Setup

### Prerequisites

- Python 3.10+
- Claude CLI installed and configured
- (Optional) TensorBoard for visualization
- (Optional) Firestore for production storage

### Installation

```bash
cd life
pip install -r requirements.txt  # If exists
```

### Running Tests

```bash
# Single debate test
python test_arena.py

# Full training test
python test_training.py

# Hierarchical staking test (legacy)
python test_hierarchical.py
```

## Project Structure

```
life/
├── world_model/
│   ├── __init__.py           # Main exports
│   ├── models/
│   │   ├── __init__.py
│   │   ├── observation.py    # Observation, ObservationStore
│   │   ├── agent.py          # Agent, AgentSet, Tendency
│   │   ├── tree.py           # Tree, Node, Position, Stake
│   │   ├── deviation.py      # Legacy deviation model
│   │   └── evidence.py       # Legacy evidence model
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── extractor.py      # Legacy extractor
│   │   └── observation_extractor.py  # Claude-based extraction
│   ├── staking/
│   │   ├── __init__.py
│   │   ├── staker.py         # Legacy single staker
│   │   └── hierarchical_staker.py  # Legacy hierarchical
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── arena.py          # Adversarial debate orchestration
│   │   └── trainer.py        # ML-style training loop
│   └── storage/
│       ├── __init__.py
│       ├── graph.py          # Legacy graph storage
│       ├── world_model_store.py  # JSON persistence
│       └── firestore_adapter.py  # Firestore persistence
├── api/
│   ├── __init__.py
│   └── main.py               # FastAPI service
├── docs/                     # Documentation
├── observations.json         # Sample data
└── test_*.py                 # Test scripts
```

## Key Components

### Adding a New Tendency

1. Add to `Tendency` enum in `models/agent.py`:
```python
class Tendency(Enum):
    ...
    NEW_TENDENCY = "new_tendency"
```

2. Add default allocation in `DEFAULT_ALLOCATIONS`:
```python
DEFAULT_ALLOCATIONS = {
    ...
    Tendency.NEW_TENDENCY: 0.10,
}
```

3. Update prompt in `dynamics/arena.py` proposal phase

### Modifying Weight Propagation

Core formula in `models/tree.py`:

```python
@property
def score(self) -> float:
    direct = sum(self.stakes.values())
    pro_sum = sum(c.score for c in self.pro_children)
    con_sum = sum(c.score for c in self.con_children)
    return direct + pro_sum - con_sum
```

Modify for different propagation strategies (e.g., decay, normalization).

### Adding a New Logger

1. Create class implementing the logger interface:
```python
class MyLogger:
    def log_epoch(self, metrics: EpochMetrics):
        # Log epoch data

    def log_validation(self, result: ValidationResult):
        # Log validation data

    def log_config(self, config: TrainConfig):
        # Log configuration

    def finish(self):
        # Cleanup
```

2. Use in training:
```python
trainer.train(..., logger=MyLogger())
```

### Adding a Storage Backend

1. Implement the adapter pattern:
```python
class MyStorageAdapter:
    async def save_world_model(self, model: WorldModel):
        ...

    async def load_world_model(self, model_id: str) -> WorldModel:
        ...

    async def update_observations(self, model_id: str, obs: ObservationStore):
        ...

    async def update_agents(self, model_id: str, agents: AgentSet):
        ...
```

2. See `storage/firestore_adapter.py` for reference.

## Claude CLI Integration

The Arena uses Claude CLI for semantic analysis:

```python
def _call_claude(self, prompt: str, timeout: int = 300) -> str:
    # Write prompt to temp file
    prompt_file = self.work_dir / f"prompt_{hash(prompt)}.txt"
    prompt_file.write_text(prompt)

    # Call Claude
    if os.name == 'nt':  # Windows
        cmd = f'type "{prompt_file}" | claude -p --dangerously-skip-permissions'
        result = subprocess.run(cmd, shell=True, ...)
    else:  # Unix
        with open(prompt_file) as f:
            result = subprocess.run(
                ['claude', '-p', '--dangerously-skip-permissions'],
                stdin=f, ...
            )

    return result.stdout
```

### Prompt Engineering

Prompts are in `dynamics/arena.py`. Key patterns:

1. **Clear role**: "You are simulating adversarial staking..."
2. **Structured context**: Claims, observations formatted clearly
3. **JSON output**: Request specific JSON schema
4. **Examples**: Show expected format

### Handling Claude Failures

```python
try:
    response = self._call_claude(prompt, timeout=120)
    data = self._parse_json(response)
except subprocess.TimeoutExpired:
    # Retry or skip batch
except json.JSONDecodeError:
    # Retry or use fallback
```

## Testing

### Unit Tests

```python
def test_weight_propagation():
    tree = Tree(root_value="Test")
    node = Node(content="Evidence", tree_id=tree.id)
    node.add_stake("meaning", 0.5)
    tree.add_node(tree.root_node.id, node, Position.PRO)

    assert tree.score == 0.5
```

### Integration Tests

```python
def test_full_debate():
    store = ObservationStore()
    # Add observations...
    agents = AgentSet.with_defaults()

    arena = Arena()
    trees, result = arena.run_full_debate(store, agents)

    assert result.winner is not None
    assert sum(result.scores.values()) > 0
```

### Validation Tests

```python
def test_validation_accuracy():
    # Train on subset
    # Validate on held-out
    # Assert accuracy > baseline
```

## Performance Considerations

### Claude API Calls

Each epoch makes multiple Claude calls:
- 1 for proposal phase
- N for staking phase (batch_size=20 -> ceil(obs/20) calls)
- 1 for validation (if enabled)

Optimize by:
- Increasing batch size (more tokens per call)
- Reducing epochs
- Caching proposals across epochs

### Memory

Large observation stores can consume memory:
- ObservationStore holds all observations in memory
- TreeStore holds all trees with nodes

Consider streaming for very large datasets.

### Parallelization

Currently sequential. Potential parallelization:
- Staking batches could run in parallel
- Multiple epochs could checkpoint and resume

## Common Issues

### "charmap codec can't encode character"

Windows encoding issue. Solution:
```python
prompt_file.write_text(prompt, encoding="utf-8")
```

### "'X' is not a valid Tendency"

Parsing error from Claude response. Check:
- JSON format in response
- Exact field names match expectations
- Add fallback handling

### Timeout Errors

Increase timeout or reduce batch size:
```python
result = subprocess.run(cmd, timeout=300)  # 5 minutes
```

### Allocations Don't Converge

Lower learning rate or increase patience:
```python
config = TrainConfig(
    initial_lr=0.1,
    patience=5,
    convergence_threshold=0.02,
)
```

## Future Extensions

### Temporal Dynamics

Track allocation changes over time:
```python
@dataclass
class AllocationSnapshot:
    timestamp: datetime
    allocations: dict[Tendency, float]
    trigger: str  # "debate", "new_observations", etc.
```

### Context Modes

Different allocations for different contexts:
```python
class ContextualAgentSet:
    contexts: dict[str, AgentSet]  # "work", "family", "creative"

    def get_for_context(self, context: str) -> AgentSet:
        return self.contexts.get(context, self.default)
```

### Embedding Integration

Use embeddings for faster relevance scoring:
```python
def relevance_score(obs_embedding, tree_embedding) -> float:
    return cosine_similarity(obs_embedding, tree_embedding)
```

### Multi-Person Models

Compare world models across people:
```python
def similarity(model_a: WorldModel, model_b: WorldModel) -> float:
    # Compare allocation distributions
    # Compare tree structures
    # Compute divergence score
```
