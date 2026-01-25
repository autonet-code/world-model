# Training Guide

The World Model uses ML-style training patterns to calibrate agent allocations. This guide covers the training loop, configuration, validation, and visualization.

## Overview

Training runs multiple **epochs** of adversarial debate. Each epoch:
1. Agents propose claims
2. Agents stake observations adversarially
3. Resolution determines winners
4. Allocations shift toward winners

Training continues until **convergence** (allocations stabilize) or max epochs reached.

## Quick Start

```python
from world_model import create_world_model
from world_model.dynamics import Trainer, TrainConfig

# Load observations
model = create_world_model("Person", "observations.json")

# Configure training
config = TrainConfig(
    max_epochs=5,
    convergence_threshold=0.01,
    validation_split=0.2,
)

# Train
trainer = Trainer(config)
history, result = trainer.train(
    observations=model.observations,
    agents=model.agents,
)

# Check results
print(f"Epochs: {history.epochs_run}")
print(f"Converged: {history.converged}")
print(f"Accuracy: {history.validation_results[-1].accuracy:.1%}")
```

## Configuration

```python
@dataclass
class TrainConfig:
    # Epochs
    max_epochs: int = 10          # Maximum training rounds
    min_epochs: int = 2           # Minimum before early stopping

    # Convergence
    convergence_threshold: float = 0.005  # Stop if delta < this
    patience: int = 3             # Epochs to wait before stopping

    # Learning rate
    initial_lr: float = 0.15      # Starting learning rate
    lr_decay: float = 0.9         # Multiply by this each epoch

    # Validation
    validation_split: float = 0.2 # Hold out for testing

    # Allocation bounds
    min_allocation: float = 0.03  # Floor (no tendency below 3%)
    max_allocation: float = 0.50  # Ceiling (no tendency above 50%)
```

### Parameter Guide

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `max_epochs` | More = finer calibration, more API calls | 3-10 |
| `convergence_threshold` | Lower = stricter convergence | 0.005-0.02 |
| `initial_lr` | Higher = faster but noisier learning | 0.1-0.2 |
| `lr_decay` | Lower = more aggressive decay | 0.8-0.95 |
| `validation_split` | Higher = better validation, less training data | 0.15-0.3 |

## Training Phases

### Phase 1: Setup

```python
# Split observations
train_store, val_store = trainer._split_observations(observations, config.validation_split)
# 80% for training, 20% for validation
```

### Phase 2: Epoch Loop

Each epoch runs a full adversarial debate:

```python
for epoch in range(max_epochs):
    # Run debate
    trees, result = arena.run_full_debate(
        observations=train_store,
        agents=agents,
        learning_rate=current_lr,
    )

    # Check convergence
    delta = sum(abs(change) for change in result.allocation_changes.values())
    if delta < convergence_threshold:
        converged = True

    # Decay learning rate
    current_lr *= lr_decay

    # Run validation
    val_result = validator.validate(val_store, result.claims, agents)
```

### Phase 3: Validation

Held-out observations test prediction accuracy:

```python
class Validator:
    def validate(self, test_obs, claims, agents):
        correct = 0
        for obs in test_obs:
            # Ask Claude which claim/tendency best "owns" this observation
            predicted = self._predict_owner(obs, claims)
            # Compare to highest-scoring claim
            actual = max(claims, key=lambda c: relevance_score(obs, c))
            if predicted == actual:
                correct += 1

        accuracy = correct / len(test_obs)
        p_value = binomial_test(correct, len(test_obs), 1/7)
        return ValidationResult(accuracy, p_value, ...)
```

## Training History

```python
@dataclass
class TrainHistory:
    epochs_run: int
    converged: bool
    best_epoch: int
    metrics: list[EpochMetrics]
    validation_results: list[ValidationResult]

@dataclass
class EpochMetrics:
    epoch: int
    learning_rate: float
    winner: Tendency
    winning_score: float
    scores: dict[Tendency, float]
    allocations: dict[Tendency, float]
    allocation_changes: dict[Tendency, float]
    total_stakes: int
```

## Logging

Four logging backends available:

### Console Logger

```python
logger = ConsoleLogger()
trainer.train(..., logger=logger)
```

Output:
```
EPOCH 1 (lr=0.150)
  Winner: SURVIVAL (score: 4.027)
  Allocations: survival=18.7%, meaning=10.5%, ...
  Validation: 27.3% accuracy (p=0.001)**
```

### JSON Logger

```python
logger = JSONLogger("training_log.json")
trainer.train(..., logger=logger)
```

Creates structured JSON for visualization:
```json
{
  "config": {...},
  "epochs": [
    {
      "epoch": 1,
      "winner": "survival",
      "scores": {...},
      "allocations": {...}
    }
  ]
}
```

### TensorBoard Logger

```python
logger = TensorBoardLogger("runs/experiment1")
trainer.train(..., logger=logger)
```

Then visualize:
```bash
tensorboard --logdir runs/
```

### Weights & Biases Logger

```python
logger = WandbLogger(project="world-model", run_name="andrei-v1")
trainer.train(..., logger=logger)
```

## Interpreting Results

### Convergence

```python
if history.converged:
    print("Allocations stabilized - equilibrium found")
else:
    print("Max epochs reached - may need more training")
```

### Validation Accuracy

| Accuracy | Interpretation |
|----------|----------------|
| ~14% (1/7) | Random chance - model not learning |
| 20-30% | Significant - model captures some structure |
| 30-50% | Strong - model predicts tendency well |
| >50% | Very strong - clear tendency patterns |

### P-Value

- **p < 0.05**: Statistically significant
- **p < 0.01**: Highly significant
- **p < 0.001**: Very highly significant

### Allocation Trajectories

```python
for m in history.metrics:
    print(f"Epoch {m.epoch}:")
    for t, alloc in sorted(m.allocations.items(), key=lambda x: -x[1]):
        print(f"  {t.value}: {alloc:.1%}")
```

Watch for:
- **Convergence**: Allocations stabilizing
- **Dominance**: One tendency taking over (may indicate bias)
- **Balance**: Multiple tendencies maintaining influence

## Example: Full Training Run

```python
from world_model import create_world_model
from world_model.dynamics import (
    Trainer, TrainConfig,
    JSONLogger, ConsoleLogger,
)

# Load
model = create_world_model("Andrei", "observations.json")

# Configure
config = TrainConfig(
    max_epochs=5,
    min_epochs=2,
    convergence_threshold=0.01,
    initial_lr=0.15,
    lr_decay=0.85,
    validation_split=0.2,
)

# Set up logging
json_logger = JSONLogger("training_log.json")

# Train
trainer = Trainer(config)
history, result = trainer.train(
    observations=model.observations,
    agents=model.agents,
    logger=json_logger,
    verbose=True,
)

# Results
print(f"\nTraining complete!")
print(f"Epochs: {history.epochs_run}")
print(f"Converged: {history.converged}")
print(f"Winner: {result.winner.value if result.winner else 'None'}")

print("\nFinal allocations:")
for agent in sorted(model.agents.all(), key=lambda a: -a.allocation):
    print(f"  {agent.tendency.value}: {agent.allocation:.1%}")

if history.validation_results:
    val = history.validation_results[-1]
    print(f"\nValidation: {val.accuracy:.1%} accuracy (p={val.p_value:.4f})")

# Save
model.save("andrei_trained.json")
```

## Troubleshooting

### Timeout Errors

Claude CLI calls may timeout on large batches:

```python
# In arena.py, increase timeout
result = subprocess.run(cmd, timeout=300)  # 5 minutes
```

Or reduce batch size:
```python
batch_size = 10  # Instead of 20
```

### No Convergence

If allocations keep oscillating:
- Lower `initial_lr` (e.g., 0.1)
- Increase `patience`
- Lower `convergence_threshold`

### One Tendency Dominates

If one tendency reaches max allocation:
- Check observation balance (are observations biased?)
- Increase `min_allocation` floor
- Lower `max_allocation` ceiling

### Low Validation Accuracy

If accuracy is near random:
- More training epochs
- More observations
- Check observation quality (are they meaningful?)
