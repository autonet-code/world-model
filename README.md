# World Model

**Modeling personality as adversarial equilibrium**

A computational framework for representing a person's decision-making as competing internal drives that stake evidence and find equilibrium. Not a profile. Not a summary. A dynamic system that *is* the personality, in computational form.

---

## The Core Insight

> Ideas are atomic. Position is relational.

The observation "Lives paycheck to paycheck at 42" has no inherent meaning. Its meaning emerges from what you're optimizing for:

- **SURVIVAL** sees: financial risk, precarity
- **MEANING** sees: sacrifice for purpose, choosing mission over money
- **AUTONOMY** sees: rejecting salary slavery, freedom at cost
- **COMFORT** sees: unsustainable intensity, warning sign

Same fact. Different positions. The World Model captures this by letting the same observation appear across multiple value trees with different polarities.

---

## How It Works

Seven universal human tendencies act as **agents** competing for influence:

| Tendency | Question It Asks |
|----------|------------------|
| SURVIVAL | "Is this safe? Do I have enough?" |
| STATUS | "Am I respected? Do I matter to others?" |
| MEANING | "Does this matter? Will it outlast me?" |
| CONNECTION | "Am I known? Do I belong?" |
| AUTONOMY | "Am I free? Can I choose?" |
| COMFORT | "Is this pleasant? Can I sustain this?" |
| CURIOSITY | "Do I understand? What's there to learn?" |

Each agent:
1. **PROPOSES** claims about what matters
2. **STAKES** observations as evidence (PRO or CON)
3. **WINS or LOSES** debates based on evidence strength
4. **GAINS or LOSES** allocation based on outcomes

The equilibrium that emerges—which tendencies dominate, where they conflict, how they resolve—**is** the personality.

---

## Validation

First training run on 165 observations:

| Metric | Value |
|--------|-------|
| Accuracy | 27.3% |
| Baseline | 7.1% (random chance = 1/7) |
| P-value | 0.001 |

The model predicts which tendency "owns" an observation **4x better than chance**.

---

## Quick Start

```python
from world_model import create_world_model, Arena

# Load observations about a person
model = create_world_model("Person", "observations.json")

# Run adversarial debate
arena = Arena()
trees, result = arena.run_full_debate(
    observations=model.observations,
    agents=model.agents,
)

print(f"Winner: {result.winner}")
print(f"Allocations: {model.agents}")
```

### ML-Style Training

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

print(f"Validation accuracy: {history.validation_results[-1].accuracy:.1%}")
```

---

## Project Structure

```
world_model/
├── models/           # Core data structures
│   ├── observation.py    # Atomic facts (~280 bytes)
│   ├── agent.py          # Tendencies with allocations
│   └── tree.py           # Value hierarchies + weight propagation
├── dynamics/         # Adversarial competition
│   ├── arena.py          # Debate orchestration
│   └── trainer.py        # ML-style training loop
├── extraction/       # Extract observations from text
├── staking/          # Evidence staking mechanisms
└── storage/          # JSON and Firestore persistence

api/                  # FastAPI service
docs/                 # Full documentation
```

---

## The Bet

This architecture bets that:

1. **Binary distinction + recursion** = sufficient to model meaning
2. **Seven tendencies** = comprehensive but tractable agent set
3. **Competition** = produces coherent personality from plurality
4. **Same structure** = applies to anyone (swap observations + allocations)
5. **More useful** = than unstructured text at scale

If the bet pays off, this is a general architecture for modeling minds.

---

## Future: Digital Twin

The world model captures the **mind**—values and decision-making. A complete digital twin adds the **body**—voice, face, mannerisms.

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE DIGITAL TWIN                    │
├────────────────────────────┬────────────────────────────────┤
│      WORLD MODEL           │      EMBODIMENT MODEL          │
│      (Mind) ✓              │      (Body) ○ Future           │
├────────────────────────────┼────────────────────────────────┤
│  What they'd decide        │  What they look/sound like     │
│  Why they'd decide it      │  How they move/gesture         │
│  Internal tensions         │  Mannerisms and style          │
├────────────────────────────┴────────────────────────────────┤
│                    EXPRESSION BRIDGE                        │
│   Tendency activation → Physical manifestation              │
│   MEANING at 0.8 → animated gestures, faster speech         │
│   SURVIVAL at 0.7 → tense voice, guarded posture            │
└─────────────────────────────────────────────────────────────┘
```

See [docs/future-digital-twin.md](docs/future-digital-twin.md) for the full vision.

---

## Connection to After Me

This system is infrastructure for [After Me](https://github.com/dOrgTech/afterme-contracts)—trustless estate planning with posthumous digital continuity.

The **diary** in After Me isn't just a journal. It's a training corpus. Enough video entries create the raw material for a digital double that:

- **Reasons from your values** (world model)
- **Looks and sounds like you** (embodiment model)
- **Carries cryptographic attestation** via embedded weight hashes—proving content derives from your authentic model, not an impersonation

This inverts the deepfake problem: instead of detecting fakes, you prove authenticity. Content without your embedded signature is suspect by default.

---

## Documentation

- [Architecture](docs/architecture.md) — System design and components
- [Core Concepts](docs/concepts.md) — Key ideas and terminology
- [Training Guide](docs/training.md) — ML-style training with convergence detection
- [API Reference](docs/api-reference.md) — Module documentation
- [Future: Digital Twin](docs/future-digital-twin.md) — Vision for complete representation

---

## Philosophy

> A person isn't monolithic. They're a coalition of drives finding equilibrium. The structure of that equilibrium IS the personality.

Traditional profiles describe a person. This system **is** the person's decision-making structure—or at least, a computationally faithful approximation that can speak for them when they no longer can.

Not immortality. But continuity.

---

## License

MIT
