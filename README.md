# World Model

**Modeling personality as adversarial equilibrium, with novelty-driven attention**

A computational framework for representing a person's decision-making as competing internal drives that stake evidence, debate in adversarial arenas, and shift allocations based on novelty-modulated attention. Not a profile. Not a summary. A dynamic system that *is* the personality, in computational form.

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

## Architecture

Three interlocking systems form a feedback loop:

```
Input (text/voice/tweets)
    │
    ▼
┌──────────┐     "How surprising is this?"
│ NOVELTY  │     Fetch/parse loop against reference frame
│          │     4 dimensions: integration resistance,
│          │     contradiction depth, coverage gap,
│          │     allocation disruption
└────┬─────┘
     │ novelty score
     ▼
┌──────────┐     "Should I attend to it?"
│ATTENTION │     Bounded symbol streams force prioritization
│          │     Novelty captures attention → shifts toward CURIOSITY
│          │     Salience filters: recency, convergence, repetition
└────┬─────┘
     │ promoted observations
     ▼
┌──────────┐     "What do my tendencies think?"
│  ARENA   │     7 tendencies propose claims, stake evidence
│ (life)   │     Winners gain allocation, losers shrink
│          │     Equilibrium IS the personality
└────┬─────┘
     │ updated allocations
     └──────────────► feeds back into novelty & attention
```

---

## Seven Tendencies

Seven universal human drives act as **agents** competing for influence:

| Tendency | Default | Question It Asks |
|----------|---------|------------------|
| SURVIVAL | 18% | "Is this safe? Do I have enough?" |
| CONNECTION | 20% | "Am I known? Do I belong?" |
| COMFORT | 18% | "Is this pleasant? Can I sustain this?" |
| STATUS | 12% | "Am I respected? Do I matter to others?" |
| AUTONOMY | 12% | "Am I free? Can I choose?" |
| MEANING | 10% | "Does this matter? Will it outlast me?" |
| CURIOSITY | 10% | "Do I understand? What's there to learn?" |

Each agent:
1. **PROPOSES** claims about what matters
2. **STAKES** observations as evidence (PRO or CON)
3. **WINS or LOSES** debates based on evidence strength
4. **GAINS or LOSES** allocation based on outcomes

The equilibrium that emerges—which tendencies dominate, where they conflict, how they resolve—**is** the personality.

---

## Quick Start

### Adversarial Debate

```python
from world_model import create_world_model, Arena

model = create_world_model("Person", "observations.json")

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

### Novelty Measurement

```python
from world_model.novelty import measure_against_claims

result = measure_against_claims(
    concept="Bitcoin",
    claim_texts=[
        "Traditional banking provides security",
        "Trust in institutions is necessary",
    ]
)

print(f"Termination: {result.termination}")  # CONTRADICTS_ROOT
print(f"Composite novelty: {result.composite:.3f}")
```

### Attention Routing

```python
from world_model.attention import Sequence, Symbol, NoveltyProcess

# Bounded buffers force prioritization (like working memory)
conscious = Sequence("conscious", capacity=7, min_value=0.5)
working = Sequence("working", capacity=20, min_value=0.3)

# Novel items auto-promote from working memory to conscious attention
proc = NoveltyProcess("filter", inputs=[working], outputs=[conscious])
proc.start()

working.publish(Symbol(data="something novel", value=0.6))
```

### Full Integration Loop

```python
from world_model.integration import AttentionBridge

bridge = AttentionBridge(
    agent_set=model.agents,
    observation_store=model.observations,
)

event = bridge.process("New information about the person")
print(f"Novelty: {event.novelty_score:.2f}")
print(f"Promoted: {event.was_promoted}")
print(f"Dominant tendency: {event.dominant_tendency}")
```

---

## Package Structure

```
world_model/
├── models/              Core data structures
│   ├── observation.py       Atomic facts (~280 bytes, content-hash deduped)
│   ├── agent.py             Tendency enum, Agent, AgentSet (allocations sum to 1)
│   └── tree.py              Value hierarchies, PRO/CON positioning, weight propagation
│
├── dynamics/            Adversarial competition
│   ├── arena.py             Debate orchestration (propose → stake → resolve → reallocate)
│   └── trainer.py           ML-style training (epochs, convergence, learning rate decay)
│
├── novelty/             Novelty measurement
│   ├── core.py              Abstract interfaces (NoveltyProbe, ReferenceFrame)
│   ├── hybrid_probe.py      Wikidata graph + Neural NLI (recommended)
│   ├── neural_probe.py      Pure NLI, no external dependencies
│   ├── wikidata_probe.py    Wikidata graph structure only
│   ├── wikidata.py          Wikidata API integration
│   └── embeddings.py        Sentence embeddings + NLI stance detection
│
├── attention/           Attention routing
│   ├── curves.py            Novelty → attention allocation (sigmoid curves)
│   ├── sequence.py          Bounded symbol buffers with eviction policies
│   ├── process.py           Pattern matching (repetition, convergence, loop detection)
│   ├── salience.py          Value functions (recency, keywords, novelty, allocations)
│   └── novelty_process.py   Novelty-aware routing between sequences
│
├── staking/             Evidence staking
│   ├── staker.py            Tendency-specific evidence analysis
│   └── hierarchical_staker.py  Recursive claim decomposition
│
├── extraction/          Observation extraction
│   ├── observation_extractor.py  Text → atomic observations
│   ├── voice_extractor.py       Voice transcripts → voice profile
│   └── tweet_processor.py       Twitter data → observations
│
├── agents/              Autonomous agents
│   └── moltbook_agent.py    Social media agent driven by world model
│
├── storage/             Persistence
│   ├── world_model_store.py     JSON serialization
│   └── firestore_adapter.py     Google Firestore
│
└── integration.py       AttentionBridge + ArenaFeedback (wires everything together)

api/                     FastAPI service
data/                    Sample encoded world model
docs/                    Full documentation
scripts/                 CLI tools for extraction
tests/                   Test suite
```

---

## How Novelty Works

Novelty is not a one-shot score. It's the **termination reason** of a fetch/parse loop that explores how a concept relates to existing beliefs.

```
WHILE NOT TERMINATED:
    data = fetch(focus)              # Query knowledge graph
    verdict = parse(data, frame)     # Evaluate against beliefs

    IF verdict.terminates:
        BREAK                        # Termination reason IS the novelty
    ELSE:
        frame = frame.absorb(data)   # Update reference frame
        focus = verdict.next_focus   # Expand search outward
```

**Termination reasons:**
- `INTEGRATED` — concept fits naturally (low novelty)
- `CONTRADICTS_ROOT` — opposes foundational beliefs (high novelty)
- `ORTHOGONAL` — no connection found despite search (high novelty)
- `DISRUPTS` — would restructure tendency allocations (high novelty)

**Four dimensions** (combined via geometric mean):

| Dimension | Measures |
|-----------|----------|
| Integration Resistance | How many iterations before the loop terminates |
| Contradiction Depth | How foundational the conflicting belief is |
| Coverage Gap | Fraction of worldview untouched by the concept |
| Allocation Disruption | How much tendency priorities would shift |

---

## How Attention Works

Attention is modeled as **cascading symbol streams** with finite capacity:

```
                    ┌─────────────┐
  Input ──────────► │  Sequence   │ ◄──────── Salience
                    │  (bounded)  │           Function
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐   ┌────────┐   ┌────────┐
         │Process │   │Process │   │Process │
         └───┬────┘   └───┬────┘   └───┬────┘
             │            │            │
             └────────────┼────────────┘
                          ▼
                    ┌─────────────┐
                    │  Sequence   │
                    │  (output)   │
                    └─────────────┘
```

- **Sequences** are bounded buffers (like Miller's 7±2). Overflow forces eviction — low-value items are dropped.
- **Processes** watch sequences for patterns: repetition, convergence across sources, loops.
- **Salience functions** assign value: recency decay, keyword matching, novelty scores, tendency allocations.
- **Novelty captures attention**: High novelty shifts allocation toward CURIOSITY via sigmoid curves, configurable per personality (explorer, balanced, conservative).

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

## The Bet

This architecture bets that:

1. **Binary distinction + recursion** = sufficient to model meaning
2. **Seven tendencies** = comprehensive but tractable agent set
3. **Competition** = produces coherent personality from plurality
4. **Novelty is relative** = same information, different surprise depending on who's hearing it
5. **Attention is finite** = bounded buffers produce intelligent filtering
6. **Same structure** = applies to anyone (swap observations + allocations)

If the bet pays off, this is a general architecture for modeling minds.

---

## Installation

```bash
# Core (no external dependencies)
pip install -e .

# With novelty measurement (requires ML models)
pip install -e ".[novelty]"

# With API server
pip install -e ".[api]"

# Everything
pip install -e ".[all]"
```

Requires Python 3.11+.

---

## Documentation

- [Architecture](docs/architecture.md) — System design and components
- [Core Concepts](docs/concepts.md) — Key ideas and terminology
- [Training Guide](docs/training.md) — ML-style training with convergence detection
- [API Reference](docs/api-reference.md) — Module documentation
- [Novelty Theory](docs/THEORY.md) — Theoretical foundation for novelty measurement
- [Novelty Formalization](docs/FORMALIZATION.md) — Mathematical specification
- [Wikidata Integration](docs/wikidata-integration.md) — Knowledge graph specifics
- [Attention Mechanisms](docs/attention.md) — Salience, routing, and novelty-modulated allocation
- [Future: Digital Twin](docs/future-digital-twin.md) — Vision for complete embodiment

---

## Connection to After Me

This system is infrastructure for [After Me](https://github.com/dOrgTech/afterme-contracts)—trustless estate planning with posthumous digital continuity.

The world model captures the **mind**—values and decision-making. A complete digital twin adds the **body**—voice, face, mannerisms. Together they create a digital double that:

- **Reasons from your values** (world model)
- **Looks and sounds like you** (embodiment model, future)
- **Carries cryptographic attestation** via embedded weight hashes

This inverts the deepfake problem: instead of detecting fakes, you prove authenticity.

---

## License

MIT
