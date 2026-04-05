# Attention

Dynamic resource allocation for symbol streams.

## What is Attention?

Attention is the mechanism by which a system with finite resources decides what to process. It's not a single function but an emergent property of cascading filters.

This library models attention as:

1. **Sequences** - Bounded buffers of symbols that force prioritization
2. **Processes** - Subscribers that match patterns and propagate associations
3. **Salience** - Value functions that determine what persists

## Core Insight

From first principles, consciousness can be modeled as:

> "A very short sequence of symbols as the starting point... and a value assigning function as the main method. Every time an input stream produces a pattern, it is assigned a set of features and, if deemed relevant, it is appended to this sequence."

Attention emerges from:
- Multiple sequences with different capacities
- Processes that hash symbols and look up responses
- Reinforcement of repeated/convergent signals
- Graduation of high-value items to longer-term storage

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from attention import Sequence, Symbol, RepetitionProcess

# Create sequences with different capacities
conscious = Sequence("conscious", capacity=7, min_value=0.5)
working = Sequence("working", capacity=20, min_value=0.3)

# Process that detects repeated symbols
repeater = RepetitionProcess(
    "repeat_detector",
    inputs=[working],
    outputs=[conscious],
    min_repetitions=2,
    boost_factor=1.5
)
repeater.start()

# Publish symbols
working.publish(Symbol(data="hello", value=0.4))
working.publish(Symbol(data="world", value=0.4))
working.publish(Symbol(data="hello", value=0.4))  # Repeat! Boosted to conscious
```

## Core Concepts

### Symbol

The atomic unit of attention. Carries data, a value (salience), and metadata.

```python
symbol = Symbol(
    data="the quick brown fox",
    value=0.7,
    metadata={"source": "user_input"}
)
```

### Sequence

A bounded buffer with pub/sub semantics. When full, low-value items are evicted.

```python
seq = Sequence(
    name="working_memory",
    capacity=20,
    min_value=0.3,
    eviction=EvictionPolicy.DROP_LOWEST
)

# Subscribe to new symbols
seq.subscribe("logger", lambda s: print(s.data))

# Publish
seq.publish(Symbol(data="important", value=0.9))
```

### Process

Subscribes to sequences, matches patterns, publishes to other sequences.

```python
class MyProcess(Process):
    def match(self, symbol: Symbol) -> Optional[Match]:
        if "urgent" in str(symbol.data):
            return Match(
                pattern_id="urgent",
                symbol=symbol,
                confidence=1.0,
                response=symbol.boost(0.5)
            )
        return None
```

Built-in processes:
- **LookupProcess** - Hash table matching
- **RepetitionProcess** - Detects repeated symbols
- **ConvergenceProcess** - Detects multi-source agreement
- **LoopDetector** - Breaks repetitive loops

### Salience

Functions that compute attention-worthiness.

```python
from attention import (
    CompositeSalience,
    recency_salience,
    keyword_salience
)

salience = CompositeSalience([
    recency_salience(half_life_seconds=30),
    keyword_salience({"urgent": 0.3, "error": 0.4}),
], aggregation="max")

score = salience(symbol)
```

## Integration

### With full-duplex

Feed multimodal stream symbols into attention:

```python
from full_duplex import GeminiStream, Symbol as DuplexSymbol
from attention import Sequence, Symbol

input_seq = Sequence("perception", capacity=50)

async for item in stream.receive():
    if isinstance(item, DuplexSymbol):
        input_seq.publish(Symbol(
            data=item.data,
            value=0.5,  # or compute salience
            metadata={"modality": item.modality.value}
        ))
```

### With novelty

Use novelty scores as salience:

```python
from attention import NoveltyAdapter

# Assuming novelty system provides this function
def compute_novelty(data) -> float:
    # Returns 0-1 novelty score
    ...

salience_fn = NoveltyAdapter(compute_novelty, scale=1.0)
```

### With life (world model)

Use tendency allocations as salience:

```python
from attention import AllocationAdapter

adapter = AllocationAdapter(
    get_allocations=lambda: world_model.agents.allocations(),
    classify_fn=lambda data: classify_tendency(data)
)
```

## Architecture

```
                    ┌─────────────┐
  Input ──────────► │  Sequence   │ ◄──────── Salience
                    │  (bounded)  │           Function
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐   ┌────────┐   ┌────────┐
         │Process │   │Process │   │Process │  (pattern matching)
         └───┬────┘   └───┬────┘   └───┬────┘
             │            │            │
             └────────────┼────────────┘
                          ▼
                    ┌─────────────┐
                    │  Sequence   │
                    │  (output)   │
                    └─────────────┘
```

Multiple processes can subscribe to the same sequence. When patterns match, they publish to output sequences. Convergent signals (multiple processes publishing similar symbols) indicate high salience.

## Avoiding Loops

The LoopDetector process watches for repetitive patterns:

```python
loop_detector = LoopDetector(
    "loop_break",
    inputs=[conscious],
    outputs=[interrupt_seq],
    pattern_length=3,
    max_repeats=2
)
```

When a loop is detected, it emits a break signal that downstream processes can use to redirect attention.

## Philosophy

> "At the most basic level, intelligence is the ability to act in such a way that it increases your options for future action."

Attention serves intelligence by:
1. Filtering noise (limited capacity forces selection)
2. Reinforcing signal (repetition and convergence boost value)
3. Breaking loops (detecting stuck patterns)
4. Enabling association (processes link related concepts)

The system doesn't define *what* to attend to - that comes from salience functions you provide. It defines *how* attention flows through a cascade of prioritizing filters.

## License

MIT
