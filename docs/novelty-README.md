# Novelty

Agent-relative novelty computation using structured reference frames and knowledge graphs.

## Core Concept

Novelty is not an intrinsic property of information. It emerges from the relationship between new concepts and an agent's existing knowledge structure. This system measures that relationship by:

1. Representing knowledge as hierarchical claim trees with weighted stakes
2. Using Wikidata as a grounded knowledge graph for traversal
3. Running a fetch/parse loop that terminates when integration, contradiction, or orthogonality is detected
4. Computing novelty from four orthogonal dimensions derived from how the loop terminates

## Quick Start

```python
from wikidata_probe import measure_novelty

# Measure novelty of "blockchain" against a classical economics frame
result = measure_novelty(
    concept="blockchain",
    reference_concepts=["money", "bank", "currency", "transaction", "ledger"]
)

print(f"Termination: {result.termination.value}")
print(f"Composite novelty: {result.composite:.3f}")
```

## Documentation

| Document | Description |
|----------|-------------|
| [Theory](docs/THEORY.md) | Theoretical foundation and definitions |
| [Formalization](docs/FORMALIZATION.md) | Mathematical specification with axioms |
| [Architecture](docs/architecture.md) | System components and data flow |
| [Wikidata Integration](docs/wikidata-integration.md) | Knowledge graph specifics |
| [Attention](docs/attention.md) | Attention-guided traversal and novelty-modulated allocation |

## Project Structure

```
novelty/
├── core.py              # Abstract interfaces (NoveltyProbe, ReferenceFrame)
├── wikidata_probe.py    # Wikidata-backed implementation
├── wikidata.py          # Wikidata API queries
├── embeddings.py        # Sentence embeddings and NLI
├── world_model/         # Belief structure components
│   ├── tree.py          # Binary trees with PRO/CON positioning
│   ├── agent.py         # Tendencies and stake allocations
│   └── attention.py     # Novelty-modulated attention
├── docs/                # Documentation
└── tests/               # Test suite
```

## Key Claims

Claims this system makes that can be tested:

1. **Reference dependence**: The same concept yields different novelty scores against different frames
2. **Absorption reduces novelty**: After integrating a concept, its novelty against the updated frame is lower
3. **Depth matters**: Contradicting foundational claims produces higher novelty than contradicting derived claims
4. **Stake matters**: Affecting high-stake claims produces higher novelty
5. **Attention capture**: High novelty shifts allocation toward CURIOSITY tendency

## Requirements

- Python 3.10+
- sentence-transformers (for embeddings)
- transformers (for NLI)
- requests (for Wikidata API)

## Limitations

- Wikidata coverage varies by domain
- NLI-based stance detection has ~200ms latency per inference
- No learning/adaptation of the reference frame during measurement
- Composite score uses fixed geometric mean weighting

## License

MIT
