# Attention Mechanisms

This system implements two complementary attention mechanisms:

1. **Attention-guided graph traversal**: Prioritizes which nodes to explore during the fetch/parse loop
2. **Novelty-modulated attention allocation**: Adjusts tendency weights based on novelty scores

## Attention-Guided Traversal

### Motivation

During graph exploration, many adjacent nodes are available at each step. Uniform exploration is wasteful. The attention mechanism prioritizes nodes likely to be relevant to the current inquiry.

### Salience Computation

For each candidate node, salience combines semantic relevance with notability:

```
salience(node) = semantic_relevance × notability_prior
```

Where:
- `semantic_relevance = 0.6 × query_similarity + 0.4 × frame_similarity`
- `notability_prior = 0.3 + 0.7 × normalized_sitelinks`

**Query similarity**: How related is this node to what we're exploring?
**Frame similarity**: How related is this node to existing frame claims? (potential for interaction)

### Implementation

```python
class AttentionContext:
    query_text: str           # The concept being explored
    frame_texts: List[str]    # Claims from the reference frame
    _cache: EmbeddingCache    # Cached embeddings for efficiency

    def compute_salience(self, node_label: str, notability: float) -> float:
        query_sim = self._cache.similarity(self.query_text, node_label)
        frame_sim = max(self._cache.similarity(c, node_label) for c in self.frame_texts)
        semantic_relevance = 0.6 * query_sim + 0.4 * frame_sim
        return semantic_relevance * (0.3 + 0.7 * notability)
```

### Performance

Embedding computation: ~10ms per text (cached after first computation)
Ranking: O(n) for n candidate nodes

This is substantially faster than running NLI (~200ms) on each candidate.

## Novelty-Modulated Attention Allocation

### Motivation

An agent's attention is normally distributed according to tendency weights (SURVIVAL, STATUS, MEANING, etc.). However, highly novel stimuli can "capture" attention regardless of these baseline allocations - similar to how a sudden threat captures attention even when focused on something else.

### The Attention Curve

A sigmoid function maps novelty to attention capture:

```
capture(novelty) = 1 / (1 + exp(-(novelty - midpoint) × steepness))
```

Parameters:
- `midpoint`: Novelty level at which capture = 0.5 (default: 0.5)
- `steepness`: How sharp the transition is (default: 10)

### Effective Allocations

Under novelty, allocations shift toward CURIOSITY:

```python
def effective_allocations(base_allocations, novelty, curiosity_bias=0.5):
    capture = sigmoid((novelty - midpoint) * steepness)
    curiosity_boost = capture * curiosity_bias

    for tendency, base_weight in base_allocations.items():
        if tendency == CURIOSITY:
            effective = base_weight + curiosity_boost * (1 - base_weight)
        else:
            effective = base_weight * (1 - curiosity_boost)

    return normalize(effective_allocations)
```

### Behavior by Novelty Level

| Novelty | Capture | Effect |
|---------|---------|--------|
| 0.0 | ~1% | Allocations match base tendencies |
| 0.25 | ~8% | Slight CURIOSITY boost |
| 0.5 | 50% | Balanced blend |
| 0.75 | ~92% | CURIOSITY dominates |
| 1.0 | ~99% | Near-total attention capture |

### Preset Profiles

Three profiles for different agent behaviors:

| Profile | Midpoint | Steepness | Curiosity Bias | Behavior |
|---------|----------|-----------|----------------|----------|
| EXPLORER | 0.3 | 8 | 0.7 | Novelty captures attention early |
| BALANCED | 0.5 | 10 | 0.5 | Default behavior |
| CONSERVATIVE | 0.7 | 12 | 0.3 | Resists novelty influence |

### Integration with Staking

The attention curve affects stake weights during the Arena's staking phase:

```python
state = AttentionState(agent_set, curve=BALANCED_CURVE)
state.update_novelty(measured_novelty)

for tendency in tendencies:
    stake_weight = state.get_stake_weight(tendency)
    # Use stake_weight when placing stakes on nodes
```

This allows novelty to influence which tendencies have the most say in how new observations are positioned in belief trees.

## Empirical Observations

From test runs with default parameters:

- At novelty=0.0: CURIOSITY holds ~10% (base allocation), CONNECTION dominates at 20%
- At novelty=0.5: CURIOSITY jumps to ~33% and becomes dominant
- At novelty=1.0: CURIOSITY reaches ~55%

The curve ensures smooth transitions rather than abrupt switching.
