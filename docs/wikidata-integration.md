# Wikidata Integration

## Rationale

Wikidata provides a structured knowledge graph with 100M+ items. Unlike pure embedding-based approaches, it offers:

- Explicit hierarchical relations (instance-of, subclass-of, part-of)
- Notability signals (sitelinks count indicates coverage across Wikipedia languages)
- Cross-domain coverage (concepts from science, culture, geography, etc.)
- Stable identifiers (Q-IDs persist across label changes)

## Data Structures

### WikidataEntity

Retrieved via `get_entity(qid)`:

```python
@dataclass
class WikidataEntity:
    qid: str
    label: str
    description: str
    aliases: List[str]
    sitelinks_count: int
    instance_of: List[str]      # P31
    subclass_of: List[str]      # P279
    part_of: List[str]          # P361
    has_parts: List[str]        # P527
    uses: List[str]             # P2283
```

### GraphMetrics

Retrieved via `get_graph_metrics(qid)`:

```python
@dataclass
class GraphMetrics:
    qid: str
    depth: int                  # Distance from root in hierarchy
    total_ancestors: int
    total_descendants: int
    centrality_ratio: float     # descendants / ancestors
    sitelinks: int
    integration_score: float    # Combined notability metric
```

**Centrality ratio interpretation:**
- High ratio (many descendants, few ancestors): Abstract/foundational concept
- Low ratio (few descendants, many ancestors): Specific/leaf concept

### AncestryPath

Retrieved via `get_ancestry(qid)`:

```python
@dataclass
class AncestryPath:
    qid: str
    ancestors: List[str]        # Ordered from immediate parent to root
    depth: int
    paths_to_root: int          # Multiple inheritance count
```

## Smart Expansion

The `get_topically_related()` function prioritizes neighbors by:

1. **Relation type weighting**: P31 (instance-of) and P279 (subclass-of) weighted higher than P527 (has-parts)
2. **Sitelinks filtering**: Excludes items with <5 sitelinks (low notability)
3. **Reference QID proximity**: Boosts items that connect to existing frame concepts

This reduces noise from obscure Wikidata entries while maintaining semantic relevance.

## Notability as Prior

Sitelinks count (number of Wikipedia language editions linking to the item) serves as a prior for importance:

```python
notability = min(log(sitelinks + 1) / log(150), 1.0)
```

This normalizes to [0, 1] with 150 sitelinks mapping to 1.0. Items with <5 sitelinks are typically excluded.

## Containment Check

A concept is considered "contained" in the frame if:

1. Its Q-ID is directly in the integrated set
2. It has direct relations (subclass, instance, part-of) to frame concepts
3. Its immediate ancestors are in the frame

Containment indicates low novelty - the concept is already represented or closely related to existing knowledge.

## Stance Detection via Graph Structure

Without running NLI (which is slower), stance can be inferred from graph topology:

| Relation | Interpretation |
|----------|---------------|
| Content is subclass/instance of claim | PRO (supports hierarchy) |
| Content shares ancestry with claim | PRO (weak support) |
| Content has very different centrality | Potential tension |
| Explicit "different from" (P1889) | CON |

## Limitations

- Wikidata coverage is uneven across domains (better for encyclopedic topics)
- Some relations are noisy or incomplete
- Query latency (~100-500ms per entity fetch)
- No semantic understanding - purely structural

## API Caching

Results are cached to reduce repeated queries:
- Entity data cached by Q-ID
- Ancestry paths cached by Q-ID
- Search results cached by query string

Cache invalidation is manual; for long-running processes, periodic clearing may be needed.
