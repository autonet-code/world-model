# World Model Architecture (Original Design)

> **Note**: This is the original design document from early development.
> The system has evolved significantly - see `/docs/architecture.md` for
> current architecture including adversarial dynamics, ML-style training,
> and validation.

## Overview

A system for representing a person's worldview, values, and capabilities as an emergent equilibrium of competing internal tendencies. Not a summary of a person, but a computational structure that IS the person's mind, capable of answering questions, revealing tensions, and evolving with new information.

## Core Insight

The same observation means different things depending on what you're optimizing for. "Built DAO governance framework" is PRO for "decentralized coordination," CON for "personal financial stability" (time spent not earning), PRO for "institutional alternatives."

Ideas are atomic. Position is relational. Structure emerges from competition.

---

## Architecture Layers

### Layer 1: Observations

Atomic units of information about the person. Sentence-sized, capped (~280 bytes). No inherent polarity - they're just facts.

```
Observation:
  id: string
  content: string           # sentence-sized, capped
  source_id: string         # which conversation/document
  timestamp: datetime
  embedding: vector         # for semantic similarity
  metadata: dict
```

Examples:
- "Spent 10 years building governance frameworks"
- "Lives paycheck to paycheck at 42"
- "Built Jurisdiction system with fractal DAO topology"
- "ViraTrace was rejected by EU for not being profitable"
- "Uses ayahuasca for psychological calibration"

### Layer 2: Agents (Human Tendencies)

Generic drives that exist in every human. Each agent has a token allocation representing how strong that tendency is in this specific person.

```
Agent:
  id: string
  tendency: string          # the human drive this represents
  allocation: float         # percentage of total tokens (sums to 1.0)
  description: string       # what this tendency optimizes for
```

Possible tendencies:
- **Survival/Security** - physical safety, resource acquisition, risk mitigation
- **Status/Recognition** - social standing, achievement, being valued
- **Meaning/Purpose** - significance, impact, legacy
- **Connection/Belonging** - relationships, community, being known
- **Autonomy/Sovereignty** - independence, self-determination, freedom
- **Comfort/Pleasure** - ease, enjoyment, avoiding pain
- **Curiosity/Understanding** - knowledge, making sense, exploration

The allocation is inferred from the data. Andrei might be:
- Meaning: 35%
- Autonomy: 30%
- Curiosity: 20%
- Survival: 8%
- Connection: 5%
- Status: 2%

### Layer 3: Trees (Value Hierarchies)

Each tree is rooted in a core value/concern. The roots ARE what the person cares about. Agents stake tokens on nodes within trees.

```
Tree:
  id: string
  root_value: string        # the core concern (e.g., "decentralized coordination")
  description: string
  root_node: Node
```

Trees for Andrei might include:
- "Decentralized coordination systems"
- "Post-institutional alternatives"
- "Personal sovereignty"
- "Mission completion"
- "Financial sustainability"

### Layer 4: Nodes (Ideas with Stakes)

Nodes are observations positioned within trees. The same observation can appear in multiple trees with different positions and weights.

```
Node:
  id: string
  observation_id: string    # links to atomic observation
  tree_id: string           # which tree this node is in
  parent_id: string | null  # null for root
  position: pro | con       # relative to parent

  # Stakes from agents
  stakes: [
    { agent_id: string, weight: float }
  ]

  # Computed
  direct_weight: float      # sum of stakes
  net_score: float          # after child propagation

  # Children
  pro_children: Node[]
  con_children: Node[]
```

### Layer 5: Weight Propagation

From the debate model:

```
net_score = direct_weight + sum(pro_children.net_score) - sum(con_children.net_score)
```

A node's strength isn't just its own stakes. It's adjusted by how well its sub-arguments hold up. Contradicting evidence (CON children) drags the score down.

### Layer 6: Cross-Tree Relationships

The same observation appears in multiple trees. This creates implicit relationships:

```
CrossReference:
  observation_id: string
  appearances: [
    { tree_id: string, node_id: string, position: pro|con, net_score: float }
  ]
```

Observations that flip polarity across trees reveal tensions. Observations that are consistently PRO across many trees are core to identity.

### Layer 7: Multi-Dimensional Space

Each tree is a dimension. An observation's position across all trees is its "meaning vector" for this person.

```
meaning_vector[observation] = [
  score_in_tree_1,
  score_in_tree_2,
  ...
  score_in_tree_n
]
```

Observations that cluster in this space share similar meaning-structures across the person's value systems.

---

## Key Mechanisms

### Agent Competition

Agents don't fight each other directly. They stake on nodes according to their tendencies. The equilibrium of all stakes produces the structure.

- Meaning-agent stakes heavily on "10 years on governance mission"
- Survival-agent stakes against "lives paycheck to paycheck"
- When they stake on the same node with opposite positions, there's tension
- The net score reflects the balance of power

### Internal Conflict

When tendencies pull opposite directions on high-stakes nodes:
- Meaning-agent: "pursuing mission is PRO for purpose"
- Survival-agent: "pursuing mission is CON for security"

The person experiences this as internal conflict. Computationally, it's contested nodes with high stakes from multiple agents.

### Learning / Reallocation

As new observations come in and nodes get reinforced:
1. Nodes update their weights
2. Agent performance is evaluated (did their stakes pay off?)
3. Token allocations shift over time
4. The person "learns" - becomes more oriented toward successful tendencies

### Fractal Expansion

Any node can be expanded into a sub-tree. "ViraTrace was rejected" can unpack into:
- "EU demanded profitability" (PRO for the rejection claim)
- "India violated privacy despite P2P design" (PRO)
- "3x more effective than alternatives" (CON - it was good, rejection was unjust)

The structure is self-similar at every level.

---

## Queries

### Score Query

"How does Andrei feel about institutional trust?"

1. Find or create tree rooted in "institutional trust"
2. Retrieve relevant observations
3. Agents stake on nodes (position + weight)
4. Propagate to get net_score
5. Return: score + supporting structure

### Comparison Query

"How similar are these two beliefs?"

1. Get meaning_vector for each observation
2. Compute distance in multi-dimensional space
3. Close = similar function across value systems

### Tension Query

"Where are Andrei's internal conflicts?"

1. Find nodes with high stakes from multiple agents
2. Where agents disagree on position (one PRO, one CON)
3. These are the contested areas

### Prediction Query

"How would Andrei respond to X?"

1. Parse X into relevant value trees
2. Find observations that bear on X
3. Compute scores across relevant trees
4. Generate response consistent with the equilibrium

---

## Relationship to Original Debate Model

The tokenized debates model from WeRule provides the foundation:

| Debate Model | World Model |
|--------------|-------------|
| Debate | Tree (value hierarchy) |
| Thesis | Root node (core value) |
| Argument | Node (observation + position) |
| Pro/Con | Same - binary edge direction |
| Stake (voting power) | Agent stake (tokens) |
| Author | Agent (human tendency) |
| Net Score | Same formula |
| Sentiment | Aggregate across trees |

The key extension: multiple simultaneous debates (trees), with the same arguments (observations) appearing across them, staked by different authors (agents) who represent internal tendencies rather than external voters.

---

## What This Captures

### More Than Summaries

- Summaries describe. This structure IS.
- Summaries pile up. This integrates, contests, resolves.
- Summaries are read. This is queried.

### The Multi-Faceted Self

- Same observation, different meanings in different contexts
- Internal plurality modeled as agent competition
- Tensions and conflicts are first-class citizens

### Evolution

- New observations flow in
- Stakes update
- Allocations shift
- The person grows/changes

### Expertise

- The nodes aren't just beliefs
- They encode HOW the person reasons
- The structure of arguments under a value = capability in that domain

---

## Open Questions

### Weight Assignment

How do agents decide how much to stake? Options:
- Relevance to their tendency (semantic similarity)
- Explicitness of evidence (direct quote vs inference)
- Recency
- Some learned function

### Initial Allocation

How do we bootstrap agent allocations?
- Predefined defaults (equal?)
- Inferred from initial corpus
- Set by the person explicitly

### Tree Discovery

How do we know what trees (core values) to create?
- Emerge from high-stakes observation clusters?
- Predefined human universals?
- Explicitly stated by the person?

### Adversarial Dynamics

How do agents "compete" exactly?
- Do they see each other's stakes?
- Is there a bidding process?
- What's the equilibrium mechanism?

### Scaling

- How many observations can this handle?
- How many trees?
- When does complexity become unmanageable?

---

## Next Steps

1. **Build observation extraction** - parse summaries into atomic observations
2. **Define agent set** - which tendencies, with what initial allocations
3. **Build tree structure** - nodes, edges, weight propagation
4. **Implement staking mechanism** - how agents assign weights
5. **Test with real data** - Andrei's conversation summaries
6. **Evaluate** - does it surface insights? predict responses? reveal tensions?

---

## The Bet

This architecture bets that:

1. Binary distinction (pro/con) + recursion = sufficient to model meaning
2. Human tendencies can be decomposed into a finite set of agents
3. Competition between agents produces coherent personality
4. The same structure applies to anyone (just different allocations + observations)
5. This is more useful than unstructured text at scale

If the bet pays off, this is a general architecture for modeling minds - not just for personal profiles, but for any domain where contested, contextual, hierarchical value matters.
