# Novelty: Theoretical Foundation

## Definition

**Novelty is deviation from a reference frame.**

The reference frame is itself accumulated novelty - the residue of all prior deviations that have been integrated into the observer's model.

## The Recursion

This definition is inherently recursive:

1. To measure novelty, you need a reference
2. The reference is made of prior novelty measurements
3. Those measurements required their own references
4. ...and so on, without bottoming out

**This is not a bug. It's the nature of the thing.**

Novelty cannot be defined in terms of something more fundamental. It IS the fundamental unit - the primitive capacity to register difference. Like how you can't define "length" without already having a concept of spatial difference.

## Structure of Reference Frames

A reference frame is not an opaque blob - it has STRUCTURE. This structure is part of the definition, not implementation detail:

### 1. Hierarchical Claims

Claims are arranged hierarchically. Parent claims are more fundamental than child claims:

```
"Decentralization enables flourishing" (depth 0, foundational)
├── "P2P networks resist censorship" (depth 1)
│   └── "Bitcoin enables trustless transactions" (depth 2)
└── "Centralized systems have single points of failure" (depth 1)
```

The depth of a claim indicates how foundational it is. Disrupting a shallow claim cascades to everything below it.

### 2. Adversarial Positioning (PRO/CON)

Observations don't just "exist" - they are POSITIONED relative to claims:

- **PRO**: Supports/entails the claim (evidence for)
- **CON**: Opposes/contradicts the claim (evidence against)
- **NEUTRAL**: Topically related but doesn't take a stance

This enables measuring **conflict**, not just **difference**.

### 3. Weighted Stakes

Tendencies (drives, values) stake weights on claims:

```
"Decentralization enables flourishing"
  - autonomy: 0.4 (cares a lot)
  - survival: 0.2 (moderate concern)
  - comfort: 0.05 (barely cares)
```

This creates the motivational structure - what the agent prioritizes.

### 4. Integrated Observations

Raw observations that have been absorbed. These form the evidential base from which positions draw.

## Why This Structure?

This isn't arbitrary - it's what's REQUIRED for novelty to work:

| Property | Requires |
|----------|----------|
| Depth-dependence | Hierarchy |
| Contradiction detection | PRO/CON positioning |
| Importance weighting | Stakes |
| Learning over time | Observation tracking |

Without hierarchy, all novelty is equal. Without positioning, we can only measure "different" not "conflicting." Without stakes, paradigm shifts look like trivia.

## Agent = Reference Frame

There is no separate "agent" that "has" a world model. The agent IS the reference frame at a particular configuration.

- `agent_t1` and `agent_t2` are different reference frames
- They share structural continuity but are distinct configurations
- Time enters through WHICH frame you anchor to, not as a separate parameter

## The Four Components of Novelty

When measuring novelty against a structured reference frame, four components emerge:

### 1. Integration Resistance

How hard is it to position this concept in existing claim hierarchies?

- Low: Fits naturally as PRO or CON to existing claims
- High: Doesn't relate to anything in the worldview

### 2. Contradiction Depth

If the concept opposes existing claims, how deep are those claims?

- Low: Opposes only leaf claims (easily revised beliefs)
- High: Opposes root claims (foundational assumptions)

### 3. Coverage Gap

What fraction of belief hierarchies have no place for this concept?

- Low: Relevant to many areas of the worldview
- High: Orthogonal to everything the agent cares about

### 4. Allocation Disruption

Would integrating this concept shift how attention is allocated across tendencies?

- Low: Fits within current priority structure
- High: Would restructure what the agent cares about (paradigm shift)

## Combination: Geometric Mean

Components combine via geometric mean:

```
composite = (IR × CD × CG × AD)^(1/4)
```

This ensures all dimensions matter. A concept that's maximally novel on one dimension but zero on another is less novel overall than one that's moderately novel on all dimensions.

## Theoretical vs Practical

| Aspect | Theoretical (core.py) | Practical (novelty.py) |
|--------|----------------------|------------------------|
| Anchor | Not required | Required |
| Structure | Defined abstractly | Implemented concretely |
| Similarity | Any valid function | Neural embeddings + NLI |
| Result | Abstract score | Detailed breakdown |

The theoretical layer defines what novelty IS. The practical layer makes it computable.

## Axioms

Any valid novelty implementation must satisfy:

1. **Reference dependence**: Same concept, different frames → different scores
2. **Absorption reduces novelty**: R2 = R1.absorb(X) → novelty(X, R2) < novelty(X, R1)
3. **Depth matters**: Contradicting ancestors is more novel than contradicting descendants
4. **Stake matters**: Affecting high-stake claims is more novel than low-stake
5. **Stance matters**: Contradiction > mere absence/difference

## Code Structure

```
core.py          - Abstract interfaces (Claim, ClaimHierarchy, ReferenceFrame, NoveltyMeasure)
novelty.py       - Concrete implementation (WorldModelReference, AnchoredNoveltyMeasure)
world_model/     - Reference frame components (Trees, Agents, Observations)
embeddings.py    - Neural similarity + NLI stance detection
test_harness.py  - Validation against historical epoch data
```
