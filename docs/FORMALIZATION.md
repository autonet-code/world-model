# Novelty: Mathematical Formalization

## Primitives

Let **C** be the space of all concepts (anything that can be evaluated for novelty).

Let **Σ** = {PRO, CON, NEUTRAL} be the stance space.

## Similarity Function

```
sim: C × C → [0,1]
```

Measures topical relatedness between concepts. Two concepts with low similarity
cannot meaningfully support or contradict each other - they're in different domains.

Implementation: neural embeddings (sentence-transformers) with cosine similarity.

## Stance Detector

The stance detector σ is a **guarded function**:

```
σ: C × Claims → Σ × [0,1]

σ(c, v) = { (NEUTRAL, γ)    if sim(c, v) < θ_topic
          { ψ(c, v)         otherwise
```

where:
- **θ_topic** ∈ [0,1] is the topical relevance threshold (e.g., 0.55)
- **γ** is a low default confidence (e.g., 0.3)
- **ψ**: C × Claims → Σ × [0,1] is the core stance inference function

The guard condition ensures that stance is only evaluated between topically related
concepts. This prevents false contradictions between unrelated domains (e.g.,
"photosynthesis" cannot contradict claims about "banking" - they're orthogonal).

Implementation of ψ: Natural Language Inference (NLI) model (DeBERTa-MNLI).

## Reference Frame

A **reference frame** R is a tuple:

```
R = (H, T, O, σ, sim)
```

where:
- **H** = {h₁, h₂, ..., hₘ} is a set of claim hierarchies
- **T**: H × Claims → ℝ⁺ is the stake function (tendency weights on claims)
- **O** ⊂ C is the set of integrated observations
- **σ**: C × Claims → Σ × [0,1] is the guarded stance detector
- **sim**: C × C → [0,1] is the similarity function

### Claim Hierarchy

A claim hierarchy h ∈ H is a rooted tree:

```
h = (V, E, ρ, d)
```

where:
- **V** is a set of claims (nodes)
- **E** ⊆ V × V is the parent-child relation
- **ρ** ∈ V is the root (foundational claim)
- **d**: V → ℕ is the depth function, d(ρ) = 0

For any claim v ∈ V:
- d(v) = 0 means v is foundational
- d(v) > d(u) means v depends on u (v is derived, u is more fundamental)

### Total Stake

```
S(R) = Σᵢ Σᵥ∈Hᵢ T(hᵢ, v)
```

## The Novelty Loop

### Focus

A **focus** is the current point of examination:

```
f = (c, n, π)
```

where:
- **c** ∈ C is the concept being examined
- **n** ∈ ℕ is the iteration count (depth from initial concept)
- **π** is the path trace (sequence of concepts visited)

### Termination

**τ** ∈ {INTEGRATED, CONTRADICTS, ORTHOGONAL, DISRUPTS, MAX_ITER}

### Fetch Function

```
fetch: Focus × R → D ∪ {∅}
```

Maps current focus and frame to data D about that concept, or ∅ if not found.

### Parse Function

```
parse: D × Focus × R → (τ, f', α, μ)
```

where:
- **τ** is termination condition (or ⊥ if continuing)
- **f'** is next focus (if continuing)
- **α** ∈ C ∪ {∅} is content to absorb
- **μ** = (s, δ, w) are metrics: similarity, contradiction depth, stake affected

### The Loop

Given concept c₀ and frame R₀:

```
f₀ = (c₀, 0, [])
i = 0

WHILE i < n_max:
    i = i + 1
    d = fetch(fᵢ₋₁, Rᵢ₋₁)
    (τ, f', α, μ) = parse(d, fᵢ₋₁, Rᵢ₋₁)

    IF τ ≠ ⊥:
        RETURN NoveltyResult(τ, i, μ)

    IF α ≠ ∅:
        Rᵢ = absorb(Rᵢ₋₁, α)
    ELSE:
        Rᵢ = Rᵢ₋₁

    fᵢ = f'

RETURN NoveltyResult(MAX_ITER, n_max, μ)
```

## Termination → Components

The novelty result N is derived from HOW the loop terminates:

```
N = (τ, i, IR, CD, CG, AD)
```

### Integration Resistance (IR)

```
IR = min(i / n_max, 1)
```

More iterations before termination → higher resistance.

### Contradiction Depth (CD)

```
CD = { 1 - δ/d_max    if τ = CONTRADICTS
     { 0              otherwise
```

where δ is the depth of the contradicted claim, d_max is max depth in R.

Shallower contradiction (lower δ) → higher CD (more foundational disruption).

### Coverage Gap (CG)

```
CG = { 1.0                    if τ = ORTHOGONAL
     { 0.8                    if τ = MAX_ITER
     { min(i/n_max, 1) × 0.5  otherwise
```

### Allocation Disruption (AD)

```
AD = { w / S(R)    if τ = DISRUPTS
     { 0.8         if τ = CONTRADICTS ∧ δ ≤ 1
     { 0.1         otherwise
```

where w is stake affected, S(R) is total stake.

## Composite Score

```
N_composite = (∏ᵢ (Nᵢ + ε))^(1/4)
```

Geometric mean of components. All dimensions must be non-trivial for high composite novelty.

## Absorption

```
absorb: R × C → R'
```

Creates new frame R' = (H', T', O', σ', sim') where:
- O' = O ∪ {c}
- H', T' may be updated based on positioning of c relative to existing claims
- σ' and sim' are typically unchanged (same functions, expanded domain)

Key property: **Absorption reduces novelty**

```
novelty(c, absorb(R, c)) < novelty(c, R)
```

## Axioms

**A1. Reference Dependence**
```
∀c ∈ C, ∀R₁ ≠ R₂: novelty(c, R₁) ≠ novelty(c, R₂) in general
```

**A2. Absorption Monotonicity**
```
R' = absorb(R, c) ⟹ novelty(c, R') < novelty(c, R)
```

**A3. Depth Ordering**
```
If v₁ is ancestor of v₂ in H, and c contradicts v₁:
novelty(c, R) > novelty(c', R) where c' only contradicts v₂
```

**A4. Stake Ordering**
```
If T(h, v₁) > T(h, v₂) and c affects v₁, c' affects v₂:
novelty(c, R) > novelty(c', R), ceteris paribus
```

**A5. Termination Guarantee**
```
∀c ∈ C, ∀R: the loop terminates in at most n_max iterations
```

**A6. Stance Asymmetry**
```
If σ(c, v) = (CON, p) with p > θ:
novelty(c, R) > novelty(c', R) where σ(c', v) = (NEUTRAL, p)
```

Contradiction is more novel than mere unrelatedness.

**A7. Topical Relevance Requirement**
```
sim(c, v) < θ_topic ⟹ σ(c, v) = (NEUTRAL, γ)
```

Stance detection requires topical relevance. Concepts in unrelated domains cannot
meaningfully contradict each other - they are orthogonal, not adversarial.

This prevents the degenerate case where a stance detector (e.g., an NLI model)
produces false positives on unrelated inputs.

**A8. Similarity Symmetry**
```
sim(c₁, c₂) = sim(c₂, c₁)
```

Topical similarity is symmetric - if A is related to B, B is related to A.

## Connection to Information Theory

In the degenerate case where:
- H contains a single hierarchy with depth 0 (flat)
- T is uniform across all claims
- σ returns only NEUTRAL (no adversarial structure)
- sim returns 1 for all pairs (everything is related)

The framework reduces to measuring **coverage** - whether c is in O or not.

With a probability measure P over C:
```
novelty(c, R) ≈ -log P(c ∈ O)
```

This recovers Shannon surprise as a special case.

Our framework generalizes this by adding:
1. **Hierarchical structure** (depth matters)
2. **Adversarial positioning** (contradiction vs. absence)
3. **Stake weighting** (importance varies)
4. **Iterative discovery** (the loop, not one-shot)
5. **Topical gating** (similarity determines if stance applies)

## The Role of Similarity

The similarity function sim partitions the concept space into regions:

```
For any claim v ∈ H:
  Related(v) = {c ∈ C : sim(c, v) ≥ θ_topic}
  Orthogonal(v) = {c ∈ C : sim(c, v) < θ_topic}
```

Only concepts in Related(v) can support or contradict v.
Concepts in Orthogonal(v) are automatically NEUTRAL - they exist in a different
semantic domain and have no bearing on v.

This captures the intuition that:
- "Bitcoin" can contradict "traditional banking is stable" (same domain)
- "Photosynthesis" cannot contradict "traditional banking is stable" (different domains)

The similarity function thus acts as a **domain filter** before stance detection.
