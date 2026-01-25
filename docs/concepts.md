# Core Concepts

## The Central Insight

> Ideas are atomic. Position is relational.

An observation like "Lives paycheck to paycheck at 42" has no inherent meaning. Its meaning emerges from **context** - what you're optimizing for, what question you're answering.

In one context, it's evidence of financial risk. In another, evidence of sacrifice for purpose. In another, evidence of choosing freedom over salary.

The World Model captures this by allowing the same observation to appear in multiple trees with different positions (PRO or CON).

## Observations vs. Beliefs

Traditional profile systems store **beliefs**:
- "Andrei believes in decentralization"
- "Andrei values autonomy"

These are already interpreted. Position is baked in.

The World Model stores **observations** - atomic facts without inherent polarity:
- "Built DAO governance framework"
- "Rejected VC funding"
- "Works alone"

Position emerges from competition between agents.

## The Debate Model Foundation

The architecture extends a tokenized debate model with binary tree structure:

```
                    [Claim]
                   /       \
                PRO         CON
               /   \       /   \
            [obs] [obs]  [obs] [obs]
```

**Weight propagation formula**:
```
net_score = direct_weight + sum(pro_children.score) - sum(con_children.score)
```

A claim's strength depends on:
1. Direct stakes (how much agents invested)
2. Supporting evidence (PRO children)
3. Contradicting evidence (CON children)

## Agents as Internal Plurality

A person isn't monolithic. They're a coalition of drives finding equilibrium.

The seven tendencies model universal human drives:

| Tendency | Question It Asks |
|----------|------------------|
| SURVIVAL | "Is this safe? Do I have enough?" |
| STATUS | "Am I respected? Do I matter to others?" |
| MEANING | "Does this matter? Will it outlast me?" |
| CONNECTION | "Am I known? Do I belong?" |
| AUTONOMY | "Am I free? Can I choose?" |
| COMFORT | "Is this pleasant? Can I sustain this?" |
| CURIOSITY | "Do I understand? What's there to learn?" |

The **allocation** represents how strong each drive is in a specific person. Andrei might be high MEANING/AUTONOMY, low STATUS/COMFORT.

## Adversarial Competition

This is what makes the model "alive."

In passive models, agents just categorize observations. In the adversarial model:

1. **Agents PROPOSE**: Each agent makes a claim about what matters
2. **Agents STAKE**: Support their claims, undermine competitors
3. **Agents WIN/LOSE**: Based on how well evidence supports claims
4. **Allocations SHIFT**: Winners gain influence for future debates

The competition surfaces **tensions**. When SURVIVAL and MEANING stake the same observation with opposite positions, that's internal conflict made visible.

## The Equilibrium IS The Personality

Traditional models describe a person. This model **is** the person's decision-making structure.

The equilibrium - which tendencies dominate, how they relate, where they conflict - constitutes personality. Not a representation of it. The thing itself, in computational form.

## Training as Calibration

ML-style training isn't teaching the model new information. It's **calibrating** the agent allocations to match the evidence.

- **Epochs**: Multiple rounds of debate
- **Convergence**: When allocations stabilize (tendencies find equilibrium)
- **Validation**: Testing on held-out observations
- **Learning rate**: How fast allocations shift per round

The goal isn't accuracy in a traditional sense. It's finding the equilibrium that best explains the person's observations.

## Multi-Lens Meaning

The same observation has different **meaning vectors** across trees:

```
"Lives paycheck to paycheck"
  ├── SURVIVAL tree: +0.8 (strong evidence)
  ├── MEANING tree: +0.6 (sacrifice for purpose)
  ├── AUTONOMY tree: +0.4 (chose freedom)
  ├── COMFORT tree: +0.7 (unsustainable)
  └── STATUS tree: -0.3 (against status claims)
```

Observations that flip polarity across trees reveal **tensions**. Observations that are consistently PRO reveal **core identity**.

## The Bet

This architecture bets that:

1. **Binary distinction + recursion** = sufficient to model meaning
2. **Seven tendencies** = comprehensive but tractable agent set
3. **Competition** = produces coherent personality from plurality
4. **Same structure** = applies to anyone (different allocations + observations)
5. **More useful** = than unstructured text at scale

If the bet pays off, this is a general architecture for modeling minds.

## What This Doesn't Model

- **Temporal dynamics**: How the person changes over time
- **Context switching**: Different "modes" in different situations
- **Emotional states**: Momentary feelings vs. stable tendencies
- **Social relationships**: How the person relates to specific others
- **Skills/Knowledge**: What they can do vs. what drives them

These could be extensions, but the core model focuses on **stable value structures**.

## Key Formulas

### Weight Propagation
```
net_score = direct_weight + sum(pro_children.score) - sum(con_children.score)
```

### Allocation Rebalancing
```python
target_allocation[t] = score[t] / sum(scores)
new_allocation[t] = current[t] + (target[t] - current[t]) * learning_rate
```

### Validation Accuracy
```
accuracy = correct_predictions / total_predictions
baseline = 1 / num_tendencies  # Random chance
p_value = binomial_test(correct, total, baseline)
```

## Terminology

| Term | Definition |
|------|------------|
| **Observation** | Atomic fact about a person (~280 bytes) |
| **Agent** | Human tendency with token allocation |
| **Tendency** | One of 7 universal human drives |
| **Allocation** | Agent's share of influence (0.0-1.0) |
| **Tree** | Value hierarchy with root claim |
| **Claim** | Statement proposed by an agent |
| **Node** | Observation positioned in a tree |
| **Position** | PRO or CON relative to parent |
| **Stake** | Weight an agent assigns to a node |
| **Arena** | Where adversarial competition happens |
| **Epoch** | One full round of debate |
| **Convergence** | When allocations stabilize |
