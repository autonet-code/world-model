# Substrate architecture (post-and-coparent)

This document describes the world-model substrate as it stands after
the post-and-coparent refactor. It complements the older
`architecture.md` (which describes the original 7-tendencies arena);
the substrate here is the generalized, deployer-configurable layer
that autonet rides on.

## Mental model in one paragraph

A deployer picks a small set of *roots* — utility primitives for
their domain (e.g. for code work: correctness, simplicity, idiom).
Each root is a tendency: a thesis with an anchor and a polarity axis
in coordinate space. As observations arrive, every tendency posts
unit-weight stakes on sub-claims that emerge in its tree. When a
sub-claim's coordinate falls within another tendency's locality
bandwidth, it acquires a parent edge into that tendency's tree
automatically — and a node with parents in multiple trees is what
we call a *work item*. There is no explicit "work item" type; the
shape is emergent. Per-node persistent novelty `n` evolves under a
continuous two-way dynamic (decays on PRO, regrows on CON, drifts
toward uncertainty when nothing happens), and per-root scores can
optionally be evolved by a Lindblad master-equation kernel that
captures the resist-then-yield-decisively cognitive shape.

## Key concepts

### Posts, not weighted stakes

Every `Stake` has weight=1. The schema preserves the field for
forward compatibility, but the dynamics no longer modulate it.
Magnitude lives entirely in (a) the count of posts on a node and
(b) the per-node `n` value used by mint and pruning. Helper:
`Node.add_post(agent_id)`.

### Multi-parent nodes (work items emerge)

A `Node` carries `parents: list[ParentLink]`, where each link is
`(parent_id, position, tendency_id)`. Single-parent nodes are
length-1 lists (the common case); multi-parent nodes are work items
bridging multiple tendencies. The properties `node.parent_id` and
`node.position` expose the first parent for backward-compat readers.

### Content-addressed identity over coordinates

`sprout_child` hashes a new node id over `{anchor, axis}` only —
parent context does not participate. Two solvers that propose the
same coordinate-anchored claim under different parents produce the
same node id, and the merge step accumulates parent edges on the
shared node. This is what makes federation natural: work items grow
their parent set as multiple tendencies (or solvers) reference them.

### Cross-tendency edge discovery at sprout time

When `sprout_child` is called with a `world` argument, it walks every
other tendency and appends a parent edge if the new node's anchor
sits within that tendency's `bandwidth * 1.5`. Position at the new
edge is determined by the sign of the dot product between the anchor
and the other tendency's polarity axis. This replaces the older
`_sub_claim_staking` mechanism (which propagated weight between
neighboring nodes); now the relationship is structural — either a
node is part of a tendency's tree or it isn't.

### Score: intrinsic vs. tendency-tree

Two distinct readouts:

  - `intrinsic_score(node)`: how strongly this node is supported by
    its full subtree, regardless of which tendency reads it. Walks
    every child edge across all tendencies that share the node.

        intrinsic_score(N) = len(N.stakes)
                           + Σ intrinsic_score(c)  for c PRO of N (any tendency)
                           - Σ intrinsic_score(c)  for c CON of N (any tendency)

  - Tendency-tree score: walking tendency T's tree top-down, each
    parent->child edge contributes `±intrinsic_score(child)` signed
    by the polarity at the edge in T. A co-parented child can
    contribute positively to one tree and negatively to another.

### Persistent novelty `n` (continuous two-way)

Each node has `n ∈ [0, 1]`. The discrete update per round:

    dn/dt = -γ_pro · n · pro_rate
          + γ_con · (1-n) · con_rate
          + ε · (1-n)

  - `pro_rate` = count of PRO posts received this round.
  - `con_rate` = ∑ max(0, c.net_score) over CON children of this node.
  - `γ_pro > γ_con` (default 1.0 vs 0.5): PRO confirmation reduces
    surprise faster than CON contradiction restores it.
  - `ε` (default 0.01): drift toward uncertainty under quiet rounds.

These rate constants are per-tendency dataclass fields, so a deployer
can give the correctness root a higher `γ_con` to make CON evidence
settle fast.

### Two equilibrate kernels

  - `equilibrate(world)`: discrete per-round (act → apply_stakes →
    update_novelty). Each round, every tendency walks observations,
    forms intents (signed unit values), and apply_stakes writes
    posts for the positive intents.
  - `equilibrate_continuous(world)`: integrates the Lindblad master
    equation forward in time, capturing resist-then-yield-decisively
    score dynamics + tilted steady states + cross-root coupling
    via a substrate Hamiltonian + locality-kernel jump operators.
    Updates per-root scores in place.

The discrete kernel handles graph-structure changes (sprouts,
prunes, capacity); the continuous kernel only updates per-root
scores and is intended to run *between* observation events.

### Correctness as veto

A tendency tagged `veto_shaped=True` carries hard-veto semantics. The
deployer pairs this with a higher `γ_con` for fast settling of CON
evidence. The asymmetric pruning helper `prune_veto_negatives(world)`
drops any direct child of a veto-shaped root whose `intrinsic_score`
falls below `tendency.veto_score_floor` — regardless of `n`,
regardless of settled-quiet history. This is how "correctness fails
this work item" propagates as a veto without giving correctness an
exotic gate mechanism.

### Federation merge

Solver-side, each solver runs its own substrate and emits an event
stream (`SubClaimSprouted`, `ObservationAdded`). On the aggregator,
events are concatenated, sorted by `(author_agent, seq)` for
determinism, and replayed onto the live world. Because identity is
coordinate-only, the same claim from multiple solvers consolidates
into one node; the multi-parent `parents` field on
`SubClaimSprouted` accumulates as edges in the live world.

## What stays unchanged

  - Tree topology APIs (creation, traversal): extended, not replaced.
  - Locator (Keyword, Coordinate, Chain): operates on coordinates,
    doesn't care about stake weights.
  - Render: operates on node content + structure, doesn't care about
    weights.
  - Locality kernel: same Gaussian on coordinates.
  - Charter tendencies and their setup.

## What's intentionally deferred

  - **Smooth-promotion-to-root**: a sub-claim accumulating enough
    structure to graduate into a root. The mechanism for utility
    primitives to drift over time. Likely paired with governance/
    voting on root composition.
  - **Delegation as a substrate primitive**: agents directing other
    agents' attention without manually re-posting.

These are not blocked by the post-and-coparent shape — they layer on
top once the deployer has reason to want them.

## Pointers

  - Engine schema: `world_model/models/tree.py` (Node, ParentLink,
    Stake, Tree).
  - Engine dynamics: `world_model/generalized/tendency.py` (sprout,
    act, edge discovery, intrinsic_score, update_novelty),
    `world_model/generalized/world.py` (apply_stakes, intrinsic_score,
    root_scores), `world_model/generalized/equilibrate.py` (discrete
    + continuous kernels), `world_model/generalized/lindblad.py`
    (Lindblad kernel internals).
  - Pruning: `world_model/generalized/prune.py`
    (`prune_settled_negatives`, `prune_veto_negatives`).
  - Tests: `test_content_addressed_ids.py`, `test_coparenting.py`,
    `test_federation_parent_merge.py`, `test_correctness_veto.py`,
    `test_lindblad_equilibrate.py`, `test_reseed.py`.
