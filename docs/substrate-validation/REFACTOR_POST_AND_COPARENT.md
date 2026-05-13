# Refactor spec: post-only substrate + co-parenting

Two coordinated changes to the substrate's data model and dynamics:

  1. Weighted stakes degenerate. Each post is weight=1 always. The
     `weight` field stays in the schema for forward-compatibility but
     is never set to anything other than 1.
  2. Nodes can have multiple parents across tendencies. A node that
     bridges roots is the emergent shape of a "work item" — no explicit
     type field is added.

Goal: a substrate where the architecture matches the cognitive model
("agents post nodes, position is the signal") and where cross-
tendency reference happens through structure rather than through
weight propagation.

## What changes in the data model

### Node

Today:
```python
parent_id: Optional[str]
position: Position
stakes: list[Stake]   # each Stake has agent_id + weight
```

After refactor:
```python
parents: list[ParentLink]    # one entry per (parent_node_id, position, tendency_id)
stakes: list[Stake]           # weight is always 1.0; field kept for future use
n: float                      # already exists, unchanged
```

A `ParentLink` is `(parent_node_id, position, tendency_id)`. The
position is per-edge: a node can be PRO of one parent and CON of
another, or PRO of both. The tendency_id is redundant in the
single-tendency case but explicit in the cross-tendency case so we
know which tree this edge participates in.

Backward-compatible read accessors:
  - `node.parent_id` returns `parents[0].parent_node_id` (the first
    parent, for callers that haven't been updated yet).
  - `node.position` returns `parents[0].position` (same).
  - These let existing call sites keep working during transition.

### Stake

```python
@dataclass
class Stake:
    agent_id: str
    weight: float = 1.0   # only ever 1.0; placeholder for future use
```

Helper: `Node.add_post(agent_id)` sets weight=1 explicitly. Existing
`add_stake` calls keep working but pass weight=1 silently.

### Score recursion

Score has two distinct concepts under co-parenting:

  - **`node.intrinsic_score`**: how strongly this node is supported
    by its full subtree, regardless of which tendency is reading.
    Computed by walking ALL child edges (across all tendencies):

    ```python
    intrinsic_score(node) = len(node.stakes)
                          + sum(intrinsic_score(c) for c in all_pro_children(node))
                          - sum(intrinsic_score(c) for c in all_con_children(node))
    ```

    where `all_pro_children`/`all_con_children` enumerate every edge
    where this node is the parent-side, regardless of (child, tendency)
    combination. The intrinsic_score is what the network has decided
    about this claim; it's the same number whichever tendency is
    looking at it.

  - **Tendency-tree score**: when computing a tendency T's root score,
    walk T's tree top-down. At each parent->child edge in T, the child
    contributes `+intrinsic_score(child)` if PRO of parent in T,
    `-intrinsic_score(child)` if CON of parent in T.

This means a co-parented node's *full subtree* propagates up both
parent edges (in both tendencies' trees), but signed differently
depending on the edge polarity in each tree. A node that's PRO of A
and CON of B still has the same intrinsic_score; the difference is
only in the sign at the A-edge vs the B-edge.

Cognitive check: if X = "Threnian foundries can produce resonant
bells," and X is PRO of A = "trade is competitive" but CON of B =
"Kvell's monopoly holds," then a sub-claim Y under X = "Threnian
foundries use traditional methods" *strengthens X*. Y participates
in X's intrinsic_score. That intrinsic_score then propagates up A's
tree with PRO sign (raising A's score) AND up B's tree with CON sign
(lowering B's score). One contribution, two signed effects. That
matches intuition.

### `_sub_claim_staking` is replaced by edge discovery

The current `_sub_claim_staking` propagates per-stake weight to
neighboring nodes via the locality rule. This was doing two
conflated jobs:

  (a) Cross-tendency influence: a sub-claim under A also affects
      neighboring nodes in B's tree.
  (b) Within-tendency reinforcement: a strongly-staked sub-claim
      reinforces its own tree's neighbors.

Under the refactor, (a) becomes structural: when a node N at
coordinates `c` exists, the substrate ensures N is co-parented into
any tendency whose anchor is within `bandwidth * 1.5` of `c`. The
parent edge is added at sprout time and re-checked on relevant
events; once added, the connection is permanent until N is pruned.

(b) goes away entirely. With unit weights and structural co-parenting,
nodes don't "post on neighbors." They're either part of the tree or
they're not. If a sub-claim's relevance to a neighbor is real, the
neighbor has its own posts; if not, the substrate shouldn't
manufacture cross-staking to simulate relevance.

The locality function moves into `sprout_child`:

```python
def sprout_child(parent_id, position, anchor, polarity_axis, ...):
    # ... existing tree mechanics ...
    new_node = Node(..., parents=[(parent_id, position, my_tendency_id)])

    # Cross-tendency edge discovery
    for other_tendency in world.tendencies.values():
        if other_tendency.id == my_tendency_id:
            continue
        d = euclidean(anchor, other_tendency.anchor)
        if d < bandwidth * 1.5:
            # Find the appropriate parent in other_tendency's tree
            # (the closest existing node, or the root)
            other_parent_id = other_tendency.find_nearest_node(anchor)
            other_position = polarity_match(anchor, other_tendency.polarity_axis)
            new_node.parents.append((other_parent_id, other_position, other_tendency.id))
    return new_node
```

The polarity at the new edge in the other tendency is determined by
projecting the anchor onto that tendency's polarity_axis, the same
logic that already determines stance in the existing probe.

## What changes in the dynamics

### `tendency.act`

The current `act` does several things; under the refactor:

- For each observation, evaluate against the frame, get
  (termination, novelty, claim).
- Sprout-or-find a child node for the observation under the relevant
  parent. Default position by termination (PRO if INTEGRATED, CON if
  CONTRADICTS_ROOT etc.).
- **Add a single post** (agent_id=tendency.id, weight=1) on that node.
  No magnitude calculation, no novelty modulation of the post weight.
- Optionally: if the node's coordinates put it within another
  tendency's locality bandwidth, add a parent edge to that tendency's
  tree.

The `mag = max(0.05, min(1.0, novelty)) * discount` logic disappears
from `act`. Magnitude lives entirely in the (a) count of posts, and
(b) the per-node `n` value used downstream by mint.

### `update_novelty` (no change in shape)

The dynamics still are:
```
dn/dt = -gamma_pro * n * pro_rate + gamma_con * (1-n) * con_rate + epsilon * (1-n)
```

Under the refactor:
- `pro_rate` for a node = count of PRO posts received this round (each
  worth 1).
- `con_rate` = sum over CON children's net_score_in_tree, capped at 0.

No change to the formula; just the definition of pro_rate becomes
count-based.

### Capacity update

Today: `cap_new = decay * cap_old + (1 - decay) * sum(positive stake weights)`

Under refactor: `cap_new = decay * cap_old + (1 - decay) * count_of_pro_posts`

Same formula, count-based input. Capacity becomes a smoothed count
of accumulated PRO posts, which is exactly what smooth-promotion was
trying to express.

### Mint

Already novelty-modulated by `n_node`. No change.

## What changes in federation

Content-addressed IDs change shape under co-parenting. Today the hash
is over `{parent_path, pos, anchor, axis}`, which means two solvers
proposing the same coordinate-anchored claim under different parents
get *different* node IDs — the very thing federation should
collapse.

Under the refactor, the hash is over `{anchor, axis}` only. The
coordinate is what makes "this claim" the same claim across solvers.
Parents become a *list of edges* on the node, accumulated over time
as different solvers post the claim under different parents:

- Solver 1 posts a node N with anchor c, parent p_a in tendency A.
  Live world stores N with `parents = [(p_a, PRO, A)]`.
- Solver 2 posts a node with the same anchor c, parent p_b in
  tendency B. Live world finds the existing N (same hash from
  anchor+axis) and *appends* `(p_b, PRO, B)` to its parents list.
- N is now co-parented; emergent work item without anyone declaring
  it as such.

This is a meaningful schema change to the events stream. A
`SubClaimSprouted` event today carries a single parent_id; under the
refactor it carries a parent-link tuple. Existing single-parent
events stay valid as a length-1 parents list.

Tree structure is preserved because each tendency tree still walks
its own edges. A node appearing in two tendencies' trees is just
the same node with two different parent links pointing into it.

## What stays unchanged

- Tree topology APIs (creation, traversal) — extended, not replaced.
- Locator (Keyword, Coordinate, Chain) — operates on coordinates,
  doesn't care about stake weights.
- Render — operates on node content + structure, doesn't care about
  weights.
- Locality kernel — same Gaussian on coordinates.
- Charter tendencies and their setup.
- The events schema (`SubClaimSprouted`, `ObservationAdded`,
  `apply_events`) — extended to carry parent lists where needed.

## Code surface

Estimated touch list:

  Engine (world-model):
    - `world_model/models/tree.py` — `Node`, `Tree`, `add_node`,
      `add_child`, `add_stake`. Most invasive changes here.
    - `world_model/generalized/tendency.py` — `act`, `_sub_claim_staking`,
      `_update_capacities`, `sprout_child`, `update_novelty`.
    - `world_model/generalized/world.py` — `apply_stakes`,
      `total_stake_on`, score readers.
    - `world_model/generalized/equilibrate.py` — `_extract_subclaims`
      now reads count-based stakes.
    - `world_model/generalized/lindblad.py` — `_direction`,
      `_normalized_tension` already work with stake-magnitudes; they
      just become count-based naturally.

  Adapter (autonet):
    - `nodes/common/world_model_substrate/adapter.py` — `train_world_model_on_task`,
      `compute_stake_delta` (already migrated to events; minimal change).
    - `nodes/common/world_model_substrate/aggregate.py` — `apply_events`
      handles parent lists.
    - `nodes/common/world_model_substrate/events.py` — event schema
      grows to carry parent lists.
    - `nodes/common/world_model_substrate/reconcile.py` — already uses
      `node.n`; survives the refactor.

  Tests (world-model):
    - `test_content_addressed_ids.py` — hash now over parent set;
      adjust assertions if needed.
    - `test_reseed.py` — should still pass.
    - `test_lindblad_equilibrate.py` — already structured around
      observation flow; should still pass.

## Open questions, with my leans

  - **Deduplication of posts**: should `(agent_id, node_id)` be a
    unique key, so a single agent can't post twice on the same node?
    My lean: yes, dedupe. Posting twice doesn't say anything new
    structurally; it's only meaningful if weights existed.

  - **Backward-compat parent_id**: keep the alias for one transition
    cycle, or break cleanly?
    My lean: keep the alias. Many call sites read `node.parent_id`;
    breaking them all simultaneously is a bigger change than needed.

  - **Removing or keeping `_sub_claim_staking`**: the locality logic
    needs to live somewhere. Either it stays as edge-discovery (find
    co-parents) or moves into `sprout_child` (when sprouting near
    another tendency's region, automatically add the cross edge).
    My lean: move into `sprout_child`. It's where new edges happen
    anyway.

  - **What happens to `Tendency.budget`**: today it's a float that
    multiplies stake intent magnitudes. Under unit weights, it
    becomes "max posts per round" (an integer cap on how many posts
    the tendency can place per equilibrate round).
    My lean: switch to integer post-budget.

  - **Score sign for co-parented nodes**: see "Score recursion"
    above. Intrinsic_score is computed once (walks all edges
    regardless of tendency); each tendency-tree score signs that
    intrinsic value by the edge polarity at the parent->child edge
    in that tree. One node, one intrinsic value, two signed
    contributions. Confirmed correct semantics above.

## Self-check on dynamics

Empty post on every node, all unit weights, no co-parenting:
  - net_score recursion produces same numbers as before with weights=1
    always. ✓
  - update_novelty math unchanged structurally; pro_rate now in units
    of "posts per round" rather than "stake weight per round." ✓
  - Capacity stays smooth-promotion-shaped, just count-based. ✓
  - Mint formula unchanged. ✓

Co-parented node:
  - Settle node X under tendency A by posting PRO. Score in A's tree
    moves up. ✓
  - X also has parent in B with PRO position. B's tree score also
    moves up by the same recursion. ✓
  - X has parent in B with CON position. B's tree score moves DOWN
    while A's moves UP. Same node, opposite contribution to two
    tendencies. ✓
  - This is the "work item bridges two trees" semantics.

Federation:
  - Solver 1 posts node X with parent-set {(p_a, PRO, A)}.
  - Solver 2 posts node X' with parent-set {(p_b, PRO, B)} where
    coords match.
  - Content-addressed ID is over coordinate (and axis), not parent
    set, so X.id == X'.id. Merge step unions the parent sets.
  - Result: one node X with parents in both A and B. Emerged a work
    item without anyone declaring it.

## What this gives us

  - Architecture matches the cognitive model. No more hybrid drift.
  - Cross-tendency connection is structural, not via weight
    propagation. The reshaped-A1 contamination disappears.
  - Federation grows work items naturally.
  - The Lindblad correspondence stays clean: omega and zeta are
    computed from counts, which read identically as "rate" in the
    quantum mapping (a per-unit-time count).
  - Mint focuses on novel synthesis automatically: synthesis nodes
    have more parents, more chances to participate in mint events,
    and `n` modulates them properly.

## Sequencing

Done as one feature on `feature/post-and-coparent` across
world-model and autonet. Not coupling to the existing
`feature/lindblad-equilibrate` branch — they're independent and can
land in either order. The lindblad branch's `equilibrate_continuous`
already reads stake magnitudes via `_direction` etc., which become
count-based naturally; that branch will need a small follow-up to
verify but no rework.

Estimated effort: 1-2 days of careful work. Most of the time goes
into:

  - Adjusting `Node` and verifying every call site that reads
    `node.parent_id` or `node.stakes` still works.
  - Updating `apply_events` to handle parent-list events.
  - Revising `_sub_claim_staking` to discover-edges rather than
    stake-on-neighbors.
  - Re-running existing tests; expecting `test_content_addressed_ids`
    to need updates because hash payload may change (if we include
    parent set vs. only coords).
