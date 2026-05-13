# Implementation plan: bring substrate up to date with brainstormed architecture

Goal: implement the post-and-coparent refactor plus correctness-as-
veto dynamics so that subsequent experiments (Tier 0 and beyond) run
on architecture that matches the cognitive model we agreed on.

Spec: `REFACTOR_POST_AND_COPARENT.md` in this directory.

## What's in scope

1. Post-only data model: `Stake.weight = 1` always; field kept for
   forward compat.
2. Multi-parent nodes: `Node.parents = list[(parent_id, position,
   tendency_id)]`. Work items emerge from multi-parenthood without an
   explicit type.
3. Content-addressed hash drops parent. ID is over `{anchor, axis}`.
   Federation merge accumulates parent edges on existing nodes.
4. `intrinsic_score(node)` walks all child edges (across tendencies),
   computed once. Tendency-tree score signs by per-edge polarity in
   that tendency.
5. `_sub_claim_staking` replaced by edge-discovery in `sprout_child`.
   No more weight propagation.
6. Per-tendency rate-constant overrides on `update_novelty` so we
   can give correctness higher γ_con.
7. Asymmetric pruning: subtree pruned when its correctness-root
   intrinsic_score drops below a hard threshold regardless of n.
8. Documentation: substrate-architecture overview naming the agreed-
   upon concepts (roots as utility primitives, work items as emergent,
   correctness-as-veto, etc.).

## What's NOT in scope

- Smooth-promotion-to-root (subclaim → root via accumulated structure).
- Governance / voting on root promotion.
- Delegation as a first-class primitive.
- Tier 0 experiment build itself (separate, after this lands).

## Branch layout

`feature/post-and-coparent` on three repos:

- `c:\code\world-model` — branched off `feature/lindblad-equilibrate`
  (we keep the Lindblad kernel; this builds on top).
- `c:\code\autonet` — branched off `world-model-substrate`.
- `D:\videos\SF` — branched off `feature/lindblad-equilibrate`.

`BRANCHES.md` updated to register the new feature.

## Implementation order (bottom-up)

### Step 1 — `tree.py` (Node schema)

- Add `Node.parents: list[ParentLink]` where `ParentLink = NamedTuple(parent_id, position, tendency_id)`.
- Keep `Node.parent_id` as a property reading `parents[0].parent_id` (backward compat).
- Keep `Node.position` as a property reading `parents[0].position`.
- Update `to_dict` / `from_dict` to serialize the parents list.
- `Stake.weight` default = 1.0 explicit.
- `Node.add_post(agent_id)` helper that calls `add_stake(agent_id, weight=1.0)`.
- Update `direct_weight` to be count-based (sum still works since weights are 1).

### Step 2 — `tendency.py` (sprout + act)

- `sprout_child` writes parents list with the single creating-tendency
  link, then walks other tendencies and adds edges where coordinate
  distance is within `bandwidth * 1.5`.
- `act` posts unit-weight stakes (no `mag = max(..., novelty * discount)`
  calculation).
- `_update_capacities` reads count-of-positive-posts.
- `_sub_claim_staking` removed (its locality job moved into `sprout_child`).
- `update_novelty` accepts per-tendency overrides for γ_pro, γ_con, ε.
- Add `intrinsic_score(node)` walking all child edges.

### Step 3 — `world.py` (apply + expose)

- `apply_stakes` works against unit weights.
- Expose `intrinsic_score(node)` at world level.
- `total_stake_on(...)` returns count.
- Score readers use intrinsic_score signed by per-edge polarity for
  tendency-tree readouts.

### Step 4 — `equilibrate.py` (both kernels)

- `_extract_subclaims` reads count-based stakes.
- Discrete `equilibrate` per-round flow unchanged structurally.
- Continuous `equilibrate_continuous` keeps working — `_direction`,
  `_normalized_tension` already operate on (stake, capacity, coords)
  triples; values become count-based naturally.

### Step 5 — `events.py` and `aggregate.py` (autonet adapter)

- `SubClaimSprouted` carries parent links list (was single parent_id).
- `apply_events` checks for existing node by content-addressed id;
  if exists, append new parent links rather than create duplicate.
- Backward compat: a single-parent event materializes as a length-1
  parents list.

### Step 6 — `reconcile.py` (autonet)

- Already reads `node.n`; survives untouched.
- `_node_mint` keeps current shape; n already does the work.
- Verify per-agent attribution still works when events carry parent
  lists.

### Step 7 — Correctness-as-veto dynamics

- Add per-root rate-constant config (probably `Tendency.novelty_gamma_*`
  already exists; expose so adapter can set higher γ_con on the
  correctness root).
- Add asymmetric pruning: in `prune.py` (or a new helper), prune
  subtrees rooted under a correctness-root child if intrinsic_score
  drops below a configurable hard threshold.
- Document the convention: substrate deployers tag one root as
  "veto-shaped" via a flag.

### Step 8 — Tests

- `test_content_addressed_ids.py`: update assertions for new hash
  shape (anchor+axis only).
- `test_reseed.py`: should still pass.
- `test_lindblad_equilibrate.py`: should still pass behaviorally.
- New: `test_coparenting.py` — verifies that a node sprouted near
  another tendency's locality region acquires a parent edge there
  automatically.
- New: `test_federation_parent_merge.py` — applies events with
  different parent links for the same coordinate-anchored node;
  verifies parent set accumulates.
- New: `test_correctness_veto.py` — verifies a CON observation under
  the correctness root prunes the subtree even if other roots score
  positive.

### Step 9 — Documentation

- New: `c:\code\world-model\docs\architecture.md` — overview of the
  substrate concepts: utility-primitive roots, posts, work items as
  emergent multi-parented nodes, novelty as continuous coherence,
  correctness-as-veto, the two equilibrate kernels.
- Update existing module docstrings as needed where behavior changed.
- README updates if relevant.

### Step 10 — Commits + branch close

- Commit per step on each repo.
- Update `BRANCHES.md` with final commit hashes.
- Optionally merge to default branch when satisfied (separate decision).

## Estimated effort

~2 days of focused work. Most time goes into Step 1-2 (touching every
call site that reads `parent_id` or `stakes`), Step 5 (events schema
change ripples through aggregate and applies), and Step 8 (tests).

## Pre-flight checks

Before starting:
- Confirm both repos clean on their starting branches.
- Confirm pre-existing tests pass (baseline).
- Take a tag/snapshot of current state in case we need to bail.

## Definition of done

- All steps above complete.
- All pre-existing tests pass.
- New tests for co-parenting, federation merge, correctness-veto
  pass.
- Documentation reflects the agreed architecture.
- BRANCHES.md updated.
- One end-to-end smoke run: build a small substrate with the new
  code, drive a few observations through it, confirm `n` per node
  evolves and a co-parented node forms when expected.

## After this lands

Tier 0 build can begin on top of the updated architecture. The
running plan for that lives separately and isn't part of this plan.
