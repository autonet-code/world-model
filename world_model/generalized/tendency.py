"""GeneralizedTendency: thesis + budget + frame + tree + behaviour.

A tendency in the generalized model is a *thesis about its world* and
the disposition to keep that thesis satisfied. Concretely:

  - It owns a tree (from world_model.models.tree.Tree). The root of the
    tree is the tendency's foundational claim.
  - It owns a CoordinateFrame whose claims correspond to the nodes of
    its tree (one CoordinateClaim per Node, at the same depth, with
    the node's stake aggregated as the claim's stake).
  - It has a budget (real-valued) that it can spend staking on its own
    tree (defense) or on other tendencies' trees (support if PRO,
    attack if CON).
  - It has a CoordinateProbe to evaluate novelty of incoming
    observations and of other tendencies' nodes.

When asked to act, a tendency:
  1. Runs the novelty probe on each incoming observation against its
     own frame.
  2. Based on the termination, stakes on the appropriate node:
       - INTEGRATED    -> stake PRO on the related claim's node, absorb
       - CONTRADICTS   -> stake CON on the contradicted node
       - DISRUPTS      -> stake CON heavily; consider sprouting child
       - ORTHOGONAL    -> ignore (out of this tendency's scope)
       - MAX_ITERATIONS -> ignore for now
  3. For every other tendency in the world, runs the probe on each of
     that tendency's nodes against its own frame. Same staking logic
     but cross-staked into the other's tree.

The tendency doesn't reason about budget allocation between defense
and cross-staking explicitly; the novelty probe's output (which
tendency's tree it's evaluating against) determines where stake
lands. Budget is a hard cap: if the sum of intended stakes exceeds
budget, they're scaled down proportionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple
import math

from ..models.tree import Node, Position, Stake, Tree
from ..novelty.core import Stance, Termination
from .coordinate_frame import (
    CoordinateClaim,
    CoordinateFrame,
    CoordinateProbe,
    _coords_of,
)
from .observation import Observation


# ---------------------------------------------------------------------------
# Tendency
# ---------------------------------------------------------------------------


@dataclass
class GeneralizedTendency:
    """A thesis owning a tree, a frame, a budget, and a probe."""

    id: str
    thesis: str
    anchor: Tuple[float, ...]                # location of root claim in space
    polarity_axis: Tuple[float, ...]         # unit vector for PRO direction
    budget: float = 1.0
    bandwidth: float = 1.0                   # frame's similarity bandwidth

    # Internal state -- built in __post_init__
    tree: Tree = field(init=False)
    frame: CoordinateFrame = field(init=False)
    probe: CoordinateProbe = field(init=False)
    _root_claim: CoordinateClaim = field(init=False)
    _node_to_claim: dict[str, CoordinateClaim] = field(init=False, default_factory=dict)

    # Last computed stakes, keyed by (other_tendency_id, node_id) -> signed stake.
    # Negative = CON, positive = PRO. Used by World to apply stakes.
    last_stakes: dict[tuple[str, str], float] = field(default_factory=dict)

    # Observation ids that have already been positioned as CON children
    # in our tree. Prevents re-sprouting deeper descendants on every
    # round; the CON-position is the persistent record of "this
    # contradicts us."
    _con_positioned: set = field(default_factory=set)

    def __post_init__(self):
        # Create root claim and tree
        self._root_claim = CoordinateClaim(
            content=self.thesis,
            depth=0,
            stake=1.0,
            anchor=self.anchor,
            polarity_axis=self.polarity_axis,
        )
        self.tree = Tree(root_value=self.thesis)
        # Bind root node's content to the thesis label (already done by Tree)
        # and link it in the index
        self._node_to_claim[self.tree.root_node.id] = self._root_claim
        # Build frame from claim hierarchy
        self.frame = CoordinateFrame(
            claims=[self._root_claim],
            integrated={},
            bandwidth=self.bandwidth,
        )
        self.probe = CoordinateProbe(max_iterations=8, disruption_threshold=0.6)

    # ----- Tree growth -----

    def sprout_child(
        self,
        parent_node_id: str,
        position: Position,
        anchor: Tuple[float, ...],
        polarity_axis: Tuple[float, ...],
        observation: Optional[Observation] = None,
        content: str = "",
    ) -> Node:
        """Add a child node + matching claim under an existing node.

        The child's polarity_axis defaults to the parent's if none
        provided. Stake starts at 0 and grows as tendencies stake on
        the child.
        """
        parent_node = self.tree.get_node(parent_node_id)
        if parent_node is None:
            raise ValueError(f"unknown node: {parent_node_id}")
        parent_claim = self._node_to_claim[parent_node_id]

        if not polarity_axis:
            polarity_axis = parent_claim.polarity_axis

        # New node in the tree
        new_node = Node(
            observation_id=observation.id if observation else None,
            tree_id=self.tree.id,
            position=position,
            content=content,
        )
        self.tree.add_node(parent_node_id, new_node, position)

        # Matching claim under the parent claim
        new_claim = CoordinateClaim(
            content=content or (observation.label if observation else ""),
            depth=parent_claim.depth + 1,
            stake=0.0,
            anchor=anchor,
            polarity_axis=polarity_axis,
        )
        parent_claim.children.append(new_claim)
        self._node_to_claim[new_node.id] = new_claim

        # Refresh frame's claim list (frame is recreated to keep it
        # internally consistent; integrated set is preserved)
        self._refresh_frame()
        return new_node

    def _refresh_frame(self) -> None:
        self.frame = CoordinateFrame(
            claims=[self._root_claim],
            integrated=dict(self.frame.integrated),
            bandwidth=self.bandwidth,
            topic_threshold=self.frame.topic_threshold,
            pro_threshold=self.frame.pro_threshold,
            contain_distance=self.frame.contain_distance,
        )

    # ----- Action -----

    def evaluate(
        self,
        observation: Observation,
    ) -> Tuple[Termination, float, Optional[CoordinateClaim]]:
        """Run the novelty probe on an observation against this
        tendency's frame. Returns (termination, composite_novelty,
        most-related-claim).
        """
        result = self.probe.measure(observation, self.frame)
        # Find best-matching claim from the frame
        related = self.frame.find_claims(observation)
        best = related[0][0] if related else None
        return result.termination, result.composite, best  # type: ignore[return-value]

    def act(self, world: "World") -> None:
        """Compute and record this tendency's stakes for the current
        observation set and world state.

        For each observation:
          - Evaluate stance through the probe.
          - INTEGRATED with a PRO stance against the most-related claim
            -> sprout a PRO child observation node under that claim
            (or stake PRO if the node already exists), absorb the obs.
          - CONTRADICTS / DISRUPTS -> sprout a CON child observation
            node under the contradicted claim (drags its score down).
          - ORTHOGONAL / MAX_ITER -> ignore.

        Joint satisfaction discount: an observation that is *already*
        satisfied through coordinates of OTHER variables (variables
        resolved with high confidence by independent evidence) gets
        its contribution to this tendency discounted. This breaks the
        symmetric-tie problem in chained implications: an implication
        c_imp2 = (y, z) at (0, -1, +1) supports both y=F AND z=T;
        once z=T is independently established, the obs's support for
        y=F should diminish because the implication is satisfied
        without needing y=F.

        For each foreign node, evaluate its anchor; cross-stake by sign
        of stance.

        Stakes are stored in self.last_stakes; the World's apply_stakes
        writes Stake objects onto the actual nodes. Budget is enforced
        as a hard cap.
        """
        intents: dict[tuple[str, str], float] = {}

        # Compute joint-satisfaction scores: for each obs, how much is
        # it already satisfied by OTHER tendencies' resolved positions?
        # Used as a discount factor on this tendency's contribution.
        sat_scores = self._joint_satisfaction(world)

        # 1. Stake on / sprout from incoming observations against own tree
        for obs in world.observations.values():
            # If we've already CON-positioned this obs in a prior round,
            # don't re-evaluate -- just re-stake the existing CON child
            # at standard magnitude. Prevents runaway descendant growth.
            if obs.id in self._con_positioned:
                existing_id = self._find_existing_obs_child(obs.id, position=Position.CON)
                if existing_id is not None:
                    intents[(self.id, existing_id)] = intents.get((self.id, existing_id), 0.0) + 0.5
                    continue
            term, novelty, claim = self.evaluate(obs)
            if claim is None:
                continue
            parent_node_id = self._claim_to_node_id(claim)
            if parent_node_id is None:
                continue
            # Discount this obs's contribution by how satisfied it
            # already is through OTHER variables. obs.id -> [0, 1].
            # 1 = fully satisfied elsewhere -> contribution * 0.
            # 0 = not satisfied elsewhere -> full contribution.
            discount = 1.0 - sat_scores.get(obs.id, 0.0)
            mag = max(0.05, min(1.0, novelty)) * discount

            if term == Termination.INTEGRATED:
                # Find or create a PRO child observation node under the claim
                child_id = self._ensure_obs_child(parent_node_id, obs, position=Position.PRO)
                intents[(self.id, child_id)] = intents.get((self.id, child_id), 0.0) + mag
                # Absorb only on PRO -- this obs *fits* our worldview.
                self.frame = self.frame.absorb(obs)

            elif term in (Termination.CONTRADICTS_ROOT, Termination.DISRUPTS):
                # Sprout a CON child under the contradicted claim
                child_id = self._ensure_obs_child(parent_node_id, obs, position=Position.CON)
                # Stake on the CON node positively (we are *backing* the
                # CON observation -- it really does contradict our claim).
                # Because CON children subtract from parent score, the
                # net effect is to drag the parent's net_score down.
                intents[(self.id, child_id)] = intents.get((self.id, child_id), 0.0) + mag
                # Track that we've already positioned this obs as CON
                # (so we don't keep sprouting deeper CON descendants),
                # but DO NOT absorb -- absorbed = fits our worldview.
                self._con_positioned.add(obs.id)

        # 2. Cross-stake on every other tendency's nodes.
        # For each foreign node, evaluate either:
        #   (a) the observation it links to (if any), or
        #   (b) a synthetic Observation at the node's claim anchor.
        # Both routes let cross-staking land on roots and abstract nodes,
        # not just observation-linked leaves.
        for other in world.tendencies.values():
            if other.id == self.id:
                continue
            for node in other.tree.all_nodes():
                # Resolve a probe target for this node
                if node.observation_id is not None:
                    target = world.observations.get(node.observation_id)
                    if target is None:
                        continue
                else:
                    foreign_claim = other._node_to_claim.get(node.id)
                    if foreign_claim is None or not foreign_claim.anchor:
                        continue
                    target = Observation(
                        id=f"_anchor_{other.id}_{node.id}",
                        coords=foreign_claim.anchor,
                        label=f"anchor:{foreign_claim.content}",
                    )
                term, novelty, _claim = self.evaluate(target)
                signed = self._cross_stake_sign(term, novelty)
                if signed != 0.0:
                    key = (other.id, node.id)
                    intents[key] = intents.get(key, 0.0) + signed

        # 3. Apply budget as a multiplier (not a cap). Each intent
        # already encodes magnitude per observation; budget scales the
        # tendency's overall influence. Equivalent to "allocation" in
        # the personality model.
        intents = {k: v * self.budget for k, v in intents.items()}
        self.last_stakes = intents

    def _joint_satisfaction(self, world: "World") -> dict[str, float]:
        """For each observation in the world, compute how much it is
        already satisfied by OTHER tendencies' resolved positions.

        For an obs at coords (c_1, ..., c_d), we look at each
        dimension i where:
          - This tendency's anchor has component near 0 (this dim is
            not THIS tendency's responsibility).
          - Some other tendency's anchor has component sharing sign
            with c_i (that tendency aligns with this dim of obs).
          - That other tendency has a strong root score.

        The score is max over satisfied dimensions of (alignment ×
        normalized score). Returns {obs_id: [0, 1]}.

        Variables that THIS tendency is responsible for (its anchor
        nonzero on that dim) don't count toward satisfaction --
        otherwise we'd self-discount.
        """
        result: dict[str, float] = {}
        if not self.anchor:
            return result
        my_dims = [i for i, a in enumerate(self.anchor) if abs(a) > 0.01]
        my_dim_set = set(my_dims)

        # Build a per-dim, per-sign GAP score: how decisively does the
        # winning tendency on (dim, sign) beat its opposite-sign rival?
        # We only count satisfaction when there's a clear winner --
        # otherwise the dim is itself contested and shouldn't satisfy
        # other obs.
        all_scores = world.root_scores()
        # Group tendencies by (dim, sign) of their anchor
        by_key: dict[tuple[int, int], list[tuple[str, float]]] = {}
        for tid, t in world.tendencies.items():
            if not t.anchor:
                continue
            for i, a in enumerate(t.anchor):
                if abs(a) < 0.01:
                    continue
                sign = 1 if a > 0 else -1
                by_key.setdefault((i, sign), []).append((tid, all_scores.get(tid, 0.0)))

        # For each dim, compute the gap between winning sign and losing sign.
        # gap_score(dim, sign) = max(0, score_at_sign - score_at_-sign).
        gap_score: dict[tuple[int, int], float] = {}
        all_dims = {k[0] for k in by_key}
        for dim in all_dims:
            pos = max((s for _, s in by_key.get((dim, +1), [])), default=0.0)
            neg = max((s for _, s in by_key.get((dim, -1), [])), default=0.0)
            if pos > neg:
                gap_score[(dim, +1)] = pos - neg
                gap_score[(dim, -1)] = 0.0
            else:
                gap_score[(dim, -1)] = neg - pos
                gap_score[(dim, +1)] = 0.0

        max_gap = max(gap_score.values()) if gap_score else 0.0
        if max_gap <= 0:
            return result

        for obs in world.observations.values():
            if not obs.coords:
                continue
            sat = 0.0
            for i, c in enumerate(obs.coords):
                if abs(c) < 0.01:
                    continue
                if i in my_dim_set:
                    continue   # don't self-discount on our own axis
                sign = 1 if c > 0 else -1
                key = (i, sign)
                if key in gap_score:
                    contribution = gap_score[key] / max_gap
                    if contribution > sat:
                        sat = contribution
            result[obs.id] = sat
        return result

    def _find_existing_obs_child(
        self,
        observation_id: str,
        position: Position,
    ) -> Optional[str]:
        """Search the whole tree for a node with the given observation_id
        and position. Returns its node id or None.
        """
        for node in self.tree.all_nodes():
            if node.observation_id == observation_id and node.position == position:
                return node.id
        return None

    def _ensure_obs_child(
        self,
        parent_node_id: str,
        observation: Observation,
        position: Position,
    ) -> str:
        """Find an existing node in the tree linked to this observation
        with the given position, or create one as a child of parent.
        Returns the node id.

        Tree-wide search prevents duplicate sprouts when equilibration
        re-evaluates the same observation through different paths.
        """
        existing = self._find_existing_obs_child(observation.id, position)
        if existing is not None:
            return existing
        parent_claim = self._node_to_claim[parent_node_id]
        anchor = observation.coords
        polarity = parent_claim.polarity_axis
        if position == Position.CON:
            polarity = tuple(-u for u in polarity)
        new_node = self.sprout_child(
            parent_node_id=parent_node_id,
            position=position,
            anchor=anchor,
            polarity_axis=polarity,
            observation=observation,
            content=observation.label or f"obs:{observation.id[:6]}",
        )
        return new_node.id

    # ----- Mapping helpers -----

    def _claim_to_node_id(self, claim: CoordinateClaim) -> Optional[str]:
        for node_id, c in self._node_to_claim.items():
            if c is claim:
                return node_id
        return None

    def _termination_to_signed_stake(self, term: Termination, novelty: float) -> float:
        """Translate a termination + novelty into a signed stake on
        this tendency's *own* tree. Defensive: PRO when integrating,
        CON when contradicting.
        """
        # Normalize novelty into a magnitude in [0.05, 1.0] so that
        # something always lands when on-topic.
        mag = max(0.05, min(1.0, novelty))
        if term == Termination.INTEGRATED:
            return +mag
        if term == Termination.CONTRADICTS_ROOT:
            # The observation contradicts our thesis -- we still need
            # to defend our claim, so stake PRO on the root with high
            # weight (the conflict pulls us to defend).
            return +mag * 1.5
        if term == Termination.DISRUPTS:
            return +mag * 2.0
        # ORTHOGONAL or MAX_ITERATIONS -- not our concern
        return 0.0

    def _cross_stake_sign(self, term: Termination, novelty: float) -> float:
        """Translate a termination + novelty into a signed stake on
        *another* tendency's tree.

        If their node was absorbed cleanly into our frame (INTEGRATED),
        we stake PRO -- we agree with their evidence.
        If their node contradicts our root or disrupts our allocation,
        we stake CON -- we attack.
        """
        mag = max(0.05, min(1.0, novelty))
        if term == Termination.INTEGRATED:
            return +mag
        if term in (Termination.CONTRADICTS_ROOT, Termination.DISRUPTS):
            return -mag
        return 0.0

    def __repr__(self) -> str:
        return (
            f"GeneralizedTendency(id={self.id!r}, thesis={self.thesis!r}, "
            f"budget={self.budget:.2f}, nodes={len(self._node_to_claim)})"
        )
