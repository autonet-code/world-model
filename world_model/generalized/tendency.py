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

        Stakes are stored in self.last_stakes; the World's apply_stakes
        method walks them and writes Stake objects onto the actual
        nodes (own and others'). Budget is enforced as a hard cap.
        """
        intents: dict[tuple[str, str], float] = {}

        # 1. Stake on incoming observations against own tree
        for obs in world.observations.values():
            term, novelty, claim = self.evaluate(obs)
            if claim is None:
                continue
            node_id = self._claim_to_node_id(claim)
            if node_id is None:
                continue
            signed = self._termination_to_signed_stake(term, novelty)
            if signed != 0.0:
                intents[(self.id, node_id)] = intents.get((self.id, node_id), 0.0) + signed
            # Absorb if we PRO-staked it
            if signed > 0:
                self.frame = self.frame.absorb(obs)

        # 2. Cross-stake on every other tendency's nodes
        for other in world.tendencies.values():
            if other.id == self.id:
                continue
            for node in other.tree.all_nodes():
                if node.observation_id is None:
                    continue
                obs = world.observations.get(node.observation_id)
                if obs is None:
                    continue
                term, novelty, _claim = self.evaluate(obs)
                signed = self._cross_stake_sign(term, novelty)
                if signed != 0.0:
                    key = (other.id, node.id)
                    intents[key] = intents.get(key, 0.0) + signed

        # 3. Enforce budget: scale to total |intent| <= budget
        total = sum(abs(v) for v in intents.values())
        if total > self.budget and total > 0:
            scale = self.budget / total
            intents = {k: v * scale for k, v in intents.items()}

        self.last_stakes = intents

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
            return +mag * 0.5
        if term in (Termination.CONTRADICTS_ROOT, Termination.DISRUPTS):
            return -mag
        return 0.0

    def __repr__(self) -> str:
        return (
            f"GeneralizedTendency(id={self.id!r}, thesis={self.thesis!r}, "
            f"budget={self.budget:.2f}, nodes={len(self._node_to_claim)})"
        )
