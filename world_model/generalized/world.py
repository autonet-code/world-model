"""World: coalition of tendencies, observation stream, stake graph.

A World holds:
  - tendencies: id -> GeneralizedTendency
  - observations: id -> Observation (the current observation set)
  - history: cumulative observations seen so far (for novelty floors)

The World provides the operations that actually mutate state:
  - add_observation: record a fact arriving from outside
  - apply_stakes: walk every tendency's last_stakes and write Stake
    objects onto the corresponding nodes (own + cross-staked)

Cross-tendency staking is what creates child nodes in other tendencies'
trees. When tendency A stakes CON on a node owned by tendency B, that
stake is recorded on B's node directly. When the staking pressure on
some position becomes contested enough (multiple tendencies disagree
strongly), the growth rule (in grow.py) sprouts new children to
accommodate the dispute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from ..models.tree import Node, Position, Stake
from .observation import Observation


@dataclass
class World:
    """A coalition of tendencies and the observation stream they share."""

    tendencies: Dict[str, "GeneralizedTendency"] = field(default_factory=dict)  # type: ignore[name-defined]
    observations: Dict[str, Observation] = field(default_factory=dict)
    history: List[Observation] = field(default_factory=list)

    def add_tendency(self, tendency: "GeneralizedTendency") -> None:  # type: ignore[name-defined]
        self.tendencies[tendency.id] = tendency

    def add_observation(self, obs: Observation) -> None:
        self.observations[obs.id] = obs
        self.history.append(obs)

    def clear_observations(self) -> None:
        self.observations.clear()

    def apply_stakes(self) -> None:
        """Walk every tendency's last_stakes and write Stakes onto the
        relevant node. Existing stakes from this tendency are removed
        first to avoid double-counting across rounds.
        """
        # 1. Remove existing stakes attributed to any tendency in self.tendencies
        all_tendency_ids = set(self.tendencies.keys())
        for tendency in self.tendencies.values():
            for node in tendency.tree.all_nodes():
                node.stakes = [s for s in node.stakes if s.agent_id not in all_tendency_ids]
                node.invalidate_cache()

        # 2. Apply each tendency's recorded intents
        for tendency in self.tendencies.values():
            for (target_tendency_id, node_id), signed in tendency.last_stakes.items():
                target = self.tendencies.get(target_tendency_id)
                if target is None:
                    continue
                node = target.tree.get_node(node_id)
                if node is None:
                    continue
                node.add_stake(agent_id=tendency.id, weight=signed)

    def total_stake_on(self, target_tendency_id: str, node_id: str) -> float:
        """Net stake on a node (sum of signed stakes from all
        tendencies). Positive = supported, negative = undermined.
        """
        target = self.tendencies.get(target_tendency_id)
        if target is None:
            return 0.0
        node = target.tree.get_node(node_id)
        if node is None:
            return 0.0
        return sum(s.weight for s in node.stakes)

    def root_scores(self) -> Dict[str, float]:
        """Net score of each tendency's thesis."""
        return {tid: t.tree.score for tid, t in self.tendencies.items()}
