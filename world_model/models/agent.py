"""
Agent model - human tendencies that stake on nodes in value trees.

Each agent represents a drive/tendency present in every human.
Allocations determine relative influence - sum to 1.0 across all agents.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Tendency(Enum):
    """Core human tendencies - the agents that stake on nodes."""

    SURVIVAL = "survival"           # Physical safety, resource acquisition, risk mitigation
    STATUS = "status"               # Social standing, achievement, being valued
    MEANING = "meaning"             # Significance, impact, legacy, purpose
    CONNECTION = "connection"       # Relationships, community, being known
    AUTONOMY = "autonomy"           # Independence, self-determination, freedom
    COMFORT = "comfort"             # Ease, enjoyment, avoiding pain
    CURIOSITY = "curiosity"         # Knowledge, understanding, exploration


# Human average allocations - baseline before calibration to individual
# These represent rough population averages, not any ideal
DEFAULT_ALLOCATIONS = {
    Tendency.SURVIVAL: 0.18,      # High - fundamental drive
    Tendency.STATUS: 0.12,
    Tendency.MEANING: 0.10,
    Tendency.CONNECTION: 0.20,    # High - humans are social
    Tendency.AUTONOMY: 0.12,
    Tendency.COMFORT: 0.18,       # High - people seek ease
    Tendency.CURIOSITY: 0.10,
}


@dataclass
class Agent:
    """
    A human tendency that stakes on nodes in value trees.

    Agents compete for influence through staking. Their allocations
    determine how much weight their stakes carry.
    """

    tendency: Tendency
    allocation: float = 0.0       # Percentage of total tokens (should sum to 1.0)

    # Optional: custom description for this person's version of the tendency
    description: Optional[str] = None

    # Performance tracking for allocation adjustment
    stakes_placed: int = 0
    stakes_validated: int = 0     # How many stakes were confirmed by later observations

    @property
    def id(self) -> str:
        return self.tendency.value

    @property
    def default_description(self) -> str:
        """Description of what this tendency optimizes for."""
        descriptions = {
            Tendency.SURVIVAL: "Physical safety, resource acquisition, risk mitigation",
            Tendency.STATUS: "Social standing, achievement, recognition, being valued",
            Tendency.MEANING: "Significance, impact, legacy, purpose beyond self",
            Tendency.CONNECTION: "Relationships, belonging, community, being known",
            Tendency.AUTONOMY: "Independence, self-determination, freedom from constraint",
            Tendency.COMFORT: "Ease, pleasure, avoiding pain, reducing friction",
            Tendency.CURIOSITY: "Knowledge, understanding, exploration, novelty",
        }
        return descriptions.get(self.tendency, "")

    @property
    def validation_rate(self) -> float:
        """How often this agent's stakes are validated by new evidence."""
        if self.stakes_placed == 0:
            return 0.0
        return self.stakes_validated / self.stakes_placed

    def __repr__(self):
        return f"Agent({self.tendency.value}, allocation={self.allocation:.2%})"

    def to_dict(self) -> dict:
        return {
            "tendency": self.tendency.value,
            "allocation": self.allocation,
            "description": self.description,
            "stakes_placed": self.stakes_placed,
            "stakes_validated": self.stakes_validated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        return cls(
            tendency=Tendency(data["tendency"]),
            allocation=data["allocation"],
            description=data.get("description"),
            stakes_placed=data.get("stakes_placed", 0),
            stakes_validated=data.get("stakes_validated", 0),
        )


@dataclass
class AgentSet:
    """
    The set of all agents for a person.

    Manages allocations and provides methods for adjustment.
    Allocations always sum to 1.0.
    """

    agents: dict[Tendency, Agent] = field(default_factory=dict)

    # Whether allocations have been calibrated from defaults
    calibrated: bool = False

    def __post_init__(self):
        # Initialize with defaults if empty
        if not self.agents:
            self._initialize_defaults()

    def _initialize_defaults(self):
        """Create agents with default human-average allocations."""
        for tendency in Tendency:
            self.agents[tendency] = Agent(
                tendency=tendency,
                allocation=DEFAULT_ALLOCATIONS[tendency],
            )

    def get(self, tendency: Tendency) -> Agent:
        """Get agent by tendency."""
        return self.agents[tendency]

    def all(self) -> list[Agent]:
        """All agents."""
        return list(self.agents.values())

    @property
    def total_allocation(self) -> float:
        """Sum of all allocations (should be 1.0)."""
        return sum(a.allocation for a in self.agents.values())

    def normalize(self):
        """Ensure allocations sum to 1.0."""
        total = self.total_allocation
        if total == 0:
            # Reset to defaults
            self._initialize_defaults()
            return

        for agent in self.agents.values():
            agent.allocation /= total

    def adjust_allocation(self, tendency: Tendency, delta: float):
        """
        Adjust an agent's allocation by delta.

        Other agents are adjusted proportionally to maintain sum = 1.0.
        """
        agent = self.agents[tendency]
        old_alloc = agent.allocation
        new_alloc = max(0.01, min(0.99, old_alloc + delta))  # Clamp to [1%, 99%]

        actual_delta = new_alloc - old_alloc
        if abs(actual_delta) < 0.001:
            return  # No meaningful change

        agent.allocation = new_alloc

        # Distribute the inverse delta across other agents proportionally
        others = [a for t, a in self.agents.items() if t != tendency]
        others_total = sum(a.allocation for a in others)

        if others_total > 0:
            for other in others:
                proportion = other.allocation / others_total
                other.allocation -= actual_delta * proportion

        self.normalize()  # Ensure precision

    def set_allocation(self, tendency: Tendency, value: float):
        """Set an agent's allocation to a specific value."""
        current = self.agents[tendency].allocation
        self.adjust_allocation(tendency, value - current)

    def rebalance_by_performance(self, learning_rate: float = 0.1):
        """
        Shift allocations based on agent performance (validation rates).

        Agents with higher validation rates gain allocation.
        Learning rate controls how fast allocations shift.
        """
        # Only adjust if we have performance data
        agents_with_stakes = [a for a in self.agents.values() if a.stakes_placed > 0]
        if len(agents_with_stakes) < 2:
            return

        avg_rate = sum(a.validation_rate for a in agents_with_stakes) / len(agents_with_stakes)

        for agent in agents_with_stakes:
            # Positive delta if above average, negative if below
            performance_delta = agent.validation_rate - avg_rate
            allocation_delta = performance_delta * learning_rate
            self.adjust_allocation(agent.tendency, allocation_delta)

        self.calibrated = True

    def __repr__(self):
        allocs = ", ".join(f"{t.value}={a.allocation:.0%}" for t, a in self.agents.items())
        return f"AgentSet({allocs})"

    def to_dict(self) -> dict:
        return {
            "agents": {t.value: a.to_dict() for t, a in self.agents.items()},
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSet":
        agent_set = cls(agents={})
        for tendency_str, agent_data in data["agents"].items():
            tendency = Tendency(tendency_str)
            agent_set.agents[tendency] = Agent.from_dict(agent_data)
        agent_set.calibrated = data.get("calibrated", False)
        return agent_set

    @classmethod
    def with_profile(cls, profile: dict[Tendency, float]) -> "AgentSet":
        """
        Create agent set with custom allocations.

        Example:
            AgentSet.with_profile({
                Tendency.MEANING: 0.35,
                Tendency.AUTONOMY: 0.30,
                Tendency.CURIOSITY: 0.20,
                Tendency.SURVIVAL: 0.08,
                Tendency.CONNECTION: 0.05,
                Tendency.STATUS: 0.02,
            })
        """
        agent_set = cls()
        for tendency, allocation in profile.items():
            agent_set.agents[tendency].allocation = allocation
        agent_set.normalize()
        agent_set.calibrated = True
        return agent_set
