"""
World Model - Adversarial equilibrium for personal worldview representation

A system for modeling a person's worldview as an adversarial equilibrium of
competing internal tendencies (agents). Agents propose claims about what matters,
stake observations to support their position and undermine competitors, and
gain or lose influence based on debate outcomes.

Core concepts:
- Observation: Atomic fact about a person (~280 bytes, no inherent polarity)
- Agent: Human tendency that ACTIVELY competes (proposes, stakes, wins/loses)
- Tree/Claim: A position an agent proposes ("This is what matters")
- Node: Observation positioned (pro/con) as argument in the debate
- Arena: Where adversarial competition happens
- Weight propagation: net_score = direct + Σ(pro) - Σ(con)

The "life" is in the competition: same observation means different things to
different tendencies. Agents stake adversarially - supporting their claims,
undermining competitors. Winners gain allocation. The equilibrium IS personality.

Usage:
    from world_model import (
        ObservationStore, AgentSet, Arena, WorldModel
    )

    # Load observations
    model = WorldModel(name="Person")
    # ... extract observations into model.observations

    # Run adversarial debate
    arena = Arena()
    trees, result = arena.run_full_debate(
        observations=model.observations,
        agents=model.agents,
        rounds=1,
    )

    # Winner's claim is best supported by evidence
    print(f"Winner: {result.winner}")
    print(f"Final allocations: {model.agents}")
"""

# Models
from .models import (
    # Legacy (deviation-based)
    DeviationNode, DeviationType, Edge, EdgeType,
    EvidencePointer, Source,
    # Current (observation + tree-based)
    Observation, ObservationStore,
    Agent, AgentSet, Tendency, DEFAULT_ALLOCATIONS,
    Tree, TreeStore, Node, Position, Stake,
)

# Extraction
from .extraction import DeviationExtractor, ObservationExtractor

# Staking (legacy - use Arena for adversarial dynamics)
from .staking import Staker, BatchStaker, StakeDecision, HierarchicalStaker

# Dynamics (adversarial competition)
from .dynamics import Arena, Claim, DebateResult

# Storage
from .storage import DeviationGraph, WorldModel, create_world_model

__all__ = [
    # Legacy
    "DeviationNode", "DeviationType", "Edge", "EdgeType",
    "EvidencePointer", "Source", "DeviationExtractor", "DeviationGraph",
    # Observations
    "Observation", "ObservationStore",
    # Agents
    "Agent", "AgentSet", "Tendency", "DEFAULT_ALLOCATIONS",
    # Trees
    "Tree", "TreeStore", "Node", "Position", "Stake",
    # Extraction
    "ObservationExtractor",
    # Staking (legacy)
    "Staker", "BatchStaker", "StakeDecision", "HierarchicalStaker",
    # Dynamics (adversarial)
    "Arena", "Claim", "DebateResult",
    # Storage
    "WorldModel", "create_world_model",
]

__version__ = "0.5.0"
