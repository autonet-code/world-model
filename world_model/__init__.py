"""
World Model - Agent-relative computation for personal worldview representation.

Models decision-making as competing internal drives (tendencies) that stake
positions on observations, debate in adversarial arenas, and shift allocations
based on novelty-modulated attention.

Subpackages:
    world_model.models      - Core data structures (Observation, Agent, Tree, etc.)
    world_model.attention   - Attention curves, symbol streams, salience, routing
    world_model.novelty     - Novelty measurement against reference frames
    world_model.dynamics    - Adversarial arena and training loop
    world_model.staking     - Evidence staking mechanisms
    world_model.extraction  - Observation extraction from text/voice/tweets
    world_model.agents      - Autonomous agents driven by world model
    world_model.storage     - JSON and Firestore persistence

Usage:
    from world_model import Tendency, AgentSet, Observation, ObservationStore
    from world_model import Tree, TreeStore, Node, Position, Stake
    from world_model.attention import NoveltyAttentionCurve, Sequence, Symbol
    from world_model.novelty import measure_against_claims, HybridProbe
    from world_model.dynamics import Arena, Claim, DebateResult
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

# Attention curves (top-level convenience)
from .attention import (
    NoveltyAttentionCurve,
    AttentionState,
    EXPLORER_CURVE,
    BALANCED_CURVE,
    CONSERVATIVE_CURVE,
    sigmoid,
)

# Integration bridge
from .integration import AttentionBridge, ArenaFeedback, AttentionEvent, FeedbackEvent

# Extraction
from .extraction import DeviationExtractor, ObservationExtractor

# Staking
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
    # Attention curves
    "NoveltyAttentionCurve", "AttentionState",
    "EXPLORER_CURVE", "BALANCED_CURVE", "CONSERVATIVE_CURVE",
    "sigmoid",
    # Integration
    "AttentionBridge", "ArenaFeedback", "AttentionEvent", "FeedbackEvent",
    # Extraction
    "ObservationExtractor",
    # Staking
    "Staker", "BatchStaker", "StakeDecision", "HierarchicalStaker",
    # Dynamics
    "Arena", "Claim", "DebateResult",
    # Storage
    "WorldModel", "create_world_model",
]

__version__ = "1.0.0"
