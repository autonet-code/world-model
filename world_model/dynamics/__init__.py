"""
Dynamics - Adversarial competition between agents.

The dynamics layer is where "life" happens. Agents compete to
define the person's worldview through debate.
"""

from .arena import Arena, Claim, StakeDecision, DebateResult
from .trainer import (
    Trainer, TrainConfig, TrainHistory,
    Validator, ValidationResult,
    ConsoleLogger, JSONLogger, TensorBoardLogger, WandbLogger,
)

__all__ = [
    "Arena", "Claim", "StakeDecision", "DebateResult",
    "Trainer", "TrainConfig", "TrainHistory",
    "Validator", "ValidationResult",
    "ConsoleLogger", "JSONLogger", "TensorBoardLogger", "WandbLogger",
]
