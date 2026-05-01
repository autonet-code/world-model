"""Equilibration: run tendencies' actions until budgets stabilize.

Each round:
  1. Each tendency calls .act(world) -> populates last_stakes.
  2. World.apply_stakes() writes them onto nodes (own + cross).
  3. Convergence check: have last_stakes dictionaries stabilized
     (max abs delta < tolerance) compared to previous round?

Until convergence, repeat. Returns the number of rounds.

Note: the staking policy is deterministic given world state, so the
convergence is from the *interaction* between tendencies (each one's
absorption of new observations changes its frame, which changes
everyone else's staking next round).
"""

from __future__ import annotations

from typing import Dict, Tuple

from .world import World


def equilibrate(world: World, max_rounds: int = 20, tolerance: float = 1e-3) -> int:
    """Run rounds of (act, apply_stakes) until intents stabilize.

    Returns the number of rounds executed.
    """
    prev_intents: Dict[str, Dict[Tuple[str, str], float]] = {
        tid: dict(t.last_stakes) for tid, t in world.tendencies.items()
    }
    for round_idx in range(1, max_rounds + 1):
        # Each tendency computes its intents
        for tendency in world.tendencies.values():
            tendency.act(world)
        # Apply
        world.apply_stakes()
        # Check convergence
        max_delta = 0.0
        for tid, tendency in world.tendencies.items():
            old = prev_intents.get(tid, {})
            new = tendency.last_stakes
            keys = set(old.keys()) | set(new.keys())
            for k in keys:
                d = abs(old.get(k, 0.0) - new.get(k, 0.0))
                if d > max_delta:
                    max_delta = d
        if max_delta < tolerance and round_idx >= 2:
            return round_idx
        prev_intents = {tid: dict(t.last_stakes) for tid, t in world.tendencies.items()}
    return max_rounds
