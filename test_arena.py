#!/usr/bin/env python3
"""
Test the adversarial Arena dynamics.

This runs a full debate:
1. Agents propose claims about what matters
2. Agents stake observations adversarially
3. Resolution determines winners
4. Allocations shift to winners
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from world_model import (
    ObservationStore, Observation,
    AgentSet, Tendency,
    Arena,
    WorldModel, create_world_model,
)


def main():
    print("="*70)
    print("ADVERSARIAL ARENA TEST")
    print("="*70)

    # Load observations
    print("\nLoading observations...")
    model = create_world_model("Andrei", "observations.json")
    print(f"Loaded {len(model.observations)} observations")

    # Show initial allocations
    print("\nInitial agent allocations (human average):")
    for agent in sorted(model.agents.all(), key=lambda a: -a.allocation):
        print(f"  {agent.tendency.value}: {agent.allocation:.0%}")

    # Run adversarial debate
    print("\n" + "="*70)
    arena = Arena()

    # Use subset for faster testing
    test_store = ObservationStore()
    for obs in model.observations.all()[:30]:  # First 30 observations
        test_store.add(obs)

    trees, result = arena.run_full_debate(
        observations=test_store,
        agents=model.agents,
        rounds=1,
        learning_rate=0.15,
        verbose=True,
    )

    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    print(f"\nWinner: {result.winner.value.upper() if result.winner else 'None'}")

    print("\nClaim scores:")
    for claim in sorted(result.claims, key=lambda c: -c.score):
        print(f"  [{claim.proposer.value}] {claim.score:.3f}: \"{claim.tree.root_value[:50]}...\"")

    print("\nFinal allocations:")
    for agent in sorted(model.agents.all(), key=lambda a: -a.allocation):
        change = result.allocation_changes.get(agent.tendency, 0)
        direction = "+" if change > 0 else ""
        print(f"  {agent.tendency.value}: {agent.allocation:.1%} ({direction}{change:.1%})")

    # Show tree structure of winning claim
    if result.winner:
        winning_claim = next(c for c in result.claims if c.proposer == result.winner)
        print(f"\nWinning tree structure:")
        print_tree(winning_claim.tree.root_node)

    # Save the model
    model.trees = trees
    model.save("andrei_adversarial.json")
    print(f"\nSaved to andrei_adversarial.json")


def print_tree(node, depth=0):
    """Print tree structure."""
    prefix = "  " * depth
    pos = node.position.value.upper()
    content = node.content[:45] + "..." if len(node.content) > 45 else node.content
    print(f"{prefix}[{pos}] {content}")
    for c in node.pro_children[:3]:  # Limit for readability
        print_tree(c, depth + 1)
    for c in node.con_children[:3]:
        print_tree(c, depth + 1)
    remaining = len(node.pro_children) + len(node.con_children) - 6
    if remaining > 0:
        print(f"{prefix}  ... and {remaining} more")


if __name__ == "__main__":
    main()
