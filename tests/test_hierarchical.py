#!/usr/bin/env python3
"""
Test the hierarchical staking pipeline.

Phase 1: Identify anchors (observations that directly address root value)
Phase 2: Position remaining observations relative to existing nodes
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from world_model import (
    Observation, ObservationStore,
    AgentSet, Tendency,
    Tree, Position,
    HierarchicalStaker,
)


def load_observations(path: str) -> ObservationStore:
    """Load observations from JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    store = ObservationStore()

    for obs_data in data["observations"]:
        obs = Observation.from_dict(obs_data)
        store.add(obs)

    return store


def print_tree(tree: Tree, indent: int = 0):
    """Pretty print tree structure."""
    def print_node(node, depth):
        prefix = "  " * depth
        pos = node.position.value.upper() if node.position != Position.ROOT else "ROOT"
        score = f"score={node.net_score:.3f}"
        content = node.content[:50] + "..." if len(node.content) > 50 else node.content
        print(f"{prefix}[{pos}] {content} ({score})")

        for child in node.pro_children:
            print_node(child, depth + 1)
        for child in node.con_children:
            print_node(child, depth + 1)

    print_node(tree.root_node, indent)


def main():
    print("=" * 70)
    print("Hierarchical Staking Pipeline Test")
    print("=" * 70)

    # Load observations
    print("\nLoading observations...")
    obs_path = Path(__file__).parent / "observations.json"
    if not obs_path.exists():
        print(f"ERROR: {obs_path} not found")
        sys.exit(1)

    store = load_observations(str(obs_path))
    print(f"Loaded {len(store)} observations")

    # Create agents
    agents = AgentSet()
    print(f"\nAgents: {agents}")

    # Create tree
    tree = Tree(root_value="Decentralized coordination and DAO governance")
    print(f"\nTree: {tree.root_value}")

    # Run hierarchical staking
    staker = HierarchicalStaker()

    print("\n" + "-" * 70)
    stats = staker.stake_all(store, tree, agents, verbose=True)
    print("-" * 70)

    # Print tree structure
    print("\n" + "=" * 70)
    print("TREE STRUCTURE")
    print("=" * 70)
    print_tree(tree)

    # Show contested nodes
    print("\n" + "=" * 70)
    print("CONTESTED NODES (internal tensions)")
    print("=" * 70)
    contested = tree.contested_nodes(min_stakes=3)
    if contested:
        for node in contested[:5]:
            print(f"\n  '{node.content[:60]}...'")
            print(f"  Position: {node.position.value}, Score: {node.net_score:.3f}")
            stakes = node.stakes_by_agent()
            for agent_id, weight in sorted(stakes.items(), key=lambda x: -x[1])[:4]:
                print(f"    {agent_id}: {weight:.4f}")
    else:
        print("  No highly contested nodes found")

    # Agent summary
    print("\n" + "=" * 70)
    print("AGENT ACTIVITY")
    print("=" * 70)
    for agent in sorted(agents.all(), key=lambda a: -a.stakes_placed):
        if agent.stakes_placed > 0:
            print(f"  {agent.tendency.value:12} {agent.stakes_placed:3} stakes  (allocation: {agent.allocation:.0%})")

    # Final stats
    print("\n" + "=" * 70)
    print("FINAL STATS")
    print("=" * 70)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
