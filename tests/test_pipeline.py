#!/usr/bin/env python3
"""
Test the world model pipeline end-to-end.

1. Load observations from observations.json
2. Create agent set with default allocations
3. Create a tree with a relevant value
4. Stake observations into the tree
5. Show results
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from world_model import (
    Observation, ObservationStore,
    AgentSet, Tendency,
    Tree, Position,
    Staker,
)


def load_observations(path: str) -> ObservationStore:
    """Load observations from JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    store = ObservationStore()

    for obs_data in data["observations"]:
        obs = Observation.from_dict(obs_data)
        store.add(obs)

    return store


def main():
    print("=" * 60)
    print("World Model Pipeline Test")
    print("=" * 60)

    # 1. Load observations
    print("\n1. Loading observations...")
    obs_path = Path(__file__).parent / "observations.json"
    if not obs_path.exists():
        print(f"   ERROR: {obs_path} not found")
        print("   Run extract_observations.py first")
        sys.exit(1)

    store = load_observations(str(obs_path))
    print(f"   Loaded {len(store)} observations")

    # Show a few
    print("\n   Sample observations:")
    for obs in store.all()[:5]:
        print(f"   - {obs.content[:70]}...")

    # 2. Create agent set
    print("\n2. Creating agent set with default allocations...")
    agents = AgentSet()
    print(f"   {agents}")

    # 3. Create trees for different values
    print("\n3. Creating value trees...")

    trees = [
        Tree(root_value="Decentralized coordination and DAO governance"),
        Tree(root_value="Personal financial stability and security"),
        Tree(root_value="Finding meaning and purpose in life"),
    ]

    for tree in trees:
        print(f"   - {tree.root_value}")

    # 4. Stake observations (just a few for testing)
    print("\n4. Staking observations into first tree...")
    print("   (This calls Claude for each observation - may take a moment)")

    staker = Staker()
    test_tree = trees[0]  # Decentralized coordination
    test_obs = store.all()[:5]  # Just first 5 for speed

    nodes_created = 0
    for i, obs in enumerate(test_obs):
        print(f"\n   [{i+1}/{len(test_obs)}] {obs.content[:50]}...")
        try:
            node = staker.stake_observation(obs, test_tree, agents)
            if node:
                nodes_created += 1
                print(f"       -> {node.position.value.upper()}, score={node.net_score:.3f}")

                # Show agent stakes
                stakes = node.stakes_by_agent()
                top_stakes = sorted(stakes.items(), key=lambda x: x[1], reverse=True)[:3]
                for agent_id, weight in top_stakes:
                    print(f"          {agent_id}: {weight:.4f}")
            else:
                print(f"       -> (not relevant to this tree)")
        except Exception as e:
            print(f"       -> ERROR: {e}")

    # 5. Show results
    print("\n" + "=" * 60)
    print("5. Results")
    print("=" * 60)

    print(f"\n   Tree: {test_tree.root_value}")
    print(f"   Total nodes: {len(test_tree.all_nodes())}")
    print(f"   Tree score: {test_tree.score:.3f}")

    # Show pro vs con breakdown
    pro_nodes = [n for n in test_tree.all_nodes() if n.position == Position.PRO]
    con_nodes = [n for n in test_tree.all_nodes() if n.position == Position.CON]
    print(f"\n   PRO nodes: {len(pro_nodes)}")
    for node in pro_nodes:
        print(f"     + {node.content[:60]}...")

    print(f"\n   CON nodes: {len(con_nodes)}")
    for node in con_nodes:
        print(f"     - {node.content[:60]}...")

    # Show contested nodes
    contested = test_tree.contested_nodes(min_stakes=2)
    if contested:
        print(f"\n   Contested nodes (2+ agents): {len(contested)}")
        for node in contested[:3]:
            print(f"     * {node.content[:50]}...")
            print(f"       Stakes: {node.stakes_by_agent()}")

    # Agent stats
    print("\n   Agent stakes placed:")
    for agent in agents.all():
        if agent.stakes_placed > 0:
            print(f"     {agent.tendency.value}: {agent.stakes_placed} stakes")

    print("\n" + "=" * 60)
    print("Pipeline test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
