#!/usr/bin/env python3
"""
Test Firestore connection and basic operations.

Usage:
    python test_firestore.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from world_model import (
    ObservationStore, Observation,
    AgentSet, Tendency,
    TreeStore, Tree,
    WorldModel, create_world_model,
)
from world_model.storage.firestore_adapter import FirestoreAdapter

CRED_PATH = "secrets/autonet-275416-3c753cd3530d.json"
TEST_USER_ID = "test_andrei"


def test_connection():
    """Test basic Firestore connection."""
    print("=" * 60)
    print("FIRESTORE CONNECTION TEST")
    print("=" * 60)

    print(f"\nCredentials: {CRED_PATH}")

    try:
        adapter = FirestoreAdapter.from_service_account(CRED_PATH)
        print("Connected to Firestore successfully!")
        return adapter
    except Exception as e:
        print(f"Failed to connect: {e}")
        return None


def test_save_load(adapter: FirestoreAdapter):
    """Test saving and loading a world model."""
    print("\n" + "=" * 60)
    print("SAVE/LOAD TEST")
    print("=" * 60)

    # Create a simple model
    model = WorldModel(name="Test Model")

    # Add some observations
    model.observations.add(Observation(
        content="Test observation 1",
        source_id="test"
    ))
    model.observations.add(Observation(
        content="Test observation 2",
        source_id="test"
    ))

    # Modify agents
    model.agents.get(Tendency.MEANING).allocation = 0.25
    model.agents.normalize()

    print(f"\nSaving model with {len(model.observations)} observations...")
    adapter.save_world_model(TEST_USER_ID, model)
    print("Saved!")

    print("\nLoading model back...")
    loaded = adapter.load_world_model(TEST_USER_ID)

    if loaded:
        print(f"Loaded: {loaded.name}")
        print(f"Observations: {len(loaded.observations)}")
        print(f"Agents: {loaded.agents}")
        print("Save/Load test passed!")
    else:
        print("Failed to load model")


def test_full_model(adapter: FirestoreAdapter):
    """Test with actual Andrei model if available."""
    print("\n" + "=" * 60)
    print("FULL MODEL TEST")
    print("=" * 60)

    obs_file = Path("observations.json")
    if not obs_file.exists():
        print("observations.json not found, skipping full model test")
        return

    print("\nLoading Andrei model from observations.json...")
    model = create_world_model("Andrei", "observations.json")
    print(f"Loaded {len(model.observations)} observations")

    print("\nSaving to Firestore...")
    adapter.save_world_model("andrei", model)
    print("Saved!")

    print("\nLoading back from Firestore...")
    loaded = adapter.load_world_model("andrei")

    if loaded:
        print(f"Loaded: {loaded.name}")
        print(f"Observations: {len(loaded.observations)}")
        print(f"Trees: {len(loaded.trees)}")

        print("\nAgent allocations:")
        for agent in sorted(loaded.agents.all(), key=lambda a: -a.allocation):
            print(f"  {agent.tendency.value}: {agent.allocation:.1%}")

        print("\nFull model test passed!")
    else:
        print("Failed to load model")


def cleanup(adapter: FirestoreAdapter):
    """Clean up test data."""
    print("\n" + "=" * 60)
    print("CLEANUP")
    print("=" * 60)

    response = input("Delete test data from Firestore? (y/N): ")
    if response.lower() == 'y':
        print(f"Deleting {TEST_USER_ID}...")
        adapter.delete_world_model(TEST_USER_ID)
        print("Deleted!")
    else:
        print("Skipping cleanup")


def main():
    adapter = test_connection()
    if not adapter:
        sys.exit(1)

    test_save_load(adapter)
    test_full_model(adapter)
    cleanup(adapter)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
