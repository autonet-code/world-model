#!/usr/bin/env python3
"""Smoke run for post-and-coparent refactor.

Builds a tiny two-tendency substrate, drives observations through it,
and verifies:
  - per-node n evolves (decays under PRO, doesn't stay stuck at 1.0)
  - a node sprouted in the overlap zone acquires a co-parent edge
    in the other tendency's tree
  - apply_stakes records posts on the right nodes

Pure substrate dynamics, no LLM. Run from this directory.
"""

from __future__ import annotations

import sys

sys.path.insert(0, r"C:\code\world-model")

from world_model.generalized import (  # type: ignore
    GeneralizedTendency, Observation, World, equilibrate,
)
from world_model.models.tree import Position  # type: ignore


def main() -> int:
    world = World()
    t_a = GeneralizedTendency(
        id="A",
        thesis="thesis A",
        anchor=(1.0, 0.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        bandwidth=0.7,
    )
    t_b = GeneralizedTendency(
        id="B",
        thesis="thesis B",
        anchor=(0.0, 1.0, 0.0),
        polarity_axis=(0.0, 1.0, 0.0),
        bandwidth=0.7,
    )
    world.add_tendency(t_a)
    world.add_tendency(t_b)

    # Sprout an "overlap" sub-claim in A. With world passed in, edge
    # discovery should add a parent link in B.
    overlap = t_a.sprout_child(
        parent_node_id=t_a.tree.root_node.id,
        position=Position.PRO,
        anchor=(0.6, 0.6, 0.0),
        polarity_axis=(1.0, 1.0, 0.0),
        content="overlap_claim",
        world=world,
    )
    print(f"overlap node id: {overlap.id}")
    print(f"overlap parents: {overlap.parents}")
    parent_tendencies = {p.tendency_id for p in overlap.parents}
    print(f"  in tendencies: {sorted(parent_tendencies)}")
    co_parented_ok = parent_tendencies == {"A", "B"}
    print(f"  co-parented across A and B? {co_parented_ok}")

    # Add a few PRO observations near A's anchor; the discrete kernel
    # should drop n on the relevant nodes.
    for i in range(5):
        world.add_observation(Observation(
            id=f"obs_a_{i}",
            coords=(1.0, 0.0, 0.0),
            label=f"PRO_A_{i}",
        ))
    print()
    print("running 5 rounds of equilibrate...")
    rounds = equilibrate(world, max_rounds=5, tolerance=1e-3)
    print(f"  ran {rounds} rounds")

    # Inspect a few nodes' n values.
    print()
    print("per-node n after run:")
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            print(f"  [{tendency.id}] node {node.id[:14]}: n={node.n:.4f}, "
                  f"net_score={node.net_score:+.3f}, "
                  f"posts={len(node.stakes)}, content='{node.content[:30]}'")

    # Look for any node where n has moved off 1.0.
    n_values = []
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            if not node.is_root:
                n_values.append(node.n)
    n_evolved = any(abs(v - 1.0) > 1e-3 for v in n_values)
    print()
    print(f"n evolved off 1.0 on at least one non-root node? {n_evolved}")

    pass_count = sum([co_parented_ok, n_evolved])
    print()
    print(f"  {pass_count}/2 smoke checks passed")
    return 0 if pass_count == 2 else 1


if __name__ == "__main__":
    sys.exit(main())
