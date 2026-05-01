#!/usr/bin/env python3
"""
Smooth promotion: sub-claims earn voice as they accumulate stake.

The architecture's core property under test: nodes are not pre-typed
as 'tendency' or 'claim' or 'observation'. Every node has a
*staking capacity* proportional to the PRO stake it has accumulated.
A freshly sprouted sub-claim starts silent (0 capacity). As stake
flows in, it earns standing and begins to stake on its neighbors.

Setup
-----

Two opposing tendencies on a 1D axis (A=left at -1, B=right at +1).
We feed observations to A's side first. As A's PRO sub-claims
accumulate stake, they should:

  1. Start at 0 capacity (silent).
  2. Gain capacity proportional to PRO stake received.
  3. Begin to influence siblings and cross-tendency targets at
     magnitude proportional to their capacity.

We track per-node capacity over rounds and verify the gradient.
"""

from __future__ import annotations

from world_model.generalized import (
    GeneralizedTendency,
    Observation,
    World,
    equilibrate,
)


def banner(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def show_capacities(t: GeneralizedTendency, label: str) -> None:
    print(f"\n  [{t.id}] {label}")
    for node in t.tree.all_nodes():
        cap = t.node_capacity.get(node.id, 0.0)
        kind = "ROOT " if node.position.value == "root" else f"{node.position.value:5s}"
        obs_id = node.observation_id or ""
        print(f"    {kind} {node.id[:6]} obs={obs_id:6s} cap={cap:+.4f}")


def main():
    banner("SMOOTH PROMOTION: sub-claims earn voice as stake accumulates")

    A = GeneralizedTendency(
        id="A", thesis="left",
        anchor=(-1.0,), polarity_axis=(-1.0,),
        budget=1.0, bandwidth=2.0,
    )
    B = GeneralizedTendency(
        id="B", thesis="right",
        anchor=(+1.0,), polarity_axis=(+1.0,),
        budget=1.0, bandwidth=2.0,
    )
    world = World()
    world.add_tendency(A)
    world.add_tendency(B)

    # Round 1: feed A two left observations
    for i, x in enumerate([-1.2, -1.0]):
        world.add_observation(Observation(id=f"L{i}", coords=(x,), label=f"L{i}"))
    rounds = equilibrate(world, max_rounds=4, tolerance=1e-3)
    print(f"\n  Round group 1 ({rounds} rounds): two left obs introduced.")
    show_capacities(A, "After round group 1")

    # Round 2: keep feeding the same observations, capacity should rise
    rounds = equilibrate(world, max_rounds=4, tolerance=1e-3)
    print(f"\n  Round group 2 ({rounds} rounds): same obs, equilibrate again.")
    show_capacities(A, "After round group 2")

    # Round 3: more equilibrations, capacity should further smooth in
    rounds = equilibrate(world, max_rounds=6, tolerance=1e-3)
    print(f"\n  Round group 3 ({rounds} rounds): more equilibration.")
    show_capacities(A, "After round group 3")

    # Verify the capacity gradient
    banner("VERDICT")
    root_id = A.tree.root_node.id
    sub_caps = [
        (n.id, A.node_capacity.get(n.id, 0.0))
        for n in A.tree.all_nodes() if n.id != root_id
    ]
    if not sub_caps:
        print("\n  -- No sub-claims sprouted; can't test promotion.")
        return
    max_sub = max(c for _, c in sub_caps)
    min_sub = min(c for _, c in sub_caps)
    print(f"\n  Root capacity:        {A.node_capacity[root_id]:.3f} (pinned)")
    print(f"  Max sub-claim cap:    {max_sub:.4f}")
    print(f"  Min sub-claim cap:    {min_sub:.4f}")
    if max_sub > 0.01:
        print(f"\n  OK: sub-claims have earned non-zero capacity through")
        print(f"      accumulated PRO stake. They can now influence siblings")
        print(f"      and cross-tendency targets at magnitude {max_sub:.4f}.")
        print(f"      No qualitative line between 'claim' and 'tendency' --")
        print(f"      standing is continuous.")
    else:
        print(f"\n  -- Sub-claims stayed silent. Either no PRO stake landed")
        print(f"     or capacity_decay is too aggressive.")


if __name__ == "__main__":
    main()
