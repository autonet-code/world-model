#!/usr/bin/env python3
"""Tests for `scope_for_coords` / `scope_for_observation`.

The scope helper exists to feed `equilibrate(world, scope=...)` on the
hot path. Its correctness gate: it must agree with the locality
declaration the architecture already makes for cross-tendency edge
discovery (`_maybe_add_cross_tendency_edges` in tendency.py uses
`distance < bandwidth * 1.5`). If the scope helper returns a different
set from what cross-edge discovery would acquire, the scoped
equilibrate would silently skip work that the discrete kernel
considers in-locality.
"""

from __future__ import annotations

from world_model.generalized import (
    GeneralizedTendency,
    Observation,
    World,
)
from world_model.generalized.scope import (
    scope_for_coords,
    scope_for_observation,
)
from world_model.models.tree import Position


def _make_world() -> World:
    world = World()
    world.add_tendency(GeneralizedTendency(
        id="T_a",
        thesis="A",
        anchor=(1.0, 0.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
        bandwidth=0.7,
    ))
    world.add_tendency(GeneralizedTendency(
        id="T_b",
        thesis="B",
        anchor=(0.0, 1.0, 0.0),
        polarity_axis=(0.0, 1.0, 0.0),
        bandwidth=0.7,
    ))
    world.add_tendency(GeneralizedTendency(
        id="T_c",
        thesis="C",
        anchor=(0.0, 0.0, 1.0),
        polarity_axis=(0.0, 0.0, 1.0),
        bandwidth=0.7,
    ))
    return world


def test_scope_includes_only_in_bandwidth() -> None:
    """A coord squarely on T_a's anchor falls within T_a's bandwidth
    and outside T_b's/T_c's. Scope must return {T_a} only."""
    world = _make_world()
    scope = scope_for_coords(world, (1.0, 0.0, 0.0))
    assert scope == {"T_a"}, f"expected {{T_a}}, got {scope}"


def test_scope_can_include_multiple() -> None:
    """A coord near the bisector of A and B should fall within both
    of their bandwidths (slack=1.5 gives gate=1.05; coords at
    (0.6, 0.6, 0.0) sit at distance sqrt(0.16+0.36)=~0.72 from both)."""
    world = _make_world()
    scope = scope_for_coords(world, (0.6, 0.6, 0.0))
    assert scope == {"T_a", "T_b"}, f"expected {{T_a, T_b}}, got {scope}"


def test_scope_can_be_empty() -> None:
    """A far-out coord falls inside no tendency's bandwidth and
    yields the empty set."""
    world = _make_world()
    scope = scope_for_coords(world, (5.0, 5.0, 5.0))
    assert scope == set(), f"expected empty set, got {scope}"


def test_scope_matches_cross_tendency_edge_discovery() -> None:
    """The architectural correctness gate: for the same coord, the
    helper's returned set must equal the set of tendencies that
    sprout_child would acquire cross-parent edges in via
    `_maybe_add_cross_tendency_edges`. We exercise this by actually
    sprouting and reading which tendencies appear in the parent set,
    then comparing to the helper's prediction.
    """
    world = _make_world()
    t_a = world.tendencies["T_a"]
    coord = (0.6, 0.6, 0.0)
    new_node = t_a.sprout_child(
        parent_node_id=t_a.tree.root_node.id,
        position=Position.PRO,
        anchor=coord,
        polarity_axis=(1.0, 1.0, 0.0),
        world=world,
    )
    actual_parent_tendencies = {p.tendency_id for p in new_node.parents}
    predicted_scope = scope_for_coords(world, coord)
    # The helper must include T_a (the sprouting tendency); cross-edge
    # discovery skips self because the node already has T_a as parent.
    # So `predicted_scope` should be a SUPERSET of actual_parent_tendencies
    # by exactly {sprouting tendency}, OR equal when the sprouter is
    # also within its own bandwidth (which T_a is for its own anchor
    # neighborhood).
    assert "T_a" in predicted_scope, (
        f"sprouter T_a missing from scope: {predicted_scope}"
    )
    # The non-sprouter tendencies in the predicted scope must match
    # exactly the cross-tendency parent set.
    non_self_scope = predicted_scope - {"T_a"}
    non_self_actual = actual_parent_tendencies - {"T_a"}
    assert non_self_scope == non_self_actual, (
        f"helper predicted cross-tendencies {non_self_scope}, "
        f"sprout_child acquired {non_self_actual}"
    )


def test_scope_for_observation_wrapper() -> None:
    """The observation-accepting wrapper produces the same set as
    the coords-accepting form."""
    world = _make_world()
    obs = Observation(id="obs1", coords=(0.6, 0.6, 0.0), label="test")
    by_coords = scope_for_coords(world, (0.6, 0.6, 0.0))
    by_obs = scope_for_observation(world, obs)
    assert by_coords == by_obs, (
        f"wrapper diverged: coords={by_coords}, obs={by_obs}"
    )


if __name__ == "__main__":
    test_scope_includes_only_in_bandwidth()
    print("OK: scope includes only in-bandwidth tendencies")
    test_scope_can_include_multiple()
    print("OK: scope can include multiple tendencies")
    test_scope_can_be_empty()
    print("OK: scope can be empty")
    test_scope_matches_cross_tendency_edge_discovery()
    print("OK: scope matches cross-tendency edge discovery")
    test_scope_for_observation_wrapper()
    print("OK: observation wrapper matches coords form")
