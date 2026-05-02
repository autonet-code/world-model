#!/usr/bin/env python3
"""Tests for content-addressed node IDs in GeneralizedTendency.

The contract:
  1. Two GeneralizedTendency instances with the same id, anchor, and
     polarity, after sprouting an identical child (same position, same
     anchor, same axis), produce the same child node id.
  2. Sprouting different children (different position OR different
     anchor OR different axis) under the same parent produces
     different ids.
  3. Sprouting an identical child under different parents (different
     tendency id, or different intermediate-claim path) produces
     different ids.
"""

from __future__ import annotations

import sys

from world_model.generalized.tendency import GeneralizedTendency
from world_model.models.tree import Position


def _make_tendency(
    tid: str = "T_x",
    anchor: tuple = (1.0, 0.0, 0.0),
    polarity: tuple = (1.0, 0.0, 0.0),
) -> GeneralizedTendency:
    return GeneralizedTendency(
        id=tid,
        thesis=f"thesis-of-{tid}",
        anchor=anchor,
        polarity_axis=polarity,
    )


def test_identical_children_same_id() -> None:
    """Two distinct tendency instances with the same id sprouting the
    same child produce the same node id, despite their roots having
    different local UUIDs."""
    t1 = _make_tendency()
    t2 = _make_tendency()

    # Sanity: their root UUIDs differ.
    assert t1.tree.root_node.id != t2.tree.root_node.id, (
        "roots should have distinct UUIDs"
    )

    child_anchor = (1.0, 0.5, 0.0)
    child_axis = (1.0, 0.0, 0.0)

    n1 = t1.sprout_child(
        parent_node_id=t1.tree.root_node.id,
        position=Position.PRO,
        anchor=child_anchor,
        polarity_axis=child_axis,
    )
    n2 = t2.sprout_child(
        parent_node_id=t2.tree.root_node.id,
        position=Position.PRO,
        anchor=child_anchor,
        polarity_axis=child_axis,
    )

    assert n1.id == n2.id, (
        f"identical content under same tendency id should hash equal; "
        f"got {n1.id!r} vs {n2.id!r}"
    )
    assert n1.id.startswith("n_"), f"expected hash id, got {n1.id!r}"


def test_different_children_different_ids() -> None:
    """Sprouting different children under the same parent produces
    different node ids."""
    t = _make_tendency()
    parent = t.tree.root_node.id

    base_anchor = (1.0, 0.0, 0.0)
    base_axis = (1.0, 0.0, 0.0)

    n_pro = t.sprout_child(
        parent_node_id=parent,
        position=Position.PRO,
        anchor=base_anchor,
        polarity_axis=base_axis,
    )

    # Different position: CON instead of PRO.
    t2 = _make_tendency()
    n_con = t2.sprout_child(
        parent_node_id=t2.tree.root_node.id,
        position=Position.CON,
        anchor=base_anchor,
        polarity_axis=base_axis,
    )
    assert n_pro.id != n_con.id, "different position must yield different id"

    # Different anchor.
    t3 = _make_tendency()
    n_anchor = t3.sprout_child(
        parent_node_id=t3.tree.root_node.id,
        position=Position.PRO,
        anchor=(0.0, 1.0, 0.0),
        polarity_axis=base_axis,
    )
    assert n_pro.id != n_anchor.id, "different anchor must yield different id"

    # Different polarity axis.
    t4 = _make_tendency()
    n_axis = t4.sprout_child(
        parent_node_id=t4.tree.root_node.id,
        position=Position.PRO,
        anchor=base_anchor,
        polarity_axis=(0.0, 1.0, 0.0),
    )
    assert n_pro.id != n_axis.id, "different polarity axis must yield different id"


def test_identical_child_different_parents_different_ids() -> None:
    """An identical child under different parents must produce
    different ids, whether the parent differs by tendency id or by
    intermediate-claim path."""
    # Case A: different tendency ids -> different ROOT path -> different id.
    t1 = _make_tendency(tid="T_a")
    t2 = _make_tendency(tid="T_b")
    child_anchor = (1.0, 0.5, 0.0)
    child_axis = (1.0, 0.0, 0.0)
    n1 = t1.sprout_child(
        parent_node_id=t1.tree.root_node.id,
        position=Position.PRO,
        anchor=child_anchor,
        polarity_axis=child_axis,
    )
    n2 = t2.sprout_child(
        parent_node_id=t2.tree.root_node.id,
        position=Position.PRO,
        anchor=child_anchor,
        polarity_axis=child_axis,
    )
    assert n1.id != n2.id, (
        "identical content under different tendency roots should differ"
    )

    # Case B: same tendency id but child placed under different
    # intermediate parents (different intermediate anchors). The deeper
    # child should differ between the two parent paths.
    ta = _make_tendency(tid="T_shared")
    tb = _make_tendency(tid="T_shared")

    # Sprout an intermediate node in each, with DIFFERENT anchors so
    # the path string differs.
    mid_a = ta.sprout_child(
        parent_node_id=ta.tree.root_node.id,
        position=Position.PRO,
        anchor=(1.0, 0.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
    )
    mid_b = tb.sprout_child(
        parent_node_id=tb.tree.root_node.id,
        position=Position.PRO,
        anchor=(0.0, 1.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
    )
    assert mid_a.id != mid_b.id, "different intermediate anchors should differ"

    leaf_a = ta.sprout_child(
        parent_node_id=mid_a.id,
        position=Position.PRO,
        anchor=child_anchor,
        polarity_axis=child_axis,
    )
    leaf_b = tb.sprout_child(
        parent_node_id=mid_b.id,
        position=Position.PRO,
        anchor=child_anchor,
        polarity_axis=child_axis,
    )
    assert leaf_a.id != leaf_b.id, (
        "identical leaf content under different parent paths should differ"
    )

    # Sanity: same intermediate path -> same leaf id.
    tc = _make_tendency(tid="T_shared")
    td = _make_tendency(tid="T_shared")
    mid_c = tc.sprout_child(
        parent_node_id=tc.tree.root_node.id,
        position=Position.PRO,
        anchor=(1.0, 0.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
    )
    mid_d = td.sprout_child(
        parent_node_id=td.tree.root_node.id,
        position=Position.PRO,
        anchor=(1.0, 0.0, 0.0),
        polarity_axis=(1.0, 0.0, 0.0),
    )
    assert mid_c.id == mid_d.id, "same intermediate path should yield same id"
    leaf_c = tc.sprout_child(
        parent_node_id=mid_c.id,
        position=Position.PRO,
        anchor=child_anchor,
        polarity_axis=child_axis,
    )
    leaf_d = td.sprout_child(
        parent_node_id=mid_d.id,
        position=Position.PRO,
        anchor=child_anchor,
        polarity_axis=child_axis,
    )
    assert leaf_c.id == leaf_d.id, (
        "same path + same leaf content should yield same id across solvers"
    )


def main() -> int:
    tests = [
        test_identical_children_same_id,
        test_different_children_different_ids,
        test_identical_child_different_parents_different_ids,
    ]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  OK  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print()
    if failed:
        print(f"  {failed}/{len(tests)} test(s) failed")
        return 1
    print(f"  {len(tests)}/{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
