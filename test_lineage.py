#!/usr/bin/env python3
"""Smoke tests for the lineage subsystem.

Validates:
  - EngineClock is monotonic and unique per emission
  - Events propagate to neighbors per stake-weight cutoff and radius
  - Distant tendencies (below cutoff) don't see events
  - Tendencies in the same neighborhood see the same event with the
    same engine-time stamp
  - Bounded rings evict oldest on overflow
  - Allocation rolling window tracks shifts on the origin tendency
  - Global event view reconstructs a total ordering by engine-time
"""

from __future__ import annotations

from world_model import (
    EngineClock,
    Event,
    EventType,
    Lineage,
    LineageRecorder,
    StakeWeightGraph,
)


def banner(text: str) -> None:
    print()
    print("-" * 60)
    print(text)
    print("-" * 60)


def make_recorder(
    edges: list[tuple[str, str, float]],
    min_weight: float = 0.01,
    radius: int = 2,
    ring_capacity: int = 256,
) -> LineageRecorder:
    """Build a recorder with the given edges and a fresh lineage per node."""
    graph = StakeWeightGraph()
    nodes: set[str] = set()
    for a, b, w in edges:
        graph.add_edge(a, b, w)
        nodes.add(a)
        nodes.add(b)
    rec = LineageRecorder(
        clock=EngineClock(),
        graph=graph,
        propagation_min_weight=min_weight,
        propagation_radius=radius,
    )
    for n in nodes:
        rec.register(n, Lineage(event_ring_capacity=ring_capacity))
    return rec


def test_engine_clock_monotonic_and_unique() -> None:
    banner("test: EngineClock ticks monotonically and uniquely")
    clock = EngineClock()
    a = clock.tick()
    b = clock.tick()
    c = clock.now()
    d = clock.tick()
    assert a == 1 and b == 2 and c == 2 and d == 3, (a, b, c, d)
    print(f"  ticks: {a}, {b}, now()={c}, {d}")


def test_event_propagates_to_origin_only_when_isolated() -> None:
    banner("test: isolated tendency records its own events but no others'")
    rec = make_recorder([])
    rec.register("alpha")
    rec.register("beta")
    rec.emit(EventType.STAKED, origin_id="alpha", payload={"obs": "o1"})
    assert len(rec.lineage_of("alpha").events()) == 1
    assert len(rec.lineage_of("beta").events()) == 0
    print(f"  alpha events: {len(rec.lineage_of('alpha').events())}")
    print(f"  beta events:  {len(rec.lineage_of('beta').events())}")


def test_event_propagates_to_strong_neighbor() -> None:
    banner("test: event propagates to neighbors above cutoff")
    rec = make_recorder([
        ("alpha", "beta", 0.5),    # strong link
        ("alpha", "gamma", 0.005), # below cutoff
    ], min_weight=0.01, radius=1)
    rec.emit(EventType.STAKED, origin_id="alpha", payload={"obs": "o1"})
    assert len(rec.lineage_of("alpha").events()) == 1
    assert len(rec.lineage_of("beta").events()) == 1, "beta should see strong-link event"
    assert len(rec.lineage_of("gamma").events()) == 0, "gamma should not (below cutoff)"
    print(f"  alpha (origin): {len(rec.lineage_of('alpha').events())} event(s)")
    print(f"  beta  (w=0.5):  {len(rec.lineage_of('beta').events())} event(s)")
    print(f"  gamma (w=0.005, below cutoff): {len(rec.lineage_of('gamma').events())} event(s)")


def test_radius_bounds_propagation() -> None:
    banner("test: propagation radius bounds reach")
    # Chain: alpha - beta - gamma - delta
    rec = make_recorder([
        ("alpha", "beta",  0.5),
        ("beta",  "gamma", 0.5),
        ("gamma", "delta", 0.5),
    ], min_weight=0.01, radius=2)
    rec.emit(EventType.STAKED, origin_id="alpha", payload={})
    # radius=2 from alpha: alpha (0), beta (1), gamma (2). delta (3) excluded.
    assert len(rec.lineage_of("alpha").events()) == 1
    assert len(rec.lineage_of("beta").events()) == 1
    assert len(rec.lineage_of("gamma").events()) == 1
    assert len(rec.lineage_of("delta").events()) == 0, "delta is 3 hops, excluded"
    print(f"  alpha:0  beta:1  gamma:2  delta:3  (radius=2)")
    print(f"  reached: alpha,beta,gamma; excluded: delta -- as expected")


def test_shared_event_has_same_engine_time() -> None:
    banner("test: tendencies in the same neighborhood see same engine-time stamp")
    rec = make_recorder([("alpha", "beta", 0.5)])
    rec.emit(EventType.STAKED, origin_id="alpha", payload={})
    rec.emit(EventType.STAKED, origin_id="beta", payload={})
    a_times = [e.engine_time for e in rec.lineage_of("alpha").events()]
    b_times = [e.engine_time for e in rec.lineage_of("beta").events()]
    # Both emissions reach both tendencies (radius=2 default), so each
    # should have both engine-time stamps.
    assert a_times == [1, 2]
    assert b_times == [1, 2]
    print(f"  alpha engine-times: {a_times}")
    print(f"  beta  engine-times: {b_times}")


def test_event_ring_evicts_oldest_on_overflow() -> None:
    banner("test: bounded ring evicts oldest on overflow")
    rec = make_recorder([("alpha", "beta", 0.5)], ring_capacity=3)
    # Emit 5 events. Ring of 3 should keep the last 3.
    for i in range(5):
        rec.emit(EventType.STAKED, origin_id="alpha", payload={"i": i})
    a_events = rec.lineage_of("alpha").events()
    assert len(a_events) == 3
    assert [e.payload["i"] for e in a_events] == [2, 3, 4]
    print(f"  emitted 5 events into ring of 3; kept i={[e.payload['i'] for e in a_events]}")


def test_allocation_window_tracks_shifts() -> None:
    banner("test: allocation rolling window tracks shifts on the origin")
    rec = make_recorder([("alpha", "beta", 0.5)])
    rec.emit_allocation_shift("alpha", new_allocation=0.4, delta=+0.1)
    rec.emit_allocation_shift("alpha", new_allocation=0.45, delta=+0.05)
    rec.emit_allocation_shift("alpha", new_allocation=0.30, delta=-0.15)

    window = rec.lineage_of("alpha").recent_allocations()
    assert window == [0.4, 0.45, 0.30], window
    # Beta sees the events too (radius covers it) but doesn't get
    # allocation_window updates (those are origin-only).
    beta_window = rec.lineage_of("beta").recent_allocations()
    assert beta_window == []
    print(f"  alpha window: {window}")
    print(f"  beta  window: {beta_window} (correctly empty -- only origin records its own allocation)")


def test_global_event_view_reconstructs_total_ordering() -> None:
    banner("test: global view reconstructs total order by engine-time")
    rec = make_recorder([
        ("alpha", "beta", 0.5),
        ("gamma", "delta", 0.5),  # disconnected component
    ])
    rec.emit(EventType.STAKED, origin_id="alpha", payload={})    # t=1
    rec.emit(EventType.STAKED, origin_id="gamma", payload={})    # t=2
    rec.emit(EventType.STAKED, origin_id="beta", payload={})     # t=3
    rec.emit(EventType.STAKED, origin_id="delta", payload={})    # t=4

    global_view = rec.global_event_view()
    times = [e.engine_time for e in global_view]
    assert times == [1, 2, 3, 4], f"expected [1,2,3,4], got {times}"
    print(f"  global view engine-times: {times}")


def test_lineage_round_trips_through_dict() -> None:
    banner("test: Lineage serializes and reloads correctly")
    rec = make_recorder([("alpha", "beta", 0.5)])
    rec.emit_allocation_shift("alpha", new_allocation=0.3, delta=-0.1)
    rec.emit(EventType.STAKED, origin_id="alpha", payload={"obs": "o42"})

    ln = rec.lineage_of("alpha")
    payload = ln.to_dict()
    restored = Lineage.from_dict(payload)

    assert restored.recent_allocations() == ln.recent_allocations()
    assert len(restored.events()) == len(ln.events())
    # Spot-check engine_time and payload survived
    for original, recreated in zip(ln.events(), restored.events()):
        assert original.engine_time == recreated.engine_time
        assert original.payload == recreated.payload
    print(f"  events serialized: {len(ln.events())}; allocations: {ln.recent_allocations()}")


def test_lineage_visibility_scales_with_sparsity() -> None:
    banner("test: lineage visibility tracks stake-weight sparsity")
    # Coherent neighborhood: alpha strongly linked to beta, weakly to others.
    rec = make_recorder([
        ("alpha", "beta",     0.8),    # strong
        ("alpha", "gamma",    0.001),  # below cutoff
        ("alpha", "delta",    0.001),
        ("beta",  "epsilon",  0.7),    # propagates from alpha via beta
    ], min_weight=0.01, radius=2)
    rec.register("zeta")  # totally isolated tendency

    rec.emit(EventType.STAKED, origin_id="alpha", payload={"obs": "o1"})

    # Visible: alpha, beta, epsilon
    # Invisible: gamma, delta (weak edge), zeta (no edge)
    visible = {tid for tid in rec.lineages
               if len(rec.lineage_of(tid).events()) > 0}
    expected = {"alpha", "beta", "epsilon"}
    assert visible == expected, f"expected {expected}, got {visible}"
    print(f"  visible (above cutoff, within radius): {sorted(visible)}")
    print(f"  invisible (weak/absent edges):         {sorted(set(rec.lineages) - visible)}")


if __name__ == "__main__":
    test_engine_clock_monotonic_and_unique()
    test_event_propagates_to_origin_only_when_isolated()
    test_event_propagates_to_strong_neighbor()
    test_radius_bounds_propagation()
    test_shared_event_has_same_engine_time()
    test_event_ring_evicts_oldest_on_overflow()
    test_allocation_window_tracks_shifts()
    test_global_event_view_reconstructs_total_ordering()
    test_lineage_round_trips_through_dict()
    test_lineage_visibility_scales_with_sparsity()
    print()
    print("All lineage smoke tests passed.")
