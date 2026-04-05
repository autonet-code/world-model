"""Test the novelty-attention curve integration."""

from world_model import (
    Tendency,
    AgentSet,
    NoveltyAttentionCurve,
    AttentionState,
    EXPLORER_CURVE,
    BALANCED_CURVE,
    CONSERVATIVE_CURVE,
)


def test_curve_basic():
    """Test basic curve behavior."""
    curve = NoveltyAttentionCurve(midpoint=0.5, steepness=10)

    # At midpoint, capture should be ~0.5
    assert 0.45 < curve.capture(0.5) < 0.55, "Midpoint should give ~0.5 capture"

    # Low novelty = low capture
    assert curve.capture(0.1) < 0.1, "Low novelty should have low capture"

    # High novelty = high capture
    assert curve.capture(0.9) > 0.9, "High novelty should have high capture"

    print("Basic curve tests passed")


def test_effective_allocations():
    """Test that allocations shift under novelty."""
    agent_set = AgentSet()
    state = AttentionState(agent_set=agent_set, curve=BALANCED_CURVE)

    # At zero novelty, allocations should match base
    state.update_novelty(0.0)
    base = state.base_allocations
    effective = state.effective_allocations

    for tendency in Tendency:
        diff = abs(base[tendency] - effective[tendency])
        assert diff < 0.01, f"{tendency}: effective should match base at 0 novelty"

    # At high novelty, CURIOSITY should dominate
    state.update_novelty(0.95)
    effective_high = state.effective_allocations

    assert effective_high[Tendency.CURIOSITY] > base[Tendency.CURIOSITY], \
        "CURIOSITY should increase under high novelty"

    # Other tendencies should decrease
    for tendency in Tendency:
        if tendency != Tendency.CURIOSITY:
            assert effective_high[tendency] < base[tendency], \
                f"{tendency} should decrease under high novelty"

    print("Effective allocation tests passed")


def test_presets():
    """Test preset curve profiles."""
    agent_set = AgentSet()

    # Explorer: novelty captures attention earlier
    explorer_state = AttentionState(agent_set=agent_set, curve=EXPLORER_CURVE)
    explorer_state.update_novelty(0.4)

    # Conservative: novelty captures attention later
    conservative_state = AttentionState(agent_set=agent_set, curve=CONSERVATIVE_CURVE)
    conservative_state.update_novelty(0.4)

    # At moderate novelty (0.4), explorer should have higher capture
    assert explorer_state.novelty_capture > conservative_state.novelty_capture, \
        "Explorer should capture more attention at moderate novelty"

    print("Preset curve tests passed")


def demo_curve_behavior():
    """Demonstrate how the curve affects attention across novelty levels."""
    print("\n" + "=" * 60)
    print("NOVELTY-ATTENTION CURVE DEMONSTRATION")
    print("=" * 60)

    agent_set = AgentSet()
    state = AttentionState(agent_set=agent_set, curve=BALANCED_CURVE)

    print(f"\nBase allocations (no novelty):")
    for tendency, alloc in sorted(state.base_allocations.items(), key=lambda x: -x[1]):
        print(f"  {tendency.value:12}: {alloc:.1%}")

    novelty_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\nHow allocations change with novelty:")
    print("-" * 60)

    for novelty in novelty_levels:
        state.update_novelty(novelty)
        capture = state.novelty_capture
        curiosity = state.effective_allocations[Tendency.CURIOSITY]
        dominant = state.dominant_tendency

        print(f"\nNovelty = {novelty:.2f} (capture = {capture:.1%})")
        print(f"  CURIOSITY: {state.base_allocations[Tendency.CURIOSITY]:.1%} -> {curiosity:.1%}")
        print(f"  Dominant: {dominant.value}")
        print(f"  {state.describe()}")


def demo_profile_comparison():
    """Compare how different agent profiles respond to novelty."""
    print("\n" + "=" * 60)
    print("PROFILE COMPARISON AT HIGH NOVELTY (0.8)")
    print("=" * 60)

    profiles = [
        ("Explorer", EXPLORER_CURVE),
        ("Balanced", BALANCED_CURVE),
        ("Conservative", CONSERVATIVE_CURVE),
    ]

    agent_set = AgentSet()

    for name, curve in profiles:
        state = AttentionState(agent_set=agent_set, curve=curve)
        state.update_novelty(0.8)

        print(f"\n{name} profile:")
        print(f"  Novelty capture: {state.novelty_capture:.1%}")
        print(f"  CURIOSITY allocation: {state.effective_allocations[Tendency.CURIOSITY]:.1%}")
        print(f"  Dominant tendency: {state.dominant_tendency.value}")


if __name__ == "__main__":
    # Run tests
    test_curve_basic()
    test_effective_allocations()
    test_presets()

    print("\nAll tests passed!")

    # Run demos
    demo_curve_behavior()
    demo_profile_comparison()
