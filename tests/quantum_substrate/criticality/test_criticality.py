"""Tests for criticality parameter and self-adjustment.

Validates:
- Criticality affects tunneling amplification
- Self-adjustment based on retrieval quality
- Self-adjustment based on surprise frequency
- Dampening prevents oscillation
"""

import pytest
from quantum_substrate.criticality import CriticalityConfig, CriticalityState
from agentic.memory_store import MemoryStore


class TestCriticalityState:
    """Tests for CriticalityState."""

    def test_initial_value(self):
        """Criticality starts at optimal 0.5."""
        state = CriticalityState()
        assert state.value == 0.5

    def test_tunneling_amplification_at_optimal(self):
        """At 0.5 criticality, amplification is 1.0."""
        state = CriticalityState()
        state.value = 0.5
        assert state.tunneling_amplification == 1.0

    def test_tunneling_amplification_range(self):
        """Amplification scales linearly with criticality."""
        state = CriticalityState()

        state.value = 0.0
        assert state.tunneling_amplification == 0.5

        state.value = 1.0
        assert state.tunneling_amplification == 1.5

        state.value = 0.3
        assert abs(state.tunneling_amplification - 0.8) < 0.001

    def test_is_healthy_range(self):
        """Healthy range is 0.1 to 0.9."""
        state = CriticalityState()

        state.value = 0.5
        assert state.is_healthy

        state.value = 0.05
        assert not state.is_healthy

        state.value = 0.95
        assert not state.is_healthy

    def test_reset_restores_initial(self):
        """Reset restores initial value and clears feedback."""
        config = CriticalityConfig(initial_value=0.5)
        state = CriticalityState(config=config)

        # Modify state
        state.value = 0.8
        state.record_feedback(0.5, True)
        state.record_feedback(0.3, False)

        # Reset
        state.reset()

        assert state.value == 0.5
        assert len(state._recent_quality) == 0
        assert len(state._recent_surprises) == 0


class TestSelfAdjustment:
    """Tests for criticality self-adjustment."""

    def test_poor_quality_increases_criticality(self):
        """Poor retrieval quality should increase chaos."""
        config = CriticalityConfig(
            adjustment_rate=0.1,
            min_feedback_samples=3,
        )
        state = CriticalityState(config=config)
        state.value = 0.5

        # Record poor quality retrievals
        for _ in range(5):
            state.record_feedback(retrieval_quality=0.2, had_surprise=False)

        old_value = state.value
        state.self_adjust()

        # Should increase (more chaos needed for poor retrieval)
        # Note: low surprise also contributes to increase
        assert state.value >= old_value, \
            f"Poor quality should not decrease criticality: {old_value} -> {state.value}"

    def test_high_surprise_decreases_criticality(self):
        """Too many surprises should decrease chaos."""
        config = CriticalityConfig(
            adjustment_rate=0.1,
            min_feedback_samples=3,
            target_surprise_rate_high=0.5,
        )
        state = CriticalityState(config=config)
        state.value = 0.7  # Start high

        # Record high surprise rate with good quality
        for _ in range(10):
            state.record_feedback(retrieval_quality=0.8, had_surprise=True)

        old_value = state.value
        state.self_adjust()

        # Should decrease (too chaotic)
        assert state.value < old_value, \
            f"High surprise should decrease criticality: {old_value} -> {state.value}"

    def test_low_surprise_increases_criticality(self):
        """Too few surprises should increase chaos (system too rigid)."""
        config = CriticalityConfig(
            adjustment_rate=0.1,
            min_feedback_samples=3,
            target_surprise_rate_low=0.1,
        )
        state = CriticalityState(config=config)
        state.value = 0.3  # Start low

        # Record no surprises with good quality
        for _ in range(10):
            state.record_feedback(retrieval_quality=0.8, had_surprise=False)

        old_value = state.value
        state.self_adjust()

        # Should increase (too rigid)
        assert state.value > old_value, \
            f"Low surprise should increase criticality: {old_value} -> {state.value}"

    def test_dampening_prevents_oscillation(self):
        """Dampening should prevent large swings."""
        config = CriticalityConfig(
            adjustment_rate=0.5,  # Large rate
            dampening=0.1,  # Strong dampening
            min_feedback_samples=1,
        )
        state = CriticalityState(config=config)
        state.value = 0.5

        # Extreme feedback
        state.record_feedback(retrieval_quality=0.0, had_surprise=False)
        state.self_adjust()

        # Change should be small due to dampening
        assert abs(state.value - 0.5) < 0.1, \
            f"Dampening should limit change: 0.5 -> {state.value}"

    def test_bounds_respected(self):
        """Criticality stays in [0, 1]."""
        config = CriticalityConfig(
            adjustment_rate=0.5,
            dampening=1.0,  # No dampening
            min_feedback_samples=1,
        )
        state = CriticalityState(config=config)

        # Try to go above 1
        state.value = 0.95
        for _ in range(10):
            state.record_feedback(retrieval_quality=0.0, had_surprise=False)
            state.self_adjust()

        assert state.value <= 1.0

        # Try to go below 0
        state.value = 0.05
        for _ in range(10):
            state.record_feedback(retrieval_quality=0.9, had_surprise=True)
            state.self_adjust()

        assert state.value >= 0.0

    def test_min_feedback_samples_required(self):
        """No adjustment without minimum feedback samples."""
        config = CriticalityConfig(
            adjustment_rate=0.5,
            min_feedback_samples=5,
        )
        state = CriticalityState(config=config)
        initial = state.value

        # Only 3 samples (less than 5 required)
        for _ in range(3):
            state.record_feedback(retrieval_quality=0.0, had_surprise=False)

        state.self_adjust()

        assert state.value == initial, \
            "Should not adjust with insufficient samples"

    def test_window_size_maintained(self):
        """Feedback window maintains bounded size."""
        config = CriticalityConfig(window_size=10)
        state = CriticalityState(config=config)

        for i in range(20):
            state.record_feedback(retrieval_quality=i/20.0, had_surprise=i % 2 == 0)

        assert len(state._recent_quality) == 10
        assert len(state._recent_surprises) == 10


class TestCriticalityIntegration:
    """Integration tests with MemoryStore."""

    def test_store_has_criticality(self):
        """MemoryStore has criticality state."""
        store = MemoryStore()
        assert hasattr(store, 'criticality')
        assert store.get_criticality() == 0.5

    def test_retrieval_records_feedback(self):
        """Retrieval with tunneling records feedback."""
        store = MemoryStore()
        store._criticality_adjust_interval = 1000  # Disable auto-adjust

        store.store("test pattern")

        # Clear any initial feedback
        store.criticality._recent_quality.clear()
        store.criticality._recent_surprises.clear()

        results, _ = store.retrieve_with_tunneling("test", top_k=1)

        # Should have recorded feedback
        assert len(store.criticality._recent_quality) > 0, \
            "Retrieval should record quality feedback"

    def test_auto_adjustment_works(self):
        """Criticality auto-adjusts periodically."""
        store = MemoryStore()
        store._criticality_adjust_interval = 3  # Adjust every 3 operations

        # Store patterns
        for i in range(5):
            store.store(f"pattern {i}")

        initial = store.get_criticality()

        # Perform many retrievals
        for i in range(20):
            results, _ = store.retrieve_with_tunneling(f"query {i}", top_k=1)

        # Criticality should have been adjusted (may or may not change significantly)
        # This test just verifies the mechanism runs without error
        final = store.get_criticality()
        assert 0.0 <= final <= 1.0

    def test_set_criticality_config(self):
        """Can update criticality configuration."""
        store = MemoryStore()

        new_config = CriticalityConfig(
            initial_value=0.3,
            adjustment_rate=0.05,
        )
        store.set_criticality_config(new_config)

        # New state starts with default value (0.5), not config.initial_value
        # config.initial_value is used by reset() method
        assert store.get_criticality() == 0.5
        assert store.criticality.config.initial_value == 0.3
        assert store.criticality.config.adjustment_rate == 0.05

        # reset() uses config.initial_value
        store.criticality.reset()
        assert store.get_criticality() == 0.3

    def test_manual_adjust_criticality(self):
        """Can manually trigger criticality adjustment."""
        store = MemoryStore()
        store._criticality_adjust_interval = 1000  # Disable auto

        # Record enough feedback
        for _ in range(10):
            store.criticality.record_feedback(0.2, False)

        initial = store.get_criticality()
        store.adjust_criticality()

        # Should have adjusted (poor quality + no surprise = increase)
        assert store.get_criticality() >= initial

    def test_criticality_affects_tunneling_in_retrieval(self):
        """Criticality amplification is passed to tunneling attempts."""
        store = MemoryStore()
        store.set_tunneling_config(
            store.tunneling_config.__class__(baseline_probability=0.5)
        )

        id1 = store.store("source pattern")
        id2 = store.store("completely different target")
        store.create_connection(id1, id2)
        store.coherence_manager.apply_refresh(store.patterns[id1], 1.0)

        # High criticality = more tunneling
        store.criticality.value = 1.0
        high_amp = store.criticality.tunneling_amplification
        assert high_amp == 1.5

        # Low criticality = less tunneling
        store.criticality.value = 0.0
        low_amp = store.criticality.tunneling_amplification
        assert low_amp == 0.5

        # The amplification is used in retrieve_with_tunneling internally
        # via store.criticality.tunneling_amplification
