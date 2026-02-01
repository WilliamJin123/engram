"""Test coherence field and manager primitives.

These tests validate the basic building blocks before integration.
"""

import pytest
import math


class TestEvolvingPatternCoherence:
    """Test coherence field on EvolvingPattern."""

    def test_new_pattern_has_full_coherence(self):
        """New patterns start with coherence 1.0."""
        from agentic.evolving_pattern import EvolvingPattern

        pattern = EvolvingPattern.from_text("test pattern")

        assert pattern.coherence == 1.0
        assert pattern.last_access_tick == 0
        assert pattern.connection_count == 0

    def test_embeddedness_starts_at_one(self):
        """New patterns have embeddedness 1.0 (no connections)."""
        from agentic.evolving_pattern import EvolvingPattern

        pattern = EvolvingPattern.from_text("test")
        assert pattern.embeddedness == 1.0

    def test_embeddedness_increases_with_connections(self):
        """More connections = higher embeddedness."""
        from agentic.evolving_pattern import EvolvingPattern

        pattern = EvolvingPattern.from_text("test")
        pattern.connection_count = 10

        # embeddedness = 1 + 0.1 * 10 = 2.0
        assert pattern.embeddedness == 2.0

    def test_clamp_coherence_enforces_bounds(self):
        """clamp_coherence keeps value between floor and 1.0."""
        from agentic.evolving_pattern import EvolvingPattern

        pattern = EvolvingPattern.from_text("test")

        # Test upper bound
        pattern.coherence = 1.5
        pattern.clamp_coherence()
        assert pattern.coherence == 1.0

        # Test lower bound
        pattern.coherence = 0.001
        pattern.clamp_coherence(floor=0.01)
        assert pattern.coherence == 0.01


class TestCoherenceManager:
    """Test CoherenceManager tick tracking and decay."""

    def test_initial_tick_is_zero(self):
        """Manager starts at tick 0."""
        from quantum_substrate.coherence import CoherenceManager

        cm = CoherenceManager()
        assert cm.current_tick == 0

    def test_advance_tick_increments(self):
        """advance_tick increases counter by 1."""
        from quantum_substrate.coherence import CoherenceManager

        cm = CoherenceManager()
        assert cm.current_tick == 0

        new_tick = cm.advance_tick()
        assert new_tick == 1
        assert cm.current_tick == 1

        cm.advance_tick()
        assert cm.current_tick == 2

    def test_reset_clears_tick(self):
        """reset() returns tick to 0."""
        from quantum_substrate.coherence import CoherenceManager

        cm = CoherenceManager()
        cm.advance_tick()
        cm.advance_tick()
        assert cm.current_tick == 2

        cm.reset()
        assert cm.current_tick == 0


class TestDecayCalculation:
    """Test decay math follows exponential formula."""

    def test_no_decay_at_same_tick(self):
        """No decay when last_access_tick equals current_tick."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager

        pattern = EvolvingPattern.from_text("test")
        pattern.coherence = 0.8
        pattern.last_access_tick = 5

        cm = CoherenceManager()
        cm._current_tick = 5  # Same tick

        decayed = cm.compute_decayed_coherence(pattern)
        assert decayed == 0.8  # No change

    def test_exponential_decay_formula(self):
        """Decay follows coherence * exp(-rate * dt / embeddedness)."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager, CoherenceConfig

        config = CoherenceConfig(decay_rate=0.1, floor=0.01)
        cm = CoherenceManager(config)

        pattern = EvolvingPattern.from_text("test")
        pattern.coherence = 1.0
        pattern.last_access_tick = 0

        cm._current_tick = 10

        # Expected: 1.0 * exp(-0.1 * 10 / 1.0) = exp(-1) ~ 0.368
        decayed = cm.compute_decayed_coherence(pattern)
        expected = math.exp(-1.0)

        assert abs(decayed - expected) < 0.001

    def test_embeddedness_slows_decay(self):
        """Higher embeddedness = slower decay."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager, CoherenceConfig

        config = CoherenceConfig(decay_rate=0.1, floor=0.01)
        cm = CoherenceManager(config)

        # Low embeddedness pattern
        p_low = EvolvingPattern.from_text("low")
        p_low.coherence = 1.0
        p_low.last_access_tick = 0
        p_low.connection_count = 0  # embeddedness = 1.0

        # High embeddedness pattern
        p_high = EvolvingPattern.from_text("high")
        p_high.coherence = 1.0
        p_high.last_access_tick = 0
        p_high.connection_count = 10  # embeddedness = 2.0

        cm._current_tick = 10

        decayed_low = cm.compute_decayed_coherence(p_low)
        decayed_high = cm.compute_decayed_coherence(p_high)

        # High embeddedness should decay slower (higher coherence)
        assert decayed_high > decayed_low

    def test_decay_respects_floor(self):
        """Coherence never drops below floor."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager, CoherenceConfig

        config = CoherenceConfig(decay_rate=0.5, floor=0.01)
        cm = CoherenceManager(config)

        pattern = EvolvingPattern.from_text("test")
        pattern.coherence = 0.1
        pattern.last_access_tick = 0

        cm._current_tick = 100  # Long time, should hit floor

        decayed = cm.compute_decayed_coherence(pattern)
        assert decayed == 0.01


class TestRefreshBehavior:
    """Test coherence refresh on access."""

    def test_refresh_increases_coherence(self):
        """Refresh increases coherence toward cap."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager

        cm = CoherenceManager()

        pattern = EvolvingPattern.from_text("test")
        pattern.coherence = 0.5

        old_coherence = pattern.coherence
        cm.apply_refresh(pattern, activation_strength=1.0)

        assert pattern.coherence > old_coherence

    def test_refresh_updates_last_access_tick(self):
        """Refresh updates last_access_tick to current_tick."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager

        cm = CoherenceManager()
        cm._current_tick = 5

        pattern = EvolvingPattern.from_text("test")
        pattern.last_access_tick = 0

        cm.apply_refresh(pattern)

        assert pattern.last_access_tick == 5

    def test_refresh_capped_at_one(self):
        """Refresh cannot exceed 1.0."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager

        cm = CoherenceManager()

        pattern = EvolvingPattern.from_text("test")
        pattern.coherence = 0.99

        cm.apply_refresh(pattern, activation_strength=1.0)

        assert pattern.coherence <= 1.0

    def test_refresh_proportional_to_activation(self):
        """Stronger activation = more refresh."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager

        cm = CoherenceManager()

        p_weak = EvolvingPattern.from_text("weak")
        p_weak.coherence = 0.5

        p_strong = EvolvingPattern.from_text("strong")
        p_strong.coherence = 0.5

        cm.apply_refresh(p_weak, activation_strength=0.1)
        cm.apply_refresh(p_strong, activation_strength=1.0)

        assert p_strong.coherence > p_weak.coherence


class TestDecayThenRefresh:
    """Test combined decay-then-refresh operation order."""

    def test_decay_before_refresh(self):
        """decay_then_refresh applies decay first, then refresh."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager, CoherenceConfig

        config = CoherenceConfig(decay_rate=0.1, floor=0.01)
        cm = CoherenceManager(config)

        pattern = EvolvingPattern.from_text("test")
        pattern.coherence = 1.0
        pattern.last_access_tick = 0

        cm._current_tick = 10

        # If decay happened: ~0.368, then refresh adds some back
        result = cm.decay_then_refresh(pattern, activation_strength=1.0)

        # Should be less than 1.0 (decayed) but more than 0.368 (refreshed)
        assert 0.3 < result < 1.0
        assert pattern.last_access_tick == 10
