"""Test coherence decay and refresh dynamics.

Validates COHR-01, COHR-02, COHR-03 requirements:
- COHR-01: Pattern has coherence field (0-1 scalar)
- COHR-02: Coherence decays toward 0 over time without interaction
- COHR-03: Accessing a pattern refreshes its coherence

Also validates TEST-01, TEST-02 requirements:
- TEST-01: Tests validate coherence decay behavior mathematically
- TEST-02: Tests validate coherence refresh on access
"""

import pytest
import math


class TestCoherenceDecayMath:
    """Validate decay follows exponential formula (TEST-01)."""

    def test_decay_formula_exact(self):
        """Verify decay matches spec: coherence *= exp(-decay_rate * dt)."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager, CoherenceConfig

        config = CoherenceConfig(decay_rate=0.05, floor=0.01)
        cm = CoherenceManager(config)

        pattern = EvolvingPattern.from_text("test")
        pattern.coherence = 1.0
        pattern.last_access_tick = 0

        # Test at various tick values
        test_cases = [
            (0, 1.0),  # No time passed
            (10, math.exp(-0.5)),  # exp(-0.05 * 10) = exp(-0.5)
            (20, math.exp(-1.0)),  # exp(-0.05 * 20) = exp(-1.0)
            (50, math.exp(-2.5)),  # exp(-0.05 * 50) = exp(-2.5)
        ]

        for tick, expected in test_cases:
            pattern.coherence = 1.0
            pattern.last_access_tick = 0
            cm._current_tick = tick

            result = cm.compute_decayed_coherence(pattern)
            assert abs(result - expected) < 0.001, f"At tick {tick}: expected {expected:.4f}, got {result:.4f}"

    def test_decay_monotonically_decreases(self):
        """Coherence strictly decreases over time (without refresh)."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager

        cm = CoherenceManager()

        pattern = EvolvingPattern.from_text("test")
        pattern.coherence = 1.0
        pattern.last_access_tick = 0

        prev_coherence = 1.0
        for tick in range(1, 50):
            cm._current_tick = tick
            current = cm.compute_decayed_coherence(pattern)
            assert current < prev_coherence, f"Coherence should decrease: tick {tick}"
            prev_coherence = current

    def test_decay_respects_floor(self):
        """Coherence never drops below configured floor."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager, CoherenceConfig

        config = CoherenceConfig(decay_rate=0.5, floor=0.01)  # Fast decay
        cm = CoherenceManager(config)

        pattern = EvolvingPattern.from_text("test")
        pattern.coherence = 0.1
        pattern.last_access_tick = 0

        cm._current_tick = 1000  # Very long time

        result = cm.compute_decayed_coherence(pattern)
        assert result == 0.01, f"Should hit floor, got {result}"

    def test_embeddedness_slows_decay_mathematically(self):
        """Embeddedness divides effective decay rate."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager, CoherenceConfig

        config = CoherenceConfig(decay_rate=0.1, floor=0.01)
        cm = CoherenceManager(config)
        cm._current_tick = 10

        # Pattern with embeddedness 1.0 (no connections)
        p1 = EvolvingPattern.from_text("low")
        p1.coherence = 1.0
        p1.last_access_tick = 0
        p1.connection_count = 0  # embeddedness = 1.0

        # Pattern with embeddedness 2.0 (10 connections)
        p2 = EvolvingPattern.from_text("high")
        p2.coherence = 1.0
        p2.last_access_tick = 0
        p2.connection_count = 10  # embeddedness = 2.0

        d1 = cm.compute_decayed_coherence(p1)
        d2 = cm.compute_decayed_coherence(p2)

        # p1: exp(-0.1 * 10 / 1.0) = exp(-1.0) ~ 0.368
        # p2: exp(-0.1 * 10 / 2.0) = exp(-0.5) ~ 0.607
        assert abs(d1 - math.exp(-1.0)) < 0.001
        assert abs(d2 - math.exp(-0.5)) < 0.001
        assert d2 > d1, "Higher embeddedness should decay slower"


class TestCoherenceRefresh:
    """Validate refresh on access (TEST-02, COHR-03)."""

    def test_store_refreshes_coherence(self):
        """Storing a pattern sets its coherence near 1.0."""
        from agentic.memory_store import MemoryStore

        store = MemoryStore()
        pid = store.store("test pattern")

        pattern = store.patterns[pid]
        assert pattern.coherence > 0.9, f"New pattern should have high coherence, got {pattern.coherence}"

    def test_retrieve_refreshes_matched_patterns(self):
        """Retrieve refreshes patterns proportional to match score."""
        from agentic.memory_store import MemoryStore

        store = MemoryStore()

        # Store a pattern
        pid = store.store("the quick brown fox")

        # Decay it manually by advancing tick without accessing
        for _ in range(20):
            store.coherence_manager.advance_tick()

        # Check it decayed
        decayed = store.get_effective_coherence(pid)
        assert decayed < 0.5, f"Should have decayed, got {decayed}"

        # Retrieve it (should refresh)
        results = store.retrieve("quick fox")

        # Check it refreshed
        if results and results[0].pattern_id == pid:
            refreshed = store.patterns[pid].coherence
            assert refreshed > decayed, f"Should have refreshed from {decayed} to {refreshed}"

    def test_refresh_capped_at_one(self):
        """Refresh cannot exceed 1.0 even with repeated access."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager

        cm = CoherenceManager()

        pattern = EvolvingPattern.from_text("test")
        pattern.coherence = 0.99

        # Multiple refreshes
        for _ in range(10):
            cm.apply_refresh(pattern, activation_strength=1.0)

        assert pattern.coherence <= 1.0, f"Coherence exceeded cap: {pattern.coherence}"

    def test_activation_strength_affects_refresh_amount(self):
        """Higher activation = more coherence refresh."""
        from agentic.evolving_pattern import EvolvingPattern
        from quantum_substrate.coherence import CoherenceManager

        cm = CoherenceManager()

        # Two identical patterns
        p_weak = EvolvingPattern.from_text("test")
        p_weak.coherence = 0.5

        p_strong = EvolvingPattern.from_text("test")
        p_strong.coherence = 0.5

        cm.apply_refresh(p_weak, activation_strength=0.1)
        cm.apply_refresh(p_strong, activation_strength=1.0)

        assert p_strong.coherence > p_weak.coherence, \
            f"Strong activation should refresh more: weak={p_weak.coherence}, strong={p_strong.coherence}"


class TestOperationTicks:
    """Validate tick advancement on operations."""

    def test_store_advances_tick(self):
        """Each store advances tick by 1."""
        from agentic.memory_store import MemoryStore

        store = MemoryStore()
        assert store.current_tick == 0

        store.store("first")
        assert store.current_tick == 1

        store.store("second")
        assert store.current_tick == 2

    def test_retrieve_advances_tick(self):
        """Each retrieve advances tick by 1."""
        from agentic.memory_store import MemoryStore

        store = MemoryStore()
        store.store("test")
        initial_tick = store.current_tick

        store.retrieve("test")
        assert store.current_tick == initial_tick + 1


class TestCoactivationConnection:
    """Test coactivation updates connection_count."""

    def test_coactivation_increments_connections(self):
        """Coactivation increases connection_count for embeddedness."""
        from agentic.evolving_pattern import EvolvingPattern
        from agentic.coactivation import coactivate

        p1 = EvolvingPattern.from_text("pattern one")
        p2 = EvolvingPattern.from_text("pattern two")

        assert p1.connection_count == 0
        assert p2.connection_count == 0

        coactivate([p1, p2])

        # Bidirectional = both should have connections
        assert p1.connection_count >= 1
        assert p2.connection_count >= 1

    def test_coactivation_increases_embeddedness(self):
        """Coactivation -> higher connection_count -> higher embeddedness -> slower decay."""
        from agentic.evolving_pattern import EvolvingPattern
        from agentic.coactivation import coactivate
        from quantum_substrate.coherence import CoherenceManager

        cm = CoherenceManager()

        # Isolated pattern
        p_isolated = EvolvingPattern.from_text("isolated")
        p_isolated.coherence = 1.0
        p_isolated.last_access_tick = 0

        # Connected pattern
        p_connected = EvolvingPattern.from_text("connected")
        p_connected.coherence = 1.0
        p_connected.last_access_tick = 0

        # Coactivate connected pattern multiple times
        other = EvolvingPattern.from_text("other")
        for _ in range(10):
            coactivate([p_connected, other])

        assert p_connected.connection_count >= 10
        assert p_connected.embeddedness > p_isolated.embeddedness

        # Check decay difference
        cm._current_tick = 20

        d_isolated = cm.compute_decayed_coherence(p_isolated)
        d_connected = cm.compute_decayed_coherence(p_connected)

        assert d_connected > d_isolated, \
            f"Connected pattern should decay slower: isolated={d_isolated}, connected={d_connected}"


class TestDecayThenRefreshOrder:
    """Validate correct operation order: decay first, then refresh."""

    def test_decay_applies_before_refresh(self):
        """Operation order: decay all -> refresh activated."""
        from agentic.memory_store import MemoryStore

        store = MemoryStore()

        # Store first pattern
        pid1 = store.store("first pattern")

        # Manually advance time without operations (simulating passage)
        store.coherence_manager._current_tick = 50

        # Store second pattern - should decay first, then refresh second
        pid2 = store.store("second pattern")

        # First pattern should have decayed (was not refreshed by second store)
        p1_coherence = store.patterns[pid1].coherence
        # Note: It was decayed during store of pid2

        # Second pattern should be refreshed (near 1.0)
        p2_coherence = store.patterns[pid2].coherence

        assert p2_coherence > p1_coherence, \
            f"New pattern should have higher coherence: p1={p1_coherence}, p2={p2_coherence}"


class TestIntegration:
    """End-to-end coherence behavior tests."""

    def test_frequently_accessed_stays_coherent(self):
        """Patterns accessed often maintain high coherence."""
        from agentic.memory_store import MemoryStore
        from quantum_substrate.coherence import CoherenceConfig

        # Use slower decay for this test to clearly show refresh effect
        config = CoherenceConfig(decay_rate=0.02, floor=0.01)
        store = MemoryStore(coherence_config=config)

        # Store pattern
        pid = store.store("frequently accessed")

        # Access it repeatedly with exact match query
        for _ in range(10):
            store.retrieve("frequently accessed")

        coherence = store.patterns[pid].coherence
        assert coherence > 0.7, f"Frequently accessed should stay coherent: {coherence}"

    def test_ignored_pattern_decays(self):
        """Patterns not accessed decay toward floor."""
        from agentic.memory_store import MemoryStore
        from quantum_substrate.coherence import CoherenceConfig

        config = CoherenceConfig(decay_rate=0.2, floor=0.01)  # Fast decay for test
        store = MemoryStore(coherence_config=config)

        # Store two patterns
        pid_accessed = store.store("accessed")
        pid_ignored = store.store("ignored")

        # Access only the first one many times
        for _ in range(20):
            store.retrieve("accessed")

        c_accessed = store.patterns[pid_accessed].coherence
        c_ignored = store.patterns[pid_ignored].coherence

        assert c_accessed > c_ignored, \
            f"Accessed should be higher: accessed={c_accessed}, ignored={c_ignored}"
        assert c_ignored < 0.3, f"Ignored should have decayed significantly: {c_ignored}"
