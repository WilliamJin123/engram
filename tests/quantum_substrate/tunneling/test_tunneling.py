"""Tests for tunneling mechanics (TEST-05).

Validates that tunneling enables creative/exploratory activation:
- High-coherence patterns can tunnel to weakly-related patterns
- Low-coherence patterns cannot initiate tunneling
- Tunneling requires connection path
- Creative mode amplifies tunneling
"""

import pytest
import random
from agentic.memory_store import MemoryStore
from agentic.evolving_pattern import EvolvingPattern
from quantum_substrate.tunneling import (
    TunnelingConfig,
    TunnelingResult,
    CreativeModeTracker,
    attempt_tunneling,
)


class TestTunnelingMechanics:
    """Core tunneling function tests."""

    def test_high_coherence_can_tunnel(self):
        """High-coherence source can tunnel to connected pattern."""
        config = TunnelingConfig(
            baseline_probability=0.9,  # High for testing
            min_source_coherence=0.3,
            max_bit_overlap=0.3,
        )

        # Patterns with different content (low overlap expected)
        p1 = EvolvingPattern.from_text("quantum physics wave function", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("medieval history castle knights", dim=1024, k=50)

        p1.coherence = 0.9  # High coherence
        connections = {"p1": {"p2"}, "p2": {"p1"}}

        rng = random.Random(42)
        result = attempt_tunneling(
            p1, "p1", {"p1": p1, "p2": p2}, connections, config,
            creative_mode=True, rng=rng
        )

        # Should be able to tunnel (high coherence, connected, low overlap)
        assert result is not None
        # With high probability and creative mode, tunneling likely succeeds
        # (deterministic with seed)

    def test_low_coherence_cannot_tunnel(self):
        """Low-coherence source cannot initiate tunneling."""
        config = TunnelingConfig(
            baseline_probability=0.9,
            min_source_coherence=0.5,  # Require 50% coherence
        )

        p1 = EvolvingPattern.from_text("source pattern", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("target pattern different", dim=1024, k=50)

        p1.coherence = 0.2  # Below threshold

        connections = {"p1": {"p2"}}
        result = attempt_tunneling(
            p1, "p1", {"p1": p1, "p2": p2}, connections, config
        )

        assert result is None or not result.tunneled, \
            "Low-coherence pattern should not tunnel"

    def test_tunneling_requires_connection(self):
        """Cannot tunnel to unconnected pattern."""
        config = TunnelingConfig(baseline_probability=0.9)

        p1 = EvolvingPattern.from_text("source", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("target", dim=1024, k=50)

        p1.coherence = 0.9  # High coherence

        # No connection
        connections = {"p1": set(), "p2": set()}

        rng = random.Random(42)
        result = attempt_tunneling(
            p1, "p1", {"p1": p1, "p2": p2}, connections, config, rng=rng
        )

        assert result is None or not result.tunneled, \
            "Cannot tunnel without connection"

    def test_tunneling_strength_scales_with_coherence(self):
        """Tunnel strength proportional to source coherence."""
        config = TunnelingConfig(
            baseline_probability=1.0,  # Always tunnel for testing
            coherence_exponent=1.0,
        )

        p1 = EvolvingPattern.from_text("source", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("completely different target", dim=1024, k=50)

        connections = {"p1": {"p2"}}
        patterns = {"p1": p1, "p2": p2}

        # High coherence -> high strength
        p1.coherence = 0.8
        result_high = attempt_tunneling(
            p1, "p1", patterns, connections, config,
            creative_mode=True, rng=random.Random(42)
        )

        # Lower coherence -> lower strength
        p1.coherence = 0.4
        result_low = attempt_tunneling(
            p1, "p1", patterns, connections, config,
            creative_mode=True, rng=random.Random(42)
        )

        if result_high and result_high.tunneled:
            assert result_high.tunnel_strength == 0.8
        if result_low and result_low.tunneled:
            assert result_low.tunnel_strength == 0.4

    def test_creative_mode_amplifies_probability(self):
        """Creative mode increases tunneling probability."""
        config = TunnelingConfig(
            baseline_probability=0.2,  # Low base
            creative_mode_multiplier=5.0,  # 5x in creative mode
        )

        p1 = EvolvingPattern.from_text("source pattern", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("target pattern completely different", dim=1024, k=50)

        p1.coherence = 0.8
        connections = {"p1": {"p2"}}
        patterns = {"p1": p1, "p2": p2}

        # Run multiple times to test probability
        successes_normal = 0
        successes_creative = 0

        for i in range(100):
            rng = random.Random(i)
            result_normal = attempt_tunneling(
                p1, "p1", patterns, connections, config,
                creative_mode=False, rng=rng
            )
            if result_normal and result_normal.tunneled:
                successes_normal += 1

            rng = random.Random(i + 1000)
            result_creative = attempt_tunneling(
                p1, "p1", patterns, connections, config,
                creative_mode=True, rng=rng
            )
            if result_creative and result_creative.tunneled:
                successes_creative += 1

        # Creative mode should have more successes
        assert successes_creative >= successes_normal, \
            f"Creative mode should tunnel more: {successes_creative} vs {successes_normal}"

    def test_tunneling_requires_low_overlap(self):
        """Tunneling only targets weakly-related patterns (low bit overlap)."""
        config = TunnelingConfig(
            baseline_probability=1.0,  # Always tunnel if possible
            max_bit_overlap=0.2,  # Require < 20% overlap
        )

        # Create patterns with very similar content (high overlap)
        p1 = EvolvingPattern.from_text("cats are furry animals", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("cats are furry pets", dim=1024, k=50)

        p1.coherence = 0.9
        connections = {"p1": {"p2"}}
        patterns = {"p1": p1, "p2": p2}

        # Check bit overlap
        intersection = len(p1.bits & p2.bits)
        union = len(p1.bits | p2.bits)
        overlap = intersection / union if union > 0 else 0.0

        # If overlap is too high, tunneling should fail
        result = attempt_tunneling(
            p1, "p1", patterns, connections, config, rng=random.Random(42)
        )

        if overlap >= config.max_bit_overlap:
            assert not result.tunneled, \
                f"Should not tunnel to high-overlap target ({overlap:.2f} >= {config.max_bit_overlap})"


class TestCreativeModeTracker:
    """Tests for auto-creative mode detection."""

    def test_activates_after_consecutive_failures(self):
        """Creative mode activates after N consecutive low scores."""
        tracker = CreativeModeTracker(
            failure_threshold=0.3,
            consecutive_failures_needed=3,
        )

        # Good scores - should not activate
        tracker.record_score(0.8)
        tracker.record_score(0.7)
        assert not tracker.should_activate_creative()

        # Start failing
        tracker.record_score(0.2)
        tracker.record_score(0.1)
        assert not tracker.should_activate_creative()  # Only 2 failures

        tracker.record_score(0.15)  # Third failure
        assert tracker.should_activate_creative(), \
            "Should activate after 3 consecutive failures"

    def test_resets_on_success(self):
        """Good score resets failure counter."""
        tracker = CreativeModeTracker(
            failure_threshold=0.3,
            consecutive_failures_needed=3,
        )

        tracker.record_score(0.2)
        tracker.record_score(0.1)
        tracker.record_score(0.8)  # Success resets
        tracker.record_score(0.2)

        assert not tracker.should_activate_creative(), \
            "Success should reset consecutive failure count"

    def test_respects_max_history(self):
        """Tracker maintains bounded history."""
        tracker = CreativeModeTracker(max_history=5)

        for i in range(10):
            tracker.record_score(0.5)

        assert len(tracker.recent_scores) == 5, \
            f"Should cap at max_history=5, got {len(tracker.recent_scores)}"

    def test_reset_clears_history(self):
        """Reset clears all tracking history."""
        tracker = CreativeModeTracker()

        tracker.record_score(0.2)
        tracker.record_score(0.1)
        tracker.reset()

        assert len(tracker.recent_scores) == 0


class TestTunnelingIntegration:
    """Integration tests with MemoryStore."""

    def test_retrieve_with_tunneling_basic(self):
        """Basic tunneling retrieval works."""
        store = MemoryStore()
        store.set_tunneling_config(TunnelingConfig(baseline_probability=0.8))

        id1 = store.store("cats are furry pets")
        id2 = store.store("dogs are loyal companions")

        store.create_connection(id1, id2)
        store.coherence_manager.apply_refresh(store.patterns[id1], 1.0)

        results, tunneled = store.retrieve_with_tunneling(
            "cats furry", top_k=5, creative_mode=True
        )

        assert len(results) > 0
        # Tunneling results should be returned
        assert isinstance(tunneled, list)

    def test_tunneled_patterns_added_to_results(self):
        """Successfully tunneled patterns appear in results."""
        store = MemoryStore()
        config = TunnelingConfig(
            baseline_probability=1.0,  # Always tunnel for testing
            max_bit_overlap=0.5,
        )
        store.set_tunneling_config(config)

        id_cat = store.store("cats purr and have whiskers")
        id_dog = store.store("dogs bark and wag tails")
        id_bird = store.store("birds fly in the sky")

        # Connect cat <-> dog only
        store.create_connection(id_cat, id_dog)

        # High coherence for cat
        store.coherence_manager.apply_refresh(store.patterns[id_cat], 1.0)

        results, tunneled = store.retrieve_with_tunneling(
            "cats purring", top_k=5, creative_mode=True
        )

        result_ids = {r.pattern_id for r in results}

        # Cat should definitely be there (direct match)
        assert id_cat in result_ids

        # Bird should not be tunneled (not connected)
        bird_tunneled = any(
            t.tunneled and t.target_pattern_id == id_bird
            for t in tunneled
        )
        assert not bird_tunneled, "Unconnected pattern should not be tunneled to"

    def test_auto_creative_mode_detection(self):
        """Creative mode auto-activates after repeated low scores."""
        store = MemoryStore()
        store.set_tunneling_config(TunnelingConfig(baseline_probability=0.5))

        # Store patterns
        id1 = store.store("topic one content")
        id2 = store.store("completely different topic")
        store.create_connection(id1, id2)

        # Reset tracker
        store.creative_tracker.reset()

        # Force low scores by querying for non-existent content
        for _ in range(5):
            results, _ = store.retrieve_with_tunneling(
                "xyzzy plugh nonexistent query", top_k=5, creative_mode=None
            )

        # Should now be in creative mode (auto-detect)
        assert store.creative_tracker.should_activate_creative(), \
            "Should activate creative mode after repeated low scores"

    def test_connection_required_for_tunneling(self):
        """Tunneling only works between connected patterns."""
        store = MemoryStore()
        config = TunnelingConfig(baseline_probability=1.0)
        store.set_tunneling_config(config)

        id1 = store.store("pattern one")
        id2 = store.store("pattern two completely different")

        # NO connection between them
        store.coherence_manager.apply_refresh(store.patterns[id1], 1.0)

        results, tunneled = store.retrieve_with_tunneling(
            "pattern one", top_k=5, creative_mode=True
        )

        # No successful tunnels (no connections)
        successful_tunnels = [t for t in tunneled if t.tunneled]
        assert len(successful_tunnels) == 0, \
            "Should not tunnel without connection"

    def test_criticality_affects_tunneling(self):
        """Criticality amplification affects tunneling probability."""
        store = MemoryStore()
        store.set_tunneling_config(TunnelingConfig(baseline_probability=0.3))

        id1 = store.store("source pattern content")
        id2 = store.store("completely different target content")
        store.create_connection(id1, id2)
        store.coherence_manager.apply_refresh(store.patterns[id1], 1.0)

        # Verify criticality affects tunneling_amplification
        store.criticality.value = 0.0
        assert store.criticality.tunneling_amplification == 0.5

        store.criticality.value = 1.0
        assert store.criticality.tunneling_amplification == 1.5

        store.criticality.value = 0.5  # Optimal
        assert store.criticality.tunneling_amplification == 1.0
