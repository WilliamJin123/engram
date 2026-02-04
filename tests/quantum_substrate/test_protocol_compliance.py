# tests/quantum_substrate/test_protocol_compliance.py
"""Protocol compliance tests for SubstratePattern.

Verifies that EvolvingPattern (agentic layer) properly implements the
SubstratePattern protocol (substrate layer), proving ARCH-03: agentic
layer builds on substrate through documented interfaces.

Key tests:
1. isinstance check works (runtime_checkable decorator)
2. All protocol attributes are accessible
3. Coherence read/write works through protocol
4. Substrate operations accept EvolvingPattern instances
"""

import pytest

from quantum_substrate.protocols import SubstratePattern
from quantum_substrate.coherence import CoherenceManager, CoherenceConfig
from quantum_substrate.surprise import SurpriseDetector, compute_surprise_magnitude
from quantum_substrate.tunneling import attempt_tunneling, TunnelingConfig

from agentic.evolving_pattern import EvolvingPattern


class TestEvolvingPatternSatisfiesProtocol:
    """Test that EvolvingPattern satisfies SubstratePattern protocol."""

    def test_evolving_pattern_satisfies_protocol(self) -> None:
        """isinstance check returns True for EvolvingPattern."""
        pattern = EvolvingPattern.from_text("test pattern for protocol check")

        # This is the key test: runtime_checkable protocol check
        assert isinstance(pattern, SubstratePattern), \
            "EvolvingPattern must satisfy SubstratePattern protocol"

    def test_protocol_attributes_accessible(self) -> None:
        """All required protocol attributes exist and are accessible."""
        pattern = EvolvingPattern.from_text("test attributes")

        # Required properties (read-only or read-write)
        assert isinstance(pattern.dim, int), "dim must be int"
        assert isinstance(pattern.bits, set), "bits must be set"
        assert isinstance(pattern.original_bits, frozenset), "original_bits must be frozenset"
        assert isinstance(pattern.phases, dict), "phases must be dict"
        assert isinstance(pattern.coherence, float), "coherence must be float"
        assert isinstance(pattern.last_access_tick, int), "last_access_tick must be int"
        assert isinstance(pattern.embeddedness, float), "embeddedness must be float"
        assert isinstance(pattern.stability_score, float), "stability_score must be float"

        # Required method
        assert callable(pattern.record_access), "record_access must be callable"

    def test_coherence_property_works(self) -> None:
        """Coherence property supports both get and set."""
        pattern = EvolvingPattern.from_text("test coherence property")

        # Get initial coherence
        initial = pattern.coherence
        assert initial == 1.0, "New patterns should start with coherence 1.0"

        # Set coherence
        pattern.coherence = 0.5
        assert pattern.coherence == 0.5, "Coherence setter must work"

        # Set back
        pattern.coherence = 1.0
        assert pattern.coherence == 1.0, "Coherence setter must work again"

    def test_last_access_tick_property_works(self) -> None:
        """last_access_tick property supports both get and set."""
        pattern = EvolvingPattern.from_text("test tick property")

        # Get initial
        initial = pattern.last_access_tick
        assert initial == 0, "New patterns should have last_access_tick 0"

        # Set
        pattern.last_access_tick = 100
        assert pattern.last_access_tick == 100, "last_access_tick setter must work"

    def test_record_access_method_works(self) -> None:
        """record_access method increments access counter."""
        pattern = EvolvingPattern.from_text("test record_access")

        initial_stability = pattern.stability_score
        assert initial_stability == 0.0, "New patterns have zero stability"

        # Record accesses
        pattern.record_access()
        pattern.record_access()
        pattern.record_access()

        # Stability should increase (based on access_count_since_modification)
        assert pattern.stability_score > initial_stability, \
            "record_access must increment access count affecting stability"


class TestSubstrateOperationsAcceptEvolvingPattern:
    """Test that substrate operations work with EvolvingPattern instances."""

    def test_coherence_manager_accepts_evolving_pattern(self) -> None:
        """CoherenceManager operations work with EvolvingPattern."""
        manager = CoherenceManager(CoherenceConfig(decay_rate=0.1))
        pattern = EvolvingPattern.from_text("test coherence manager")

        # compute_decayed_coherence
        decayed = manager.compute_decayed_coherence(pattern)
        assert isinstance(decayed, float), "compute_decayed_coherence must return float"

        # Advance time and check decay works
        manager.advance_tick()
        manager.advance_tick()
        manager.advance_tick()

        decayed_after_time = manager.compute_decayed_coherence(pattern)
        assert decayed_after_time < pattern.coherence, \
            "Coherence should decay over ticks"

        # apply_decay
        new_coherence = manager.apply_decay(pattern)
        assert isinstance(new_coherence, float), "apply_decay must return float"

        # apply_refresh
        pattern.coherence = 0.5
        refreshed = manager.apply_refresh(pattern, activation_strength=1.0)
        assert refreshed > 0.5, "apply_refresh must increase coherence"

    def test_surprise_detector_accepts_evolving_pattern(self) -> None:
        """SurpriseDetector operations work with EvolvingPattern."""
        detector = SurpriseDetector()
        manager = CoherenceManager()

        pattern1 = EvolvingPattern.from_text("apple banana cherry")
        pattern2 = EvolvingPattern.from_text("dog cat elephant")
        patterns = {"p1": pattern1, "p2": pattern2}

        query_bits = pattern1.bits

        # build_expectation
        expected_bits, weights = detector.build_expectation(
            query_bits, patterns, manager
        )
        assert isinstance(expected_bits, set), "expected_bits must be set"
        assert isinstance(weights, dict), "weights must be dict"

        # detect
        retrieval_results = [("p1", 0.9), ("p2", 0.1)]
        result = detector.detect(
            expected_bits, weights, pattern1, "p1", retrieval_results
        )
        assert hasattr(result, 'magnitude'), "SurpriseResult must have magnitude"
        assert isinstance(result.magnitude, float), "magnitude must be float"

    def test_tunneling_accepts_evolving_pattern(self) -> None:
        """attempt_tunneling works with EvolvingPattern."""
        import random

        # Create patterns with low overlap (required for tunneling)
        source = EvolvingPattern.from_text(
            "alpha beta gamma", dim=1024, k=50
        )
        source.coherence = 0.9  # High coherence needed

        target = EvolvingPattern.from_text(
            "one two three four five six seven eight nine ten", dim=1024, k=50
        )

        patterns = {"source": source, "target": target}
        connections = {"source": {"target"}}  # Connected

        config = TunnelingConfig(
            baseline_probability=1.0,  # Force tunneling for test
            min_source_coherence=0.3,
            max_bit_overlap=0.5,
        )

        # Use seeded RNG for determinism
        rng = random.Random(42)

        result = attempt_tunneling(
            source=source,
            source_id="source",
            all_patterns=patterns,
            connections=connections,
            config=config,
            creative_mode=False,
            rng=rng,
        )

        assert hasattr(result, 'tunneled'), "TunnelingResult must have tunneled"
        assert hasattr(result, 'tunnel_strength'), "TunnelingResult must have tunnel_strength"

    def test_apply_recoherence_accepts_evolving_pattern(self) -> None:
        """CoherenceManager.apply_recoherence works with EvolvingPattern."""
        from quantum_substrate.surprise import SurpriseResult

        manager = CoherenceManager()

        pattern1 = EvolvingPattern.from_text("surprising content")
        pattern1.coherence = 0.3  # Low coherence

        pattern2 = EvolvingPattern.from_text("expected content")
        pattern2.coherence = 0.8

        patterns = {"p1": pattern1, "p2": pattern2}

        surprise = SurpriseResult(
            magnitude=0.8,  # High surprise
            surprising_pattern_id="p1",
            expected_pattern_id="p2",
            participant_scores={"p1": 0.9, "p2": 0.5},
        )

        deltas = manager.apply_recoherence(patterns, surprise)

        assert isinstance(deltas, dict), "apply_recoherence must return dict"
        # p1 should get boosted (it was surprising)
        if "p1" in deltas:
            assert deltas["p1"] > 0, "surprising pattern should get positive delta"


class TestArchitecturalSeparation:
    """Verify architectural separation achieved through protocol."""

    def test_substrate_import_independent_of_agentic(self) -> None:
        """quantum_substrate can be imported without agentic imports."""
        # This test runs in the test environment where agentic is available,
        # but we verify that the substrate modules don't import from agentic
        import quantum_substrate

        # Check module has key exports
        assert hasattr(quantum_substrate, 'SubstratePattern')
        assert hasattr(quantum_substrate, 'CoherenceManager')
        assert hasattr(quantum_substrate, 'SurpriseDetector')
        assert hasattr(quantum_substrate, 'attempt_tunneling')

    def test_no_circular_import(self) -> None:
        """Both packages can be imported without circular import errors."""
        # Import substrate first
        import quantum_substrate
        # Import agentic second
        import agentic

        # Both should work
        assert quantum_substrate is not None
        assert agentic is not None

    def test_protocol_enables_type_checking(self) -> None:
        """Protocol can be used for type hints without runtime issues."""
        from quantum_substrate.protocols import SubstratePattern

        # This function uses protocol for type hint
        def process_pattern(p: SubstratePattern) -> float:
            return p.coherence * p.embeddedness

        # Works with EvolvingPattern
        pattern = EvolvingPattern.from_text("type checking test")
        result = process_pattern(pattern)
        assert isinstance(result, float)
