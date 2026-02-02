# tests/agentic/test_coactivation.py
"""Tests for coactivation-based associative learning."""

import random

import pytest
from agentic.evolving_pattern import EvolvingPattern
from agentic.coactivation import coactivate, CoactivationConfig


class TestCoactivationBasics:
    """Test basic coactivation mechanics."""

    def test_coactivation_shares_bits(self):
        """Co-activated patterns should share some bits."""
        p1 = EvolvingPattern.from_text("dogs are animals", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("wolves are animals", dim=1024, k=50)

        initial_overlap = len(p1.bits & p2.bits)

        for _ in range(10):
            coactivate([p1, p2], strength=0.05)

        final_overlap = len(p1.bits & p2.bits)

        assert final_overlap > initial_overlap, "Coactivation should increase overlap"
        assert p1.acquisition_count > 0
        assert p2.acquisition_count > 0

    def test_original_bits_unchanged(self):
        """Original bits should never change."""
        p1 = EvolvingPattern.from_text("hello", dim=1024, k=50)
        original = frozenset(p1.original_bits)

        p2 = EvolvingPattern.from_text("world", dim=1024, k=50)

        for _ in range(20):
            coactivate([p1, p2], strength=0.1)

        assert p1.original_bits == original, "Original bits must not change"

    def test_only_original_bits_transferred(self):
        """Only original bits should be transferred, not acquired bits."""
        p1 = EvolvingPattern.from_text("alpha", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("beta", dim=1024, k=50)
        p3 = EvolvingPattern.from_text("gamma", dim=1024, k=50)

        for _ in range(5):
            coactivate([p1, p2], strength=0.1)

        p1_acquired_from_p2 = p1.acquired_bits & p2.original_bits
        assert len(p1_acquired_from_p2) > 0, "p1 should have acquired bits from p2"

        p3_bits_before = set(p3.bits)
        for _ in range(5):
            coactivate([p1, p3], strength=0.1)

        p3_new_bits = p3.bits - p3_bits_before

        from_p1_original = p3_new_bits & p1.original_bits
        from_p1_acquired = p3_new_bits & p1.acquired_bits

        assert len(from_p1_original) >= len(from_p1_acquired), \
            "Transferred bits should come from original, not acquired"

    def test_max_bits_respected(self):
        """Patterns should not exceed max bit budget."""
        config = CoactivationConfig(max_bits=60, strength=0.2)

        p1 = EvolvingPattern.from_text("one", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("two", dim=1024, k=50)

        for _ in range(100):
            coactivate([p1, p2], config=config)

        assert len(p1.bits) <= 60, f"Pattern exceeded max_bits: {len(p1.bits)}"
        assert len(p2.bits) <= 60, f"Pattern exceeded max_bits: {len(p2.bits)}"

    def test_obesity_decay(self):
        """Large patterns should learn slower."""
        config = CoactivationConfig(obesity_decay=True, max_bits=100)

        p_large = EvolvingPattern.from_text("large pattern", dim=1024, k=50)
        p_small = EvolvingPattern.from_text("small pattern", dim=1024, k=50)
        p_donor = EvolvingPattern.from_text("donor", dim=1024, k=50)

        for i in range(30):
            p_large.bits.add(800 + i)
            p_large.acquired_bits.add(800 + i)
            p_large.phases[800 + i] = 0.0

        large_before = len(p_large.bits)
        small_before = len(p_small.bits)

        for _ in range(10):
            coactivate([p_large, p_donor], config=config)
            coactivate([p_small, p_donor], config=config)

        large_gained = len(p_large.bits) - large_before
        small_gained = len(p_small.bits) - small_before

        assert small_gained >= large_gained, \
            f"Small pattern should gain more: small={small_gained}, large={large_gained}"


class TestCoactivationDeterminism:
    """Test deterministic coactivation with seeded RNG."""

    def test_deterministic_with_seed(self):
        """Same seed produces identical results."""
        p1a = EvolvingPattern.from_text("determinism test one", dim=1024, k=50)
        p2a = EvolvingPattern.from_text("determinism test two", dim=1024, k=50)

        p1b = EvolvingPattern.from_text("determinism test one", dim=1024, k=50)
        p2b = EvolvingPattern.from_text("determinism test two", dim=1024, k=50)

        # Run with same seed
        rng_a = random.Random(42)
        rng_b = random.Random(42)

        coactivate([p1a, p2a], strength=0.1, rng=rng_a)
        coactivate([p1b, p2b], strength=0.1, rng=rng_b)

        assert p1a.bits == p1b.bits, "Same seed should produce identical p1 bits"
        assert p2a.bits == p2b.bits, "Same seed should produce identical p2 bits"
        assert p1a.acquired_bits == p1b.acquired_bits

    def test_different_seeds_differ(self):
        """Different seeds produce different results (usually)."""
        p1a = EvolvingPattern.from_text("seed test alpha", dim=1024, k=50)
        p2a = EvolvingPattern.from_text("seed test beta", dim=1024, k=50)

        p1b = EvolvingPattern.from_text("seed test alpha", dim=1024, k=50)
        p2b = EvolvingPattern.from_text("seed test beta", dim=1024, k=50)

        rng_a = random.Random(42)
        rng_b = random.Random(999)

        coactivate([p1a, p2a], strength=0.1, rng=rng_a)
        coactivate([p1b, p2b], strength=0.1, rng=rng_b)

        # With different seeds, at least one pattern should differ
        # (could technically be same by chance, but extremely unlikely)
        differs = (p1a.bits != p1b.bits) or (p2a.bits != p2b.bits)
        assert differs, "Different seeds should (usually) produce different results"

    def test_no_seed_is_stochastic(self):
        """Without seed, results vary between runs."""
        results = []
        for _ in range(5):
            p1 = EvolvingPattern.from_text("stochastic one", dim=1024, k=50)
            p2 = EvolvingPattern.from_text("stochastic two", dim=1024, k=50)
            coactivate([p1, p2], strength=0.1)  # No rng = uses global random
            results.append(frozenset(p1.bits))

        # Not all results should be identical (stochastic behavior)
        unique_results = len(set(results))
        assert unique_results > 1, "Without seed, results should vary"
