"""Tests for interference-based retrieval - basics and edge cases."""

import math
import torch
import pytest

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.interference import (
    jaccard_retrieval,
    interference_retrieval,
    create_related_pattern,
)


class TestInterferenceBasics:
    """Test fundamental interference properties."""

    def test_same_phase_constructive(self, dim, rng):
        """Patterns with same phase show constructive interference."""
        # Create two patterns with identical indices and phases
        k = 10
        pattern1 = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        # Create pattern2 with same indices and phases (perfect overlap, same phase)
        pattern2 = ComplexSparsePattern(
            dim=dim,
            indices=pattern1.indices,
            magnitudes=pattern1.magnitudes,
            phases=pattern1.phases,
        )

        # Interference retrieval should give maximum similarity (1.0)
        results = interference_retrieval(pattern1, [pattern2])
        assert len(results) == 1
        idx, similarity = results[0]
        assert idx == 0
        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_opposite_phase_destructive(self, dim, rng):
        """Patterns with opposite phases show destructive interference."""
        # Create a pattern
        k = 10
        pattern1 = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        # Create pattern2 with same indices but phases shifted by pi (opposite)
        shifted_phases = tuple((p + math.pi) % (2 * math.pi) for p in pattern1.phases)
        pattern2 = ComplexSparsePattern(
            dim=dim,
            indices=pattern1.indices,
            magnitudes=pattern1.magnitudes,
            phases=shifted_phases,
        )

        # Interference retrieval should give minimum similarity (0.0)
        # because |a*e^(i*theta) + a*e^(i*(theta+pi))| = |a - a| = 0
        results = interference_retrieval(pattern1, [pattern2])
        assert len(results) == 1
        idx, similarity = results[0]
        assert idx == 0
        assert similarity == pytest.approx(0.0, abs=0.01)

    def test_jaccard_ignores_phase(self, dim, rng):
        """Jaccard similarity is phase-invariant."""
        k = 10
        pattern1 = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        # Same indices, different phases
        shifted_phases = tuple((p + math.pi) % (2 * math.pi) for p in pattern1.phases)
        pattern2_same_phase = ComplexSparsePattern(
            dim=dim,
            indices=pattern1.indices,
            magnitudes=pattern1.magnitudes,
            phases=pattern1.phases,
        )
        pattern2_opposite_phase = ComplexSparsePattern(
            dim=dim,
            indices=pattern1.indices,
            magnitudes=pattern1.magnitudes,
            phases=shifted_phases,
        )

        # Jaccard should give same result regardless of phase
        results_same = jaccard_retrieval(pattern1, [pattern2_same_phase])
        results_opposite = jaccard_retrieval(pattern1, [pattern2_opposite_phase])

        # Both should be 1.0 (identical index sets)
        assert results_same[0][1] == pytest.approx(1.0, abs=0.01)
        assert results_opposite[0][1] == pytest.approx(1.0, abs=0.01)

    def test_create_related_pattern_overlap(self, dim, rng):
        """Related pattern shares expected fraction of indices."""
        k = 20
        source = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        overlap_frac = 0.5
        related = create_related_pattern(
            source, overlap_frac=overlap_frac, phase_noise=0.0, generator=rng
        )

        # Count actual overlap
        source_set = set(source.indices)
        related_set = set(related.indices)
        actual_overlap = len(source_set & related_set)

        expected_overlap = int(overlap_frac * k)
        assert actual_overlap == expected_overlap

    def test_create_related_pattern_same_k(self, dim, rng):
        """Related pattern has same sparsity as source."""
        k = 15
        source = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        related = create_related_pattern(source, overlap_frac=0.7, generator=rng)

        assert related.k == source.k


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_overlap(self, dim, rng):
        """Patterns with no overlap have zero similarity."""
        # Create two disjoint patterns
        k = 10
        # First pattern uses indices 0-9
        pattern1 = ComplexSparsePattern(
            dim=dim,
            indices=tuple(range(k)),
            magnitudes=tuple([1.0] * k),
            phases=tuple([0.0] * k),
        )
        # Second pattern uses indices 100-109
        pattern2 = ComplexSparsePattern(
            dim=dim,
            indices=tuple(range(100, 100 + k)),
            magnitudes=tuple([1.0] * k),
            phases=tuple([0.0] * k),
        )

        jaccard = jaccard_retrieval(pattern1, [pattern2])
        interference = interference_retrieval(pattern1, [pattern2])

        assert jaccard[0][1] == 0.0
        assert interference[0][1] == 0.0

    def test_partial_overlap(self, dim, rng):
        """Partial overlap gives intermediate similarity."""
        k = 10
        # Pattern 1: indices 0-9
        pattern1 = ComplexSparsePattern(
            dim=dim,
            indices=tuple(range(k)),
            magnitudes=tuple([1.0] * k),
            phases=tuple([0.0] * k),
        )
        # Pattern 2: indices 5-14 (50% overlap)
        pattern2 = ComplexSparsePattern(
            dim=dim,
            indices=tuple(range(5, 5 + k)),
            magnitudes=tuple([1.0] * k),
            phases=tuple([0.0] * k),
        )

        jaccard = jaccard_retrieval(pattern1, [pattern2])
        interference = interference_retrieval(pattern1, [pattern2])

        # Jaccard: |5-9| / |0-14| = 5/15 = 0.333
        expected_jaccard = 5 / 15
        assert jaccard[0][1] == pytest.approx(expected_jaccard, abs=0.01)

        # Interference with same phase should be positive
        assert interference[0][1] > 0.0

    def test_empty_patterns_list(self, dim, rng):
        """Empty patterns list returns empty results."""
        query = ComplexSparsePattern.random(dim=dim, k=10, generator=rng)

        jaccard = jaccard_retrieval(query, [])
        interference = interference_retrieval(query, [])

        assert jaccard == []
        assert interference == []
