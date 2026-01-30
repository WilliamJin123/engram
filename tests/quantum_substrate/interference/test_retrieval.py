"""Tests for interference-based retrieval - retrieval comparison tests."""

import math
import torch
import pytest

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.interference import (
    jaccard_retrieval,
    interference_retrieval,
)


class TestRetrievalComparison:
    """Compare Jaccard vs interference retrieval."""

    def test_retrieve_related_patterns(self, dim, rng):
        """Interference retrieval distinguishes phase relationships."""
        # Create query pattern
        query = ComplexSparsePattern.random(dim=dim, k=10, generator=rng)

        # Create pattern A: same indices, same phase (constructive)
        pattern_a = ComplexSparsePattern(
            dim=dim,
            indices=query.indices,
            magnitudes=query.magnitudes,
            phases=query.phases,
        )

        # Create pattern B: same indices, opposite phase (destructive)
        opposite_phases = tuple((p + math.pi) % (2 * math.pi) for p in query.phases)
        pattern_b = ComplexSparsePattern(
            dim=dim,
            indices=query.indices,
            magnitudes=query.magnitudes,
            phases=opposite_phases,
        )

        patterns = [pattern_a, pattern_b]

        # Jaccard sees them as equally similar (same index sets)
        jaccard_results = jaccard_retrieval(query, patterns)
        assert jaccard_results[0][1] == pytest.approx(jaccard_results[1][1], abs=0.01)

        # Interference correctly ranks constructive above destructive
        interference_results = interference_retrieval(query, patterns)
        # Pattern A (constructive) should have higher similarity
        assert interference_results[0][0] == 0  # Index of pattern_a
        assert interference_results[0][1] > interference_results[1][1]

    def test_retrieval_at_scale_50(self, dim, rng):
        """Interference retrieval works at 50 patterns."""
        self._test_retrieval_at_scale(dim, rng, num_patterns=50)

    def test_retrieval_at_scale_100(self, dim, rng):
        """Interference retrieval works at 100 patterns."""
        self._test_retrieval_at_scale(dim, rng, num_patterns=100)

    def test_retrieval_at_scale_500(self, dim, rng):
        """Interference retrieval works at 500 patterns."""
        self._test_retrieval_at_scale(dim, rng, num_patterns=500)

    def _test_retrieval_at_scale(self, dim, rng, num_patterns: int):
        """Test retrieval at specified scale.

        Creates a query and num_patterns patterns:
        - Pattern 0: exact match with same phase (should rank highest for interference)
        - Patterns 1-N: random patterns

        Verifies:
        - Both retrieval methods return all patterns
        - Interference correctly identifies the constructive match
        """
        # Create query
        k = 10
        query = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        # Create patterns list
        patterns = []

        # Pattern 0: exact constructive match
        exact_match = ComplexSparsePattern(
            dim=dim,
            indices=query.indices,
            magnitudes=query.magnitudes,
            phases=query.phases,
        )
        patterns.append(exact_match)

        # Rest are random patterns
        for _ in range(num_patterns - 1):
            # Create new generator state for each to avoid correlation
            patterns.append(ComplexSparsePattern.random(dim=dim, k=k, generator=rng))

        # Run both retrieval methods
        jaccard_results = jaccard_retrieval(query, patterns)
        interference_results = interference_retrieval(query, patterns)

        # Both should return all patterns
        assert len(jaccard_results) == num_patterns
        assert len(interference_results) == num_patterns

        # Interference should rank the exact match first (index 0)
        assert interference_results[0][0] == 0
        assert interference_results[0][1] == pytest.approx(1.0, abs=0.01)

        # Jaccard should also rank pattern 0 highly (but may tie with others
        # that happen to share indices)
        top_jaccard_idx = jaccard_results[0][0]
        assert jaccard_results[0][1] == pytest.approx(1.0, abs=0.01)
        assert top_jaccard_idx == 0

        print(f"\nScale test with {num_patterns} patterns:")
        print(f"  Interference top-5: {interference_results[:5]}")
        print(f"  Jaccard top-5: {jaccard_results[:5]}")
