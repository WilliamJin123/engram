"""Tests for ComplexSparsePattern."""

import math
import torch
import pytest

from quantum_substrate.patterns import ComplexSparsePattern


class TestPatternCreation:
    """Test pattern construction and basic properties."""

    def test_create_random_pattern(self, dim, rng):
        """Random pattern has correct dimensionality and sparsity."""
        k = 10  # number of active dimensions
        pattern = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        assert pattern.dim == dim
        assert len(pattern.indices) == k
        assert all(0 <= i < dim for i in pattern.indices)

    def test_pattern_has_complex_amplitudes(self, dim, rng):
        """Each active dimension has magnitude and phase."""
        pattern = ComplexSparsePattern.random(dim=dim, k=10, generator=rng)

        assert len(pattern.magnitudes) == len(pattern.indices)
        assert len(pattern.phases) == len(pattern.indices)
        assert all(m > 0 for m in pattern.magnitudes)
        assert all(0 <= p < 2 * math.pi for p in pattern.phases)

    def test_pattern_to_dense(self, rng):
        """Can convert to dense complex tensor."""
        pattern = ComplexSparsePattern.random(dim=100, k=5, generator=rng)
        dense = pattern.to_dense()

        assert dense.shape == (100,)
        assert dense.dtype == torch.complex64
        assert (dense.abs() > 0).sum() == 5  # exactly 5 non-zero

    def test_pattern_from_dense(self, rng):
        """Can create from dense complex tensor."""
        # Create dense tensor with 3 non-zero entries
        dense = torch.zeros(100, dtype=torch.complex64)
        dense[10] = 1.0 + 0.5j
        dense[50] = 0.5 + 0.5j
        dense[90] = 0.3 + 0.1j

        pattern = ComplexSparsePattern.from_dense(dense)

        assert len(pattern.indices) == 3
        assert set(pattern.indices) == {10, 50, 90}
