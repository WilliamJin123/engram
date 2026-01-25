"""Tests for random_ternary HDV generation."""

import torch
import pytest
from engram.hdv import random_ternary


class TestRandomTernary:
    """Test suite for random_ternary function."""

    def test_returns_tensor(self, dim):
        """Should return a torch tensor."""
        v = random_ternary(dim)
        assert isinstance(v, torch.Tensor)

    def test_correct_dimension(self, dim):
        """Should return tensor of specified dimension."""
        v = random_ternary(dim)
        assert v.shape == (dim,)

    def test_only_ternary_values(self, dim):
        """Should contain only -1, 0, and +1 values."""
        v = random_ternary(dim)
        unique_vals = torch.unique(v)
        for val in unique_vals:
            assert val.item() in (-1.0, 0.0, 1.0)

    def test_respects_sparsity(self, dim):
        """Should have approximately the specified fraction of non-zero elements."""
        sparsity = 0.5
        v = random_ternary(dim, sparsity=sparsity)
        nonzero_ratio = (v != 0).float().mean().item()
        # Allow 5% tolerance
        assert abs(nonzero_ratio - sparsity) < 0.05

    def test_balanced_nonzero_values(self, dim):
        """Non-zero values should be roughly balanced between +1 and -1."""
        v = random_ternary(dim, sparsity=0.5)
        nonzero = v[v != 0]
        if len(nonzero) > 0:
            ones_ratio = (nonzero == 1).float().mean().item()
            assert 0.45 <= ones_ratio <= 0.55

    def test_reproducible_with_seed(self, dim):
        """Should produce same vector with same seed."""
        v1 = random_ternary(dim, seed=42)
        v2 = random_ternary(dim, seed=42)
        assert torch.equal(v1, v2)

    def test_different_with_different_seeds(self, dim):
        """Should produce different vectors with different seeds."""
        v1 = random_ternary(dim, seed=42)
        v2 = random_ternary(dim, seed=43)
        assert not torch.equal(v1, v2)

    def test_different_without_seed(self, dim):
        """Should produce different vectors when no seed specified."""
        v1 = random_ternary(dim)
        v2 = random_ternary(dim)
        # Extremely unlikely to be equal by chance
        assert not torch.equal(v1, v2)

    def test_dtype_is_float(self, dim):
        """Should return float tensor for compatibility with other operations."""
        v = random_ternary(dim)
        assert v.dtype in (torch.float32, torch.float64)

    def test_full_sparsity_gives_bipolar(self, dim):
        """Sparsity=1.0 should give no zeros (bipolar-like)."""
        v = random_ternary(dim, sparsity=1.0)
        assert (v == 0).sum().item() == 0

    def test_zero_sparsity_gives_all_zeros(self, dim):
        """Sparsity=0.0 should give all zeros."""
        v = random_ternary(dim, sparsity=0.0)
        assert (v == 0).sum().item() == dim

    def test_default_sparsity(self, dim):
        """Default sparsity should be 0.5."""
        v = random_ternary(dim, seed=42)
        nonzero_ratio = (v != 0).float().mean().item()
        assert 0.45 <= nonzero_ratio <= 0.55
