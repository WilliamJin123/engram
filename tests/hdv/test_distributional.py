"""Tests for DistributionalHDV data class."""

import torch
import pytest
from engram.hdv.distributional import DistributionalHDV, random_distributional


class TestDistributionalHDVCreation:
    """Test suite for DistributionalHDV instantiation."""

    def test_create_with_mean_and_variance(self, dim):
        """Should create instance with mean and variance tensors."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5
        hdv = DistributionalHDV(mean=mean, variance=variance)
        assert hdv.mean is not None
        assert hdv.variance is not None

    def test_mean_shape_preserved(self, dim):
        """Mean tensor should preserve shape."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5
        hdv = DistributionalHDV(mean=mean, variance=variance)
        assert hdv.mean.shape == (dim,)

    def test_variance_shape_preserved(self, dim):
        """Variance tensor should preserve shape."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5
        hdv = DistributionalHDV(mean=mean, variance=variance)
        assert hdv.variance.shape == (dim,)

    def test_timestamps_default_to_zero(self, dim):
        """Timestamps should default to 0.0 if not provided."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5
        hdv = DistributionalHDV(mean=mean, variance=variance)
        assert hdv.last_accessed == 0.0
        assert hdv.last_updated == 0.0

    def test_timestamps_can_be_set(self, dim):
        """Should accept custom timestamps."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5
        hdv = DistributionalHDV(
            mean=mean, variance=variance,
            last_accessed=100.0, last_updated=50.0
        )
        assert hdv.last_accessed == 100.0
        assert hdv.last_updated == 50.0

    def test_dim_property(self, dim):
        """Should expose dimension as property."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5
        hdv = DistributionalHDV(mean=mean, variance=variance)
        assert hdv.dim == dim


class TestDistributionalHDVValidation:
    """Test variance constraints."""

    def test_variance_must_be_positive(self, dim):
        """Variance values must be positive."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * -0.5  # Invalid
        with pytest.raises(ValueError, match="positive"):
            DistributionalHDV(mean=mean, variance=variance)

    def test_zero_variance_not_allowed(self, dim):
        """Variance of exactly zero is not allowed."""
        mean = torch.randn(dim)
        variance = torch.zeros(dim)  # Invalid
        with pytest.raises(ValueError, match="positive"):
            DistributionalHDV(mean=mean, variance=variance)

    def test_shape_mismatch_raises(self, dim):
        """Mean and variance must have same shape."""
        mean = torch.randn(dim)
        variance = torch.ones(dim // 2)  # Wrong shape
        with pytest.raises(ValueError, match="shape"):
            DistributionalHDV(mean=mean, variance=variance)


class TestRandomDistributional:
    """Test suite for random_distributional factory."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a DistributionalHDV instance."""
        hdv = random_distributional(dim)
        assert isinstance(hdv, DistributionalHDV)

    def test_correct_dimension(self, dim):
        """Should create HDV with requested dimension."""
        hdv = random_distributional(dim)
        assert hdv.dim == dim

    def test_mean_is_ternary(self, dim):
        """Mean should contain only {-1, 0, +1} values."""
        hdv = random_distributional(dim, seed=42)
        unique_vals = torch.unique(hdv.mean)
        for val in unique_vals:
            assert val.item() in (-1.0, 0.0, 1.0)

    def test_variance_is_uniform_by_default(self, dim):
        """Default variance should be uniform across dimensions."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=42)
        assert torch.allclose(hdv.variance, torch.full((dim,), 0.5))

    def test_custom_initial_variance(self, dim):
        """Should accept custom initial variance."""
        hdv = random_distributional(dim, initial_variance=0.8, seed=42)
        assert torch.allclose(hdv.variance, torch.full((dim,), 0.8))

    def test_seed_reproducibility(self, dim):
        """Same seed should produce same mean."""
        hdv1 = random_distributional(dim, seed=123)
        hdv2 = random_distributional(dim, seed=123)
        assert torch.equal(hdv1.mean, hdv2.mean)

    def test_different_seeds_differ(self, dim):
        """Different seeds should produce different means."""
        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=2)
        assert not torch.equal(hdv1.mean, hdv2.mean)

    def test_sparsity_controls_zeros(self, dim):
        """Sparsity parameter should control fraction of non-zeros in mean."""
        hdv = random_distributional(dim, sparsity=0.3, seed=42)
        non_zero_frac = (hdv.mean != 0).float().mean().item()
        assert abs(non_zero_frac - 0.3) < 0.05  # Within 5%

    def test_timestamps_initialized_with_current_time(self, dim):
        """Should initialize timestamps with provided current_time."""
        hdv = random_distributional(dim, current_time=1000.0, seed=42)
        assert hdv.last_accessed == 1000.0
        assert hdv.last_updated == 1000.0
