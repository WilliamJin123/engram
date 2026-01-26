"""Tests for DistributionalHDV data class."""

import torch
import pytest
from engram.hdv.distributional import DistributionalHDV


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
