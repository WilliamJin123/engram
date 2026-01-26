"""Tests for uncertainty calculations (KL divergence, etc.)."""

import torch
import pytest
import math
from engram.hdv.distributional import DistributionalHDV, random_distributional
from engram.hdv.uncertainty import kl_divergence, symmetric_kl


class TestKLDivergence:
    """Test suite for KL divergence between diagonal Gaussians."""

    def test_identical_distributions_zero_kl(self, dim):
        """KL divergence of identical distributions is 0."""
        hdv = random_distributional(dim, seed=42)
        kl = kl_divergence(hdv, hdv)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_kl_is_non_negative(self, dim):
        """KL divergence is always >= 0."""
        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=2)
        kl = kl_divergence(hdv1, hdv2)
        assert kl >= 0

    def test_kl_is_asymmetric(self, dim):
        """KL(P||Q) != KL(Q||P) in general."""
        # Create distributions with different variances
        mean1 = torch.randn(dim)
        mean2 = mean1 + 0.5  # Slightly different mean
        var1 = torch.ones(dim) * 0.3
        var2 = torch.ones(dim) * 0.7

        hdv1 = DistributionalHDV(mean=mean1, variance=var1)
        hdv2 = DistributionalHDV(mean=mean2, variance=var2)

        kl_12 = kl_divergence(hdv1, hdv2)
        kl_21 = kl_divergence(hdv2, hdv1)

        assert kl_12 != pytest.approx(kl_21, rel=0.1)

    def test_larger_mean_diff_larger_kl(self, dim):
        """Larger mean difference should give larger KL."""
        base_mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5

        hdv_base = DistributionalHDV(mean=base_mean, variance=variance)
        hdv_close = DistributionalHDV(mean=base_mean + 0.1, variance=variance)
        hdv_far = DistributionalHDV(mean=base_mean + 1.0, variance=variance)

        kl_close = kl_divergence(hdv_base, hdv_close)
        kl_far = kl_divergence(hdv_base, hdv_far)

        assert kl_far > kl_close

    def test_larger_variance_ratio_larger_kl(self, dim):
        """Larger variance ratio should give larger KL."""
        mean = torch.randn(dim)
        var_base = torch.ones(dim) * 0.5

        hdv_base = DistributionalHDV(mean=mean, variance=var_base)
        hdv_similar_var = DistributionalHDV(mean=mean, variance=var_base * 1.1)
        hdv_diff_var = DistributionalHDV(mean=mean, variance=var_base * 3.0)

        kl_similar = kl_divergence(hdv_base, hdv_similar_var)
        kl_diff = kl_divergence(hdv_base, hdv_diff_var)

        assert kl_diff > kl_similar

    def test_kl_formula_validation(self):
        """Validate KL formula against known analytical result.

        For 1D Gaussians: KL(N(mu1,sigma1^2) || N(mu2,sigma2^2)) =
        log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2)/(2*sigma2^2) - 1/2
        """
        # Simple 1D case
        mean1 = torch.tensor([1.0])
        mean2 = torch.tensor([2.0])
        var1 = torch.tensor([0.5])
        var2 = torch.tensor([1.0])

        hdv1 = DistributionalHDV(mean=mean1, variance=var1)
        hdv2 = DistributionalHDV(mean=mean2, variance=var2)

        # Analytical KL
        sigma1, sigma2 = math.sqrt(0.5), math.sqrt(1.0)
        expected = (
            math.log(sigma2 / sigma1)
            + (var1[0].item() + (mean1[0] - mean2[0])**2) / (2 * var2[0].item())
            - 0.5
        )

        kl = kl_divergence(hdv1, hdv2)
        assert kl == pytest.approx(expected.item(), rel=0.01)


class TestSymmetricKL:
    """Test suite for symmetric (Jeffrey's) KL divergence."""

    def test_symmetric_kl_is_symmetric(self, dim):
        """Symmetric KL should give same result both ways."""
        hdv1 = random_distributional(dim, initial_variance=0.3, seed=1)
        hdv2 = random_distributional(dim, initial_variance=0.7, seed=2)

        skl_12 = symmetric_kl(hdv1, hdv2)
        skl_21 = symmetric_kl(hdv2, hdv1)

        assert skl_12 == pytest.approx(skl_21, rel=1e-6)

    def test_identical_distributions_zero_symmetric_kl(self, dim):
        """Symmetric KL of identical distributions is 0."""
        hdv = random_distributional(dim, seed=42)
        skl = symmetric_kl(hdv, hdv)
        assert skl == pytest.approx(0.0, abs=1e-6)

    def test_symmetric_kl_is_average_of_kl(self, dim):
        """Symmetric KL = (KL(P||Q) + KL(Q||P)) / 2."""
        hdv1 = random_distributional(dim, initial_variance=0.3, seed=1)
        hdv2 = random_distributional(dim, initial_variance=0.7, seed=2)

        kl_12 = kl_divergence(hdv1, hdv2)
        kl_21 = kl_divergence(hdv2, hdv1)
        expected = (kl_12 + kl_21) / 2

        skl = symmetric_kl(hdv1, hdv2)
        assert skl == pytest.approx(expected, rel=1e-6)
