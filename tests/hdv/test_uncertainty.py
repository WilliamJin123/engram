"""Tests for uncertainty calculations (KL divergence, etc.)."""

import torch
import pytest
import math
from engram.hdv.distributional import DistributionalHDV, random_distributional
from engram.hdv.uncertainty import (
    kl_divergence, symmetric_kl, distributional_similarity, bayesian_update
)


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


class TestDistributionalSimilarity:
    """Test suite for hyperbolic KL-based similarity."""

    def test_identical_distributions_similarity_one(self, dim):
        """Identical distributions should have similarity 1.0."""
        hdv = random_distributional(dim, seed=42)
        sim, uncertainty = distributional_similarity(hdv, hdv)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_returns_tuple_of_floats(self, dim):
        """Should return (similarity, uncertainty) tuple."""
        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=2)
        result = distributional_similarity(hdv1, hdv2)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_similarity_bounded_zero_to_one(self, dim):
        """Similarity should be in range [0, 1]."""
        for seed in range(10):
            hdv1 = random_distributional(dim, seed=seed)
            hdv2 = random_distributional(dim, seed=seed + 100)
            sim, _ = distributional_similarity(hdv1, hdv2)
            assert 0.0 <= sim <= 1.0

    def test_similarity_is_symmetric(self, dim):
        """Similarity should be symmetric."""
        hdv1 = random_distributional(dim, initial_variance=0.3, seed=1)
        hdv2 = random_distributional(dim, initial_variance=0.7, seed=2)

        sim_12, _ = distributional_similarity(hdv1, hdv2)
        sim_21, _ = distributional_similarity(hdv2, hdv1)

        assert sim_12 == pytest.approx(sim_21, rel=1e-6)

    def test_closer_means_higher_similarity(self, dim):
        """Distributions with closer means should have higher similarity."""
        base = random_distributional(dim, seed=42)

        # Create distributions at different distances
        close_mean = base.mean + 0.1 * torch.randn(dim)
        far_mean = base.mean + 2.0 * torch.randn(dim)

        close = DistributionalHDV(mean=close_mean, variance=base.variance.clone())
        far = DistributionalHDV(mean=far_mean, variance=base.variance.clone())

        sim_close, _ = distributional_similarity(base, close)
        sim_far, _ = distributional_similarity(base, far)

        assert sim_close > sim_far

    def test_hyperbolic_formula(self, dim):
        """Should use hyperbolic formula: 1 / (1 + kl_scale * KL)."""
        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=2)

        skl = symmetric_kl(hdv1, hdv2)
        expected_sim = 1.0 / (1.0 + skl)  # default kl_scale=1.0

        sim, _ = distributional_similarity(hdv1, hdv2)
        assert sim == pytest.approx(expected_sim, rel=1e-6)

    def test_custom_kl_scale(self, dim):
        """Should respect custom kl_scale parameter."""
        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=2)

        skl = symmetric_kl(hdv1, hdv2)
        kl_scale = 2.0
        expected_sim = 1.0 / (1.0 + kl_scale * skl)

        sim, _ = distributional_similarity(hdv1, hdv2, kl_scale=kl_scale)
        assert sim == pytest.approx(expected_sim, rel=1e-6)

    def test_uncertainty_higher_with_high_variance(self, dim):
        """Uncertainty should be higher when distributions have high variance."""
        low_var = random_distributional(dim, initial_variance=0.1, seed=1)
        high_var = random_distributional(dim, initial_variance=1.0, seed=1)  # Same mean

        other = random_distributional(dim, initial_variance=0.5, seed=2)

        _, unc_low = distributional_similarity(low_var, other)
        _, unc_high = distributional_similarity(high_var, other)

        assert unc_high > unc_low

    def test_uncertainty_non_negative(self, dim):
        """Uncertainty should always be >= 0."""
        for seed in range(10):
            hdv1 = random_distributional(dim, seed=seed)
            hdv2 = random_distributional(dim, seed=seed + 100)
            _, uncertainty = distributional_similarity(hdv1, hdv2)
            assert uncertainty >= 0.0

    def test_very_different_distributions_low_similarity(self, dim):
        """Very different distributions should have low (but positive) similarity."""
        # Create maximally different distributions
        mean1 = torch.ones(dim)
        mean2 = -torch.ones(dim)
        var = torch.ones(dim) * 0.1  # Low variance = confident they're different

        hdv1 = DistributionalHDV(mean=mean1, variance=var)
        hdv2 = DistributionalHDV(mean=mean2, variance=var)

        sim, _ = distributional_similarity(hdv1, hdv2)

        # Should be low but positive (hyperbolic never reaches 0)
        assert 0.0 < sim < 0.1


class TestBayesianUpdate:
    """Test suite for Kalman-style Bayesian updates."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a new DistributionalHDV."""
        prior = random_distributional(dim, initial_variance=0.5, seed=1)
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        posterior = bayesian_update(prior, observation, obs_variance, current_time=1.0)
        assert isinstance(posterior, DistributionalHDV)

    def test_preserves_dimension(self, dim):
        """Posterior should have same dimension as prior."""
        prior = random_distributional(dim, initial_variance=0.5, seed=1)
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        posterior = bayesian_update(prior, observation, obs_variance, current_time=1.0)
        assert posterior.dim == dim

    def test_variance_always_decreases(self, dim):
        """Posterior variance should be <= prior variance (we gained information)."""
        prior = random_distributional(dim, initial_variance=0.5, seed=1)
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        posterior = bayesian_update(prior, observation, obs_variance, current_time=1.0)

        # Every dimension should have reduced or equal variance
        assert (posterior.variance <= prior.variance + 1e-6).all()

    def test_mean_moves_toward_observation(self, dim):
        """Posterior mean should move toward observation."""
        prior = random_distributional(dim, initial_variance=0.5, seed=1)
        observation = torch.randn(dim) * 2  # Far from origin
        obs_variance = torch.ones(dim) * 0.3

        posterior = bayesian_update(prior, observation, obs_variance, current_time=1.0)

        # Distance to observation should decrease
        dist_before = torch.norm(prior.mean - observation)
        dist_after = torch.norm(posterior.mean - observation)
        assert dist_after < dist_before

    def test_low_obs_variance_bigger_shift(self, dim):
        """Lower observation variance (high confidence) should shift mean more."""
        prior = random_distributional(dim, initial_variance=0.5, seed=1)
        observation = torch.ones(dim)  # Same observation

        # High confidence observation
        posterior_confident = bayesian_update(
            prior, observation, torch.ones(dim) * 0.1, current_time=1.0
        )
        # Low confidence observation
        posterior_uncertain = bayesian_update(
            prior, observation, torch.ones(dim) * 1.0, current_time=1.0
        )

        # Confident observation should move mean more
        shift_confident = torch.norm(posterior_confident.mean - prior.mean)
        shift_uncertain = torch.norm(posterior_uncertain.mean - prior.mean)
        assert shift_confident > shift_uncertain

    def test_low_obs_variance_bigger_variance_reduction(self, dim):
        """Lower observation variance should reduce posterior variance more."""
        prior = random_distributional(dim, initial_variance=0.5, seed=1)
        observation = torch.ones(dim)

        posterior_confident = bayesian_update(
            prior, observation, torch.ones(dim) * 0.1, current_time=1.0
        )
        posterior_uncertain = bayesian_update(
            prior, observation, torch.ones(dim) * 1.0, current_time=1.0
        )

        # Confident observation should reduce variance more
        assert posterior_confident.variance.mean() < posterior_uncertain.variance.mean()

    def test_kalman_gain_formula(self):
        """Validate against analytical Kalman filter formula.

        K = prior_var / (prior_var + obs_var)
        posterior_var = (1 - K) * prior_var
        posterior_mean = prior_mean + K * (obs - prior_mean)
        """
        prior_var = 0.5
        obs_var = 0.3
        prior_mean_val = 1.0
        obs_val = 2.0

        prior = DistributionalHDV(
            mean=torch.tensor([prior_mean_val]),
            variance=torch.tensor([prior_var])
        )
        observation = torch.tensor([obs_val])
        obs_variance = torch.tensor([obs_var])

        posterior = bayesian_update(prior, observation, obs_variance, current_time=1.0)

        # Analytical solution
        K = prior_var / (prior_var + obs_var)
        expected_var = (1 - K) * prior_var
        expected_mean = prior_mean_val + K * (obs_val - prior_mean_val)

        assert posterior.variance[0].item() == pytest.approx(expected_var, rel=0.01)
        assert posterior.mean[0].item() == pytest.approx(expected_mean, rel=0.01)

    def test_multiple_updates_converge(self, dim):
        """Multiple observations should converge mean and reduce variance."""
        prior = random_distributional(dim, initial_variance=1.0, seed=1)
        target = torch.ones(dim) * 0.5  # True value we're observing

        current = prior
        for i in range(10):
            # Observe with some noise
            noisy_obs = target + torch.randn(dim) * 0.1
            current = bayesian_update(
                current, noisy_obs, torch.ones(dim) * 0.2, current_time=float(i)
            )

        # Should be close to target with low variance
        assert torch.norm(current.mean - target) < torch.norm(prior.mean - target)
        assert current.variance.mean() < prior.variance.mean()

    def test_timestamps_updated(self, dim):
        """Should update timestamps to current_time."""
        prior = random_distributional(dim, current_time=0.0, seed=1)
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        posterior = bayesian_update(prior, observation, obs_variance, current_time=100.0)

        assert posterior.last_accessed == 100.0
        assert posterior.last_updated == 100.0
