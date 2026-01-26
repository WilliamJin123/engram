"""Tests for uncertainty calculations (KL divergence, etc.)."""

import torch
import pytest
import math
from engram.hdv.distributional import DistributionalHDV, random_distributional
from engram.graph.edge import Edge
from engram.hdv.uncertainty import (
    kl_divergence, symmetric_kl, distributional_similarity,
    bayesian_update, temporal_decay, UncertaintyParams, human_confirm,
    handle_contradiction, bundle_observations, propagate_through_edge
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


class TestUncertaintyParams:
    """Test suite for UncertaintyParams configuration."""

    def test_default_values(self):
        """Should have sensible defaults."""
        params = UncertaintyParams()
        assert params.base_drift_rate > 0
        assert params.access_drift_rate > 0
        assert params.access_grace_period > 0
        assert params.min_variance > 0
        assert params.max_variance > params.min_variance

    def test_custom_values(self):
        """Should accept custom values."""
        params = UncertaintyParams(
            base_drift_rate=0.01,
            access_drift_rate=0.05,
            access_grace_period=50.0,
        )
        assert params.base_drift_rate == 0.01
        assert params.access_drift_rate == 0.05
        assert params.access_grace_period == 50.0


class TestTemporalDecay:
    """Test suite for temporal variance decay."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a new DistributionalHDV."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        params = UncertaintyParams()

        decayed = temporal_decay(hdv, current_time=100.0, params=params)
        assert isinstance(decayed, DistributionalHDV)

    def test_variance_increases_over_time(self, dim):
        """Variance should increase as time passes."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        params = UncertaintyParams()

        decayed = temporal_decay(hdv, current_time=1000.0, params=params)

        assert decayed.variance.mean() > hdv.variance.mean()

    def test_base_drift_always_applies(self, dim):
        """Base drift should increase variance even if recently accessed."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        # Set last_accessed to current time (just accessed)
        hdv.last_accessed = 100.0
        params = UncertaintyParams(base_drift_rate=0.01, access_drift_rate=0.05)

        decayed = temporal_decay(hdv, current_time=100.0, params=params)

        # Should still have some increase from base drift
        # (based on time since last_updated, not last_accessed)
        assert decayed.variance.mean() >= hdv.variance.mean()

    def test_access_drift_after_grace_period(self, dim):
        """Additional drift should apply after access grace period."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        params = UncertaintyParams(
            base_drift_rate=0.001,
            access_drift_rate=0.01,
            access_grace_period=50.0
        )

        # Within grace period
        decayed_early = temporal_decay(hdv, current_time=30.0, params=params)

        # After grace period
        decayed_late = temporal_decay(hdv, current_time=200.0, params=params)

        # Late should have more variance increase per unit time
        early_increase = decayed_early.variance.mean() - hdv.variance.mean()
        late_increase = decayed_late.variance.mean() - hdv.variance.mean()

        # Late has much more time AND extra access drift
        assert late_increase > early_increase * 3  # More than just proportional

    def test_variance_capped_at_max(self, dim):
        """Variance should not exceed max_variance."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        params = UncertaintyParams(
            base_drift_rate=0.1,  # Very high drift
            max_variance=1.0
        )

        # Very long time
        decayed = temporal_decay(hdv, current_time=10000.0, params=params)

        assert (decayed.variance <= params.max_variance + 1e-6).all()

    def test_variance_clamped_at_min(self, dim):
        """Variance should not go below min_variance."""
        hdv = random_distributional(dim, initial_variance=0.01, current_time=0.0, seed=1)
        params = UncertaintyParams(min_variance=0.05)

        decayed = temporal_decay(hdv, current_time=0.0, params=params)  # No time passed

        assert (decayed.variance >= params.min_variance - 1e-6).all()

    def test_mean_unchanged(self, dim):
        """Mean should not change during temporal decay."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        params = UncertaintyParams()

        decayed = temporal_decay(hdv, current_time=1000.0, params=params)

        assert torch.equal(decayed.mean, hdv.mean)

    def test_timestamps_updated(self, dim):
        """last_accessed should update to current_time."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        params = UncertaintyParams()

        decayed = temporal_decay(hdv, current_time=500.0, params=params)

        assert decayed.last_accessed == 500.0
        # last_updated stays the same (no new evidence)
        assert decayed.last_updated == hdv.last_updated

    def test_no_time_passed_no_change(self, dim):
        """If no time has passed, variance should stay the same."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=100.0, seed=1)
        params = UncertaintyParams()

        decayed = temporal_decay(hdv, current_time=100.0, params=params)

        assert torch.allclose(decayed.variance, hdv.variance)


class TestHumanConfirm:
    """Test suite for human confirmation (variance collapse)."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a new DistributionalHDV."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=1)
        params = UncertaintyParams()

        confirmed = human_confirm(hdv, current_time=1.0, params=params)
        assert isinstance(confirmed, DistributionalHDV)

    def test_variance_dramatically_reduced(self, dim):
        """Variance should be reduced by human_confirmation_factor."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=1)
        params = UncertaintyParams(human_confirmation_factor=0.01)

        confirmed = human_confirm(hdv, current_time=1.0, params=params)

        expected_var = hdv.variance * 0.01
        assert torch.allclose(confirmed.variance, expected_var.clamp(min=params.min_variance))

    def test_variance_respects_minimum(self, dim):
        """Variance should not go below min_variance."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=1)
        params = UncertaintyParams(
            human_confirmation_factor=0.001,  # Would push below min
            min_variance=0.02
        )

        confirmed = human_confirm(hdv, current_time=1.0, params=params)

        assert (confirmed.variance >= params.min_variance - 1e-6).all()

    def test_mean_unchanged_without_explicit(self, dim):
        """Mean should stay same if no confirmed_mean provided."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=1)
        params = UncertaintyParams()

        confirmed = human_confirm(hdv, current_time=1.0, params=params)

        assert torch.equal(confirmed.mean, hdv.mean)

    def test_mean_updated_with_explicit(self, dim):
        """Mean should change to confirmed_mean if provided."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=1)
        params = UncertaintyParams()
        new_mean = torch.randn(dim)

        confirmed = human_confirm(
            hdv, current_time=1.0, params=params, confirmed_mean=new_mean
        )

        assert torch.equal(confirmed.mean, new_mean)

    def test_timestamps_updated(self, dim):
        """Should update both timestamps."""
        hdv = random_distributional(dim, initial_variance=0.5, current_time=0.0, seed=1)
        params = UncertaintyParams()

        confirmed = human_confirm(hdv, current_time=100.0, params=params)

        assert confirmed.last_accessed == 100.0
        assert confirmed.last_updated == 100.0

    def test_high_variance_becomes_low(self, dim):
        """Even very uncertain concepts become confident after confirmation."""
        hdv = random_distributional(dim, initial_variance=1.5, seed=1)  # High uncertainty
        params = UncertaintyParams(human_confirmation_factor=0.01)

        confirmed = human_confirm(hdv, current_time=1.0, params=params)

        # Should be dramatically lower
        assert confirmed.variance.mean() < hdv.variance.mean() * 0.1


class TestHandleContradiction:
    """Test suite for contradiction handling (increases uncertainty)."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a new DistributionalHDV."""
        existing = random_distributional(dim, initial_variance=0.3, seed=1)
        contradicting = torch.randn(dim)
        params = UncertaintyParams()

        result = handle_contradiction(existing, contradicting, current_time=1.0, params=params)
        assert isinstance(result, DistributionalHDV)

    def test_variance_increases_on_conflict(self, dim):
        """Variance should increase when contradicting evidence arrives."""
        existing = random_distributional(dim, initial_variance=0.3, seed=1)
        # Create contradicting evidence (opposite sign, large magnitude)
        contradicting = -existing.mean * 2
        params = UncertaintyParams(contradiction_scale=0.5)

        result = handle_contradiction(existing, contradicting, current_time=1.0, params=params)

        # Variance should increase
        assert result.variance.mean() > existing.variance.mean()

    def test_no_conflict_minimal_change(self, dim):
        """When evidence is similar, variance increase should be minimal."""
        existing = random_distributional(dim, initial_variance=0.3, seed=1)
        # Evidence very close to existing mean
        contradicting = existing.mean + 0.01 * torch.randn(dim)
        params = UncertaintyParams(contradiction_scale=0.5)

        result = handle_contradiction(existing, contradicting, current_time=1.0, params=params)

        # Variance should increase only minimally
        variance_increase = result.variance.mean() - existing.variance.mean()
        assert variance_increase < 0.1  # Small increase

    def test_mean_moves_slightly_toward_contradiction(self, dim):
        """Mean should shift slightly toward the contradicting evidence."""
        existing = random_distributional(dim, initial_variance=0.3, seed=1)
        contradicting = torch.randn(dim) * 2  # Different from existing
        params = UncertaintyParams()

        result = handle_contradiction(existing, contradicting, current_time=1.0, params=params)

        # Distance to contradicting should decrease
        dist_before = torch.norm(existing.mean - contradicting)
        dist_after = torch.norm(result.mean - contradicting)
        assert dist_after < dist_before

        # But not too much (only 10% blend)
        # New mean should be closer to existing than to contradicting
        dist_to_existing = torch.norm(result.mean - existing.mean)
        assert dist_to_existing < dist_before * 0.2  # Moved less than 20%

    def test_variance_respects_maximum(self, dim):
        """Variance should not exceed max_variance."""
        existing = random_distributional(dim, initial_variance=1.5, seed=1)  # High initial
        contradicting = -existing.mean * 5  # Large contradiction
        params = UncertaintyParams(
            contradiction_scale=1.0,  # High scale
            max_variance=2.0
        )

        result = handle_contradiction(existing, contradicting, current_time=1.0, params=params)

        assert (result.variance <= params.max_variance + 1e-6).all()

    def test_conflict_strength_affects_variance_increase(self, dim):
        """Larger conflicts should increase variance more."""
        existing = random_distributional(dim, initial_variance=0.3, seed=1)
        params = UncertaintyParams(contradiction_scale=0.5)

        # Small conflict
        small_contradiction = existing.mean + 0.1
        result_small = handle_contradiction(existing, small_contradiction, current_time=1.0, params=params)

        # Large conflict
        large_contradiction = existing.mean + 2.0
        result_large = handle_contradiction(existing, large_contradiction, current_time=1.0, params=params)

        # Large conflict should increase variance more
        small_increase = result_small.variance.mean() - existing.variance.mean()
        large_increase = result_large.variance.mean() - existing.variance.mean()
        assert large_increase > small_increase

    def test_timestamps_updated(self, dim):
        """Should update both timestamps to current_time."""
        existing = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        contradicting = torch.randn(dim)
        params = UncertaintyParams()

        result = handle_contradiction(existing, contradicting, current_time=100.0, params=params)

        assert result.last_accessed == 100.0
        assert result.last_updated == 100.0


class TestBundleObservations:
    """Test suite for bundling multiple observations into one distribution."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a DistributionalHDV."""
        observations = [
            (torch.randn(dim), torch.ones(dim) * 0.5),
            (torch.randn(dim), torch.ones(dim) * 0.5),
        ]
        result = bundle_observations(observations, current_time=1.0)
        assert isinstance(result, DistributionalHDV)

    def test_single_observation_same_as_input(self, dim):
        """Single observation should return essentially the same distribution."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.3
        observations = [(mean, variance)]

        result = bundle_observations(observations, current_time=1.0)

        # Mean should be nearly identical (numerical precision)
        assert torch.allclose(result.mean, mean, atol=1e-6)
        # Variance should be nearly identical (numerical precision)
        assert torch.allclose(result.variance, variance, atol=1e-6)

    def test_more_observations_lower_variance(self, dim):
        """More observations should lead to lower variance (more confidence)."""
        base_mean = torch.randn(dim)
        base_variance = torch.ones(dim) * 0.5

        # 2 observations
        obs_2 = [(base_mean, base_variance), (base_mean, base_variance)]
        result_2 = bundle_observations(obs_2, current_time=1.0)

        # 5 observations
        obs_5 = [(base_mean, base_variance) for _ in range(5)]
        result_5 = bundle_observations(obs_5, current_time=1.0)

        # More observations = lower variance
        assert result_5.variance.mean() < result_2.variance.mean()

    def test_variance_inversely_proportional_to_count(self, dim):
        """Combined variance should be ~1/n for n identical observations."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5

        # 1 observation
        obs_1 = [(mean, variance)]
        result_1 = bundle_observations(obs_1, current_time=1.0)

        # 4 observations (should have ~1/4 the variance)
        obs_4 = [(mean, variance) for _ in range(4)]
        result_4 = bundle_observations(obs_4, current_time=1.0)

        # Variance ratio should be approximately 4:1
        variance_ratio = result_1.variance.mean() / result_4.variance.mean()
        assert variance_ratio == pytest.approx(4.0, rel=0.01)

    def test_mean_is_precision_weighted_average(self, dim):
        """Combined mean should be precision-weighted average of input means."""
        # Create two observations with different precisions
        mean1 = torch.ones(dim)
        var1 = torch.ones(dim) * 0.25  # Precision = 4

        mean2 = torch.ones(dim) * 3.0
        var2 = torch.ones(dim) * 1.0  # Precision = 1

        observations = [(mean1, var1), (mean2, var2)]
        result = bundle_observations(observations, current_time=1.0)

        # Expected mean: (4*1 + 1*3) / (4 + 1) = 7/5 = 1.4
        expected_mean = torch.ones(dim) * 1.4
        assert torch.allclose(result.mean, expected_mean, atol=1e-5)

    def test_empty_raises_error(self):
        """Empty observation list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            bundle_observations([], current_time=1.0)

    def test_timestamps_set(self, dim):
        """Should set both timestamps to current_time."""
        observations = [
            (torch.randn(dim), torch.ones(dim) * 0.5),
        ]
        result = bundle_observations(observations, current_time=42.0)

        assert result.last_accessed == 42.0
        assert result.last_updated == 42.0

    def test_dimension_mismatch_raises(self, dim):
        """Observations with mismatched dimensions should raise ValueError."""
        mean1 = torch.randn(dim)
        var1 = torch.ones(dim) * 0.5

        mean2 = torch.randn(dim + 10)  # Different dimension
        var2 = torch.ones(dim + 10) * 0.5

        with pytest.raises(ValueError, match="mismatch"):
            bundle_observations([(mean1, var1), (mean2, var2)], current_time=1.0)


class TestPropagateThroughEdge:
    """Test suite for propagation with edge uncertainty."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a DistributionalHDV."""
        source = random_distributional(dim, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.8)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams()

        result = propagate_through_edge(source, edge, edge_hdv, params)
        assert isinstance(result, DistributionalHDV)

    def test_mean_transforms_via_bind(self, dim):
        """Mean should be transformed by binding with edge HDV."""
        source = random_distributional(dim, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=1.0)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams()

        result = propagate_through_edge(source, edge, edge_hdv, params)

        expected_mean = source.mean * edge_hdv
        assert torch.allclose(result.mean, expected_mean)

    def test_variance_increases(self, dim):
        """Variance should increase after propagation."""
        source = random_distributional(dim, initial_variance=0.3, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.8)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams()

        result = propagate_through_edge(source, edge, edge_hdv, params)

        assert result.variance.mean() > source.variance.mean()

    def test_low_confidence_edge_more_variance(self, dim):
        """Lower confidence edge should add more variance."""
        source = random_distributional(dim, initial_variance=0.3, seed=1)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams(base_edge_variance=0.2)

        high_conf_edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.9)
        low_conf_edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.3)

        result_high = propagate_through_edge(source, high_conf_edge, edge_hdv, params)
        result_low = propagate_through_edge(source, low_conf_edge, edge_hdv, params)

        assert result_low.variance.mean() > result_high.variance.mean()

    def test_certain_edge_minimal_variance_increase(self, dim):
        """Confidence=1.0 edge should add minimal variance."""
        source = random_distributional(dim, initial_variance=0.3, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=1.0)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams(base_edge_variance=0.2)

        result = propagate_through_edge(source, edge, edge_hdv, params)

        # Variance increase should be minimal (only from the bind operation)
        # Not from edge uncertainty
        var_increase = result.variance.mean() - source.variance.mean()
        # With confidence=1.0, edge_uncertainty contribution is 0
        # So increase is only from the binding operation
        assert var_increase < 0.3  # Reasonable bound

    def test_completely_uncertain_edge_max_variance_increase(self, dim):
        """Confidence=0.0 edge should add maximum edge variance."""
        source = random_distributional(dim, initial_variance=0.3, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.0)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams(base_edge_variance=0.5)

        result = propagate_through_edge(source, edge, edge_hdv, params)

        # Should have added significant variance from edge uncertainty
        assert result.variance.mean() > source.variance.mean() + 0.3

    def test_multi_hop_accumulates_variance(self, dim):
        """Multiple propagation hops should accumulate variance."""
        source = random_distributional(dim, initial_variance=0.2, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.8)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams(base_edge_variance=0.1)

        # One hop
        hop1 = propagate_through_edge(source, edge, edge_hdv, params)

        # Two hops
        hop2 = propagate_through_edge(hop1, edge, edge_hdv, params)

        # Three hops
        hop3 = propagate_through_edge(hop2, edge, edge_hdv, params)

        # Variance should strictly increase with each hop
        assert hop1.variance.mean() > source.variance.mean()
        assert hop2.variance.mean() > hop1.variance.mean()
        assert hop3.variance.mean() > hop2.variance.mean()
