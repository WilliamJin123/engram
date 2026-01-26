"""Tests for strength-integrated Bayesian updates."""

import torch
import pytest
from engram.hdv import random_distributional, DistributionalHDV
from engram.hdv.uncertainty import bayesian_update_with_strength, UncertaintyParams


class TestStrengthIntegratedUpdates:
    """Verify strength modulates belief updates."""

    def test_high_strength_resists_change(self, dim):
        """High-strength nodes should update less than low-strength nodes.

        Scenario: Same observation applied to high-strength and low-strength priors.
        High-strength prior should move less toward the observation.
        """
        # Create identical priors
        prior_low_strength = random_distributional(dim, initial_variance=0.5, seed=42)
        prior_high_strength = random_distributional(dim, initial_variance=0.5, seed=42)

        # Same observation
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        # Update with different strength values
        posterior_low = bayesian_update_with_strength(
            prior_low_strength, observation, obs_variance,
            strength=1.0, current_time=1.0
        )
        posterior_high = bayesian_update_with_strength(
            prior_high_strength, observation, obs_variance,
            strength=100.0, current_time=1.0
        )

        # Compute how much each moved
        movement_low = torch.norm(posterior_low.mean - prior_low_strength.mean).item()
        movement_high = torch.norm(posterior_high.mean - prior_high_strength.mean).item()

        # High strength should move less
        assert movement_high < movement_low
        # Significant difference (not just numerical noise)
        assert movement_low > movement_high * 2

    def test_low_strength_updates_more_than_high_strength(self, dim):
        """Lower strength values should allow more update than higher strength.

        Verifies the monotonic relationship between strength and resistance.
        """
        prior = random_distributional(dim, initial_variance=0.5, seed=42)
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        # Strength-aware updates with increasing strength
        posterior_1 = bayesian_update_with_strength(
            prior, observation, obs_variance, strength=0.1, current_time=1.0
        )
        posterior_2 = bayesian_update_with_strength(
            prior, observation, obs_variance, strength=1.0, current_time=1.0
        )
        posterior_3 = bayesian_update_with_strength(
            prior, observation, obs_variance, strength=10.0, current_time=1.0
        )

        # Compute movements
        movement_1 = torch.norm(posterior_1.mean - prior.mean).item()
        movement_2 = torch.norm(posterior_2.mean - prior.mean).item()
        movement_3 = torch.norm(posterior_3.mean - prior.mean).item()

        # Monotonically decreasing movement as strength increases
        assert movement_1 > movement_2 > movement_3

    def test_strength_affects_variance_reduction(self, dim):
        """High strength should also reduce variance less aggressively."""
        prior = random_distributional(dim, initial_variance=0.5, seed=42)
        observation = prior.mean + torch.randn(dim) * 0.1
        obs_variance = torch.ones(dim) * 0.2

        posterior_low = bayesian_update_with_strength(
            prior, observation, obs_variance, strength=1.0, current_time=1.0
        )
        posterior_high = bayesian_update_with_strength(
            prior, observation, obs_variance, strength=50.0, current_time=1.0
        )

        # Both should reduce variance
        assert posterior_low.variance.mean() < prior.variance.mean()
        assert posterior_high.variance.mean() < prior.variance.mean()

        # High strength reduces less (stays more uncertain about new info)
        assert posterior_high.variance.mean() > posterior_low.variance.mean()
