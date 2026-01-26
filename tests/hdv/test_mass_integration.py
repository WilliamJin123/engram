"""Tests for mass-integrated Bayesian updates."""

import torch
import pytest
from engram.hdv import random_distributional, DistributionalHDV
from engram.hdv.uncertainty import bayesian_update_with_mass, UncertaintyParams


class TestMassIntegratedUpdates:
    """Verify mass modulates belief updates."""

    def test_high_mass_resists_change(self, dim):
        """High-mass nodes should update less than low-mass nodes.

        Scenario: Same observation applied to high-mass and low-mass priors.
        High-mass prior should move less toward the observation.
        """
        # Create identical priors
        prior_low_mass = random_distributional(dim, initial_variance=0.5, seed=42)
        prior_high_mass = random_distributional(dim, initial_variance=0.5, seed=42)

        # Same observation
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        # Update with different mass values
        posterior_low = bayesian_update_with_mass(
            prior_low_mass, observation, obs_variance,
            mass=1.0, current_time=1.0
        )
        posterior_high = bayesian_update_with_mass(
            prior_high_mass, observation, obs_variance,
            mass=100.0, current_time=1.0
        )

        # Compute how much each moved
        movement_low = torch.norm(posterior_low.mean - prior_low_mass.mean).item()
        movement_high = torch.norm(posterior_high.mean - prior_high_mass.mean).item()

        # High mass should move less
        assert movement_high < movement_low
        # Significant difference (not just numerical noise)
        assert movement_low > movement_high * 2

    def test_low_mass_updates_more_than_high_mass(self, dim):
        """Lower mass values should allow more update than higher mass.

        Verifies the monotonic relationship between mass and resistance.
        """
        prior = random_distributional(dim, initial_variance=0.5, seed=42)
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        # Mass-aware updates with increasing mass
        posterior_1 = bayesian_update_with_mass(
            prior, observation, obs_variance, mass=0.1, current_time=1.0
        )
        posterior_2 = bayesian_update_with_mass(
            prior, observation, obs_variance, mass=1.0, current_time=1.0
        )
        posterior_3 = bayesian_update_with_mass(
            prior, observation, obs_variance, mass=10.0, current_time=1.0
        )

        # Compute movements
        movement_1 = torch.norm(posterior_1.mean - prior.mean).item()
        movement_2 = torch.norm(posterior_2.mean - prior.mean).item()
        movement_3 = torch.norm(posterior_3.mean - prior.mean).item()

        # Monotonically decreasing movement as mass increases
        assert movement_1 > movement_2 > movement_3

    def test_mass_affects_variance_reduction(self, dim):
        """High mass should also reduce variance less aggressively."""
        prior = random_distributional(dim, initial_variance=0.5, seed=42)
        observation = prior.mean + torch.randn(dim) * 0.1
        obs_variance = torch.ones(dim) * 0.2

        posterior_low = bayesian_update_with_mass(
            prior, observation, obs_variance, mass=1.0, current_time=1.0
        )
        posterior_high = bayesian_update_with_mass(
            prior, observation, obs_variance, mass=50.0, current_time=1.0
        )

        # Both should reduce variance
        assert posterior_low.variance.mean() < prior.variance.mean()
        assert posterior_high.variance.mean() < prior.variance.mean()

        # High mass reduces less (stays more uncertain about new info)
        assert posterior_high.variance.mean() > posterior_low.variance.mean()
