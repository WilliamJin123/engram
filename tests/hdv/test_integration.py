"""Integration tests for the complete distributional HDV system."""

import torch
import pytest
from engram import (
    DistributionalHDV,
    UncertaintyParams,
    Edge,
    random_distributional,
    distributional_bind,
    distributional_unbind,
    distributional_similarity,
    bayesian_update,
    temporal_decay,
    human_confirm,
    handle_contradiction,
    bundle_observations,
    propagate_through_edge,
)


class TestCompleteLifecycle:
    """Test the complete lifecycle from the design doc."""

    def test_lifecycle_example(self, dim):
        """Test the complete lifecycle from creation to confirmation."""
        params = UncertaintyParams()

        # 1. Create node from LLM inference (high initial variance)
        dog = random_distributional(dim, initial_variance=0.6, current_time=0.0, seed=42)
        assert dog.variance.mean().item() == pytest.approx(0.6)

        # 2. Observe several dogs (variance reduces via updates)
        for i in range(10):
            observation = dog.mean + torch.randn(dim) * 0.1  # Noisy observations
            obs_variance = torch.ones(dim) * 0.3
            dog = bayesian_update(dog, observation, obs_variance, current_time=float(i + 1))

        # Variance should have decreased significantly
        assert dog.variance.mean() < 0.3

        # 3. Time passes without access (variance drifts up)
        dog_decayed = temporal_decay(dog, current_time=500.0, params=params)
        assert dog_decayed.variance.mean() > dog.variance.mean()

        # 4. Contradictory information arrives
        contradicting = -dog.mean  # Opposite values
        dog_conflicted = handle_contradiction(
            dog_decayed, contradicting, current_time=501.0, params=params
        )
        assert dog_conflicted.variance.mean() > dog_decayed.variance.mean()

        # 5. Human confirms correct information
        confirmed_mean = random_distributional(dim, seed=100).mean
        dog_confirmed = human_confirm(
            dog_conflicted, current_time=502.0, params=params, confirmed_mean=confirmed_mean
        )

        # Should be near-certain now
        assert dog_confirmed.variance.mean() < 0.1

    def test_similarity_returns_uncertainty(self, dim):
        """Similarity queries should return uncertainty estimates."""
        dog = random_distributional(dim, initial_variance=0.3, seed=1)
        wolf = random_distributional(dim, initial_variance=0.5, seed=2)

        sim, uncertainty = distributional_similarity(dog, wolf)

        assert 0 < sim < 1
        assert uncertainty > 0
        assert isinstance(sim, float)
        assert isinstance(uncertainty, float)

    def test_bundling_reduces_variance(self, dim):
        """Bundling multiple observations reduces variance."""
        params = UncertaintyParams()

        # Single observation
        single_obs = [(torch.randn(dim), torch.ones(dim) * 0.5)]
        single = bundle_observations(single_obs, current_time=0.0)

        # Ten observations of similar things
        many_obs = [
            (torch.randn(dim) * 0.1, torch.ones(dim) * 0.5)
            for _ in range(10)
        ]
        many = bundle_observations(many_obs, current_time=0.0)

        # Many observations should be more confident
        assert many.variance.mean() < single.variance.mean()

    def test_propagation_compounds_uncertainty(self, dim):
        """Information traveling through graph accumulates variance."""
        params = UncertaintyParams(base_edge_variance=0.1)

        source = random_distributional(dim, initial_variance=0.2, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.8)
        edge_hdv = random_distributional(dim, seed=2).mean

        # Three hops
        hop1 = propagate_through_edge(source, edge, edge_hdv, params)
        hop2 = propagate_through_edge(hop1, edge, edge_hdv, params)
        hop3 = propagate_through_edge(hop2, edge, edge_hdv, params)

        # Variance strictly increases
        assert hop1.variance.mean() > source.variance.mean()
        assert hop2.variance.mean() > hop1.variance.mean()
        assert hop3.variance.mean() > hop2.variance.mean()


class TestInitialVarianceBySource:
    """Test context-dependent initial variance."""

    def test_human_direct_low_variance(self, dim):
        """Human-sourced info should start with low variance."""
        hdv = random_distributional(dim, initial_variance=0.05, seed=1)
        assert hdv.variance.mean().item() == pytest.approx(0.05)

    def test_llm_inference_high_variance(self, dim):
        """LLM-inferred info should start with high variance."""
        hdv = random_distributional(dim, initial_variance=0.6, seed=1)
        assert hdv.variance.mean().item() == pytest.approx(0.6)

    def test_propagated_moderate_variance(self, dim):
        """Propagated info should have moderate variance."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=1)
        assert hdv.variance.mean().item() == pytest.approx(0.5)


class TestEdgeConfidenceEffects:
    """Test that edge confidence affects propagation correctly."""

    def test_high_confidence_edge_preserves_info(self, dim):
        """High confidence edges should preserve information better."""
        params = UncertaintyParams(base_edge_variance=0.3)

        source = random_distributional(dim, initial_variance=0.2, seed=1)
        edge_hdv = random_distributional(dim, seed=2).mean

        high_conf = Edge(source="a", target="b", edge_type="IS_A", confidence=0.95)
        low_conf = Edge(source="a", target="b", edge_type="IS_A", confidence=0.3)

        result_high = propagate_through_edge(source, high_conf, edge_hdv, params)
        result_low = propagate_through_edge(source, low_conf, edge_hdv, params)

        # High confidence should have lower variance increase
        assert result_high.variance.mean() < result_low.variance.mean()

    def test_edge_uncertainty_formula(self, dim):
        """Edge uncertainty contribution should be (1-confidence) * base_edge_variance."""
        params = UncertaintyParams(base_edge_variance=0.5)

        source = random_distributional(dim, initial_variance=0.2, seed=1)
        edge_hdv = torch.ones(dim)  # No binding effect
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.6)

        result = propagate_through_edge(source, edge, edge_hdv, params)

        # Expected edge uncertainty contribution: 0.4 * 0.5 = 0.2
        expected_var = source.variance + 0.2
        assert torch.allclose(result.variance, expected_var, atol=0.01)
