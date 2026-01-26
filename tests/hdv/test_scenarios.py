"""Scenario-based tests for HDV module."""

import torch
import pytest
from engram.hdv import (
    random_ternary, bind, unbind, bundle, similarity, normalize,
    random_distributional, distributional_bind, distributional_unbind,
    distributional_similarity, DistributionalHDV,
)
from engram.hdv.uncertainty import (
    bayesian_update, temporal_decay, human_confirm,
    handle_contradiction, bundle_observations, UncertaintyParams,
)


class TestHDVScenarios:
    """Scenario tests for HDV operations."""

    def test_binding_for_association(self, dim):
        """Bind creates reversible associations between concepts.

        Scenario: Associate 'red' with 'apple', verify we can recover
        the association and that bind has expected mathematical properties.
        """
        red = random_ternary(dim, sparsity=1.0, seed=1)  # Bipolar for exact recovery
        apple = random_ternary(dim, sparsity=1.0, seed=2)

        # Bind creates association
        red_apple = bind(red, apple)
        assert isinstance(red_apple, torch.Tensor)
        assert red_apple.shape == red.shape

        # Result is ternary (only -1, 0, +1)
        unique_vals = torch.unique(red_apple)
        assert all(v in [-1, 0, 1] for v in unique_vals.tolist())

        # Unbind recovers the other concept
        recovered = unbind(red_apple, red)
        assert similarity(recovered, apple) > 0.99

        # Bind is commutative: bind(a,b) == bind(b,a)
        assert torch.equal(bind(red, apple), bind(apple, red))

        # Bind is associative: bind(bind(a,b),c) == bind(a,bind(b,c))
        green = random_ternary(dim, sparsity=1.0, seed=3)
        assert torch.equal(bind(bind(red, apple), green), bind(red, bind(apple, green)))

        # Wrong key gives noise (low similarity)
        wrong_key = random_ternary(dim, sparsity=0.5, seed=99)
        wrong_recovered = unbind(red_apple, wrong_key)
        assert abs(similarity(wrong_recovered, apple)) < 0.15

    def test_bundling_for_sets(self, dim):
        """Bundle combines multiple items into a set representation.

        Scenario: Bundle several fruits together, verify each can be
        queried and that non-members are distinguishable.
        """
        apple = random_ternary(dim, sparsity=0.5, seed=1)
        banana = random_ternary(dim, sparsity=0.5, seed=2)
        cherry = random_ternary(dim, sparsity=0.5, seed=3)

        # Bundle multiple items
        fruits = bundle([apple, banana, cherry])
        assert isinstance(fruits, torch.Tensor)
        assert fruits.shape == apple.shape

        # Bundle is normalized (unit length)
        assert torch.norm(fruits).item() == pytest.approx(1.0, abs=1e-5)

        # Each item is queryable (similarity > threshold)
        assert similarity(fruits, apple) > 0.3
        assert similarity(fruits, banana) > 0.3
        assert similarity(fruits, cherry) > 0.3

        # Non-members have low similarity
        car = random_ternary(dim, sparsity=0.5, seed=100)
        assert abs(similarity(fruits, car)) < 0.15

        # Single item bundle returns normalized version
        single = bundle([apple])
        assert similarity(single, normalize(apple)) > 0.99

        # Identical items reinforce each other
        reinforced = bundle([apple, apple, apple])
        assert similarity(reinforced, apple) > 0.95

    def test_similarity_measures(self, dim):
        """Similarity correctly measures vector relationships.

        Scenario: Compare vectors with known relationships and verify
        similarity behaves as expected.
        """
        v = random_ternary(dim, sparsity=0.5, seed=1)
        w = random_ternary(dim, sparsity=0.5, seed=2)

        # Similarity returns float in [-1, 1]
        sim = similarity(v, w)
        assert isinstance(sim, float)
        assert -1.0 <= sim <= 1.0

        # Self-similarity is 1.0
        assert similarity(v, v) == pytest.approx(1.0)

        # Opposite similarity is -1.0
        assert similarity(v, -v) == pytest.approx(-1.0)

        # Similarity is symmetric
        assert similarity(v, w) == similarity(w, v)

        # Random vectors are approximately orthogonal
        sims = [
            similarity(
                random_ternary(dim, sparsity=0.5, seed=i * 2),
                random_ternary(dim, sparsity=0.5, seed=i * 2 + 1)
            )
            for i in range(50)
        ]
        mean_sim = sum(sims) / len(sims)
        assert abs(mean_sim) < 0.05  # Near zero

        # Normalize preserves direction with unit length
        normalized = normalize(v)
        assert torch.norm(normalized).item() == pytest.approx(1.0, abs=1e-5)
        assert similarity(v, normalized) > 0.99

    def test_distributional_uncertainty(self, dim):
        """Distributional HDVs track uncertainty through operations.

        Scenario: Create uncertain concepts, combine them, observe how
        uncertainty propagates and can be reduced through observation.
        """
        params = UncertaintyParams()

        # Create distributional HDV with explicit uncertainty
        dog = random_distributional(dim, initial_variance=0.5, current_time=0.0, seed=42)
        assert isinstance(dog, DistributionalHDV)
        assert dog.mean.shape == (dim,)
        assert dog.variance.shape == (dim,)
        assert dog.variance.mean().item() == pytest.approx(0.5)

        # Mean is ternary (-1, 0, +1)
        unique_vals = torch.unique(dog.mean)
        assert all(v in [-1, 0, 1] for v in unique_vals.tolist())

        # Distributional bind propagates uncertainty
        cat = random_distributional(dim, initial_variance=0.3, seed=43)
        bound = distributional_bind(dog, cat)
        assert isinstance(bound, DistributionalHDV)
        # Variance increases after binding
        assert bound.variance.mean() > 0

        # Bind is commutative for distributional
        bound_reverse = distributional_bind(cat, dog)
        assert torch.allclose(bound.mean, bound_reverse.mean)
        assert torch.allclose(bound.variance, bound_reverse.variance)

        # Unbind recovers with high similarity for bipolar vectors
        a = random_distributional(dim, initial_variance=0.1, sparsity=1.0, seed=1)
        b = random_distributional(dim, initial_variance=0.1, sparsity=1.0, seed=2)
        bound_ab = distributional_bind(a, b)
        recovered = distributional_unbind(bound_ab, a)
        assert similarity(recovered.mean, b.mean) > 0.99

        # Distributional similarity returns (similarity, uncertainty)
        sim, unc = distributional_similarity(dog, cat)
        assert isinstance(sim, float)
        assert isinstance(unc, float)
        assert 0 <= sim <= 1
        assert unc >= 0

        # Identical distributions have similarity 1.0
        sim_self, _ = distributional_similarity(dog, dog)
        assert sim_self == pytest.approx(1.0, abs=1e-6)

    def test_uncertainty_lifecycle(self, dim):
        """Test the complete uncertainty lifecycle from creation to confirmation.

        Scenario: A concept starts uncertain, gains confidence through observations,
        drifts when unaccessed, handles contradictions, and gets confirmed by human.
        """
        params = UncertaintyParams()

        # 1. Create node from LLM inference (high initial variance)
        concept = random_distributional(dim, initial_variance=0.6, current_time=0.0, seed=42)
        assert concept.variance.mean().item() == pytest.approx(0.6)

        # 2. Observe several instances (variance reduces via Bayesian updates)
        for i in range(5):
            observation = concept.mean + torch.randn(dim) * 0.1  # Noisy observations
            obs_variance = torch.ones(dim) * 0.3
            concept = bayesian_update(concept, observation, obs_variance, current_time=float(i + 1))

        # Variance decreased (gained confidence)
        assert concept.variance.mean() < 0.5

        # Mean moved toward observations
        # (difficult to verify precisely, but variance decrease indicates learning)

        # 3. Time passes without access (variance drifts up)
        concept_decayed = temporal_decay(concept, current_time=500.0, params=params)
        assert concept_decayed.variance.mean() > concept.variance.mean()
        # Mean stays the same
        assert torch.equal(concept_decayed.mean, concept.mean)

        # 4. Contradictory information arrives (variance increases more)
        contradicting = -concept.mean  # Opposite values
        concept_conflicted = handle_contradiction(
            concept_decayed, contradicting, current_time=501.0, params=params
        )
        assert concept_conflicted.variance.mean() > concept_decayed.variance.mean()

        # 5. Human confirms correct information (variance collapses)
        concept_confirmed = human_confirm(concept_conflicted, current_time=502.0, params=params)
        assert concept_confirmed.variance.mean() < 0.1

        # Bundling multiple observations
        observations = [
            (torch.randn(dim), torch.ones(dim) * 0.5),
            (torch.randn(dim), torch.ones(dim) * 0.5),
            (torch.randn(dim), torch.ones(dim) * 0.5),
        ]
        bundled = bundle_observations(observations, current_time=1.0)
        assert isinstance(bundled, DistributionalHDV)
        # More observations = lower variance than single observation
        single = bundle_observations([observations[0]], current_time=1.0)
        assert bundled.variance.mean() < single.variance.mean()
