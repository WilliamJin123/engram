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
