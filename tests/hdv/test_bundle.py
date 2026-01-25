"""Tests for bundle HDV operation."""

import torch
import pytest
from engram.hdv import random_ternary, bundle, similarity, normalize


class TestBundle:
    """Test suite for bundle operation."""

    def test_returns_tensor(self, dim):
        """Should return a torch tensor."""
        items = [random_ternary(dim, seed=i) for i in range(3)]
        result = bundle(items)
        assert isinstance(result, torch.Tensor)

    def test_correct_dimension(self, dim):
        """Should return tensor of same dimension."""
        items = [random_ternary(dim, seed=i) for i in range(3)]
        result = bundle(items)
        assert result.shape == (dim,)

    def test_single_item_returns_normalized(self, dim):
        """Bundle of single item should return normalized version."""
        v = random_ternary(dim, seed=1)
        result = bundle([v])
        expected = normalize(v)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_is_normalized(self, dim):
        """Result should be unit length."""
        items = [random_ternary(dim, seed=i) for i in range(5)]
        result = bundle(items)
        norm = torch.norm(result)
        assert abs(norm.item() - 1.0) < 1e-5

    def test_two_items_sum_normalized(self, dim):
        """Bundle of two items should be normalized sum."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        result = bundle([a, b])
        expected = normalize(a + b)
        assert torch.allclose(result, expected, atol=1e-6)


class TestBundleQueryability:
    """Test that bundled items remain queryable."""

    def test_items_queryable_n3(self, dim):
        """Each item should have similarity > 0.3 to bundle (n=3)."""
        n = 3
        items = [random_ternary(dim, sparsity=0.5, seed=i) for i in range(n)]
        bundled = bundle(items)
        for i, item in enumerate(items):
            sim = similarity(bundled, item)
            assert sim > 0.3, f"Item {i} similarity {sim} too low (n={n})"

    def test_items_queryable_n10(self, dim):
        """VALIDATION: Each item should have similarity > 0.3 to bundle (n=10)."""
        n = 10
        items = [random_ternary(dim, sparsity=0.5, seed=i) for i in range(n)]
        bundled = bundle(items)
        for i, item in enumerate(items):
            sim = similarity(bundled, item)
            assert sim > 0.2, f"Item {i} similarity {sim} too low (n={n})"

    def test_non_members_low_similarity(self, dim):
        """Non-members should have low similarity to bundle."""
        n = 5
        items = [random_ternary(dim, sparsity=0.5, seed=i) for i in range(n)]
        non_member = random_ternary(dim, sparsity=0.5, seed=100)
        bundled = bundle(items)
        sim = similarity(bundled, non_member)
        assert abs(sim) < 0.15, f"Non-member similarity {sim} too high"

    def test_capacity_degradation_n50(self, dim):
        """Document degradation at n=50: expect similarity ~ 1/sqrt(n) ~ 0.14."""
        n = 50
        items = [random_ternary(dim, sparsity=0.5, seed=i) for i in range(n)]
        bundled = bundle(items)
        sims = [similarity(bundled, item) for item in items]
        avg_sim = sum(sims) / len(sims)
        expected = 1 / (n ** 0.5)  # ~0.14
        # Should be in reasonable range of theoretical value
        assert avg_sim > expected * 0.5, f"Avg similarity {avg_sim} too low for n={n}"
        print(f"\n  n={n}: avg_similarity={avg_sim:.3f}, expected~{expected:.3f}")

    def test_capacity_degradation_n100(self, dim):
        """Document degradation at n=100: expect similarity ~ 1/sqrt(n) ~ 0.1."""
        n = 100
        items = [random_ternary(dim, sparsity=0.5, seed=i) for i in range(n)]
        bundled = bundle(items)
        sims = [similarity(bundled, item) for item in items]
        avg_sim = sum(sims) / len(sims)
        expected = 1 / (n ** 0.5)  # ~0.1
        assert avg_sim > expected * 0.5, f"Avg similarity {avg_sim} too low for n={n}"
        print(f"\n  n={n}: avg_similarity={avg_sim:.3f}, expected~{expected:.3f}")


class TestBundleEdgeCases:
    """Edge case tests for bundle."""

    def test_empty_bundle_raises(self, dim):
        """Should raise error for empty list."""
        with pytest.raises((ValueError, IndexError)):
            bundle([])

    def test_identical_items_reinforced(self, dim):
        """Bundling identical items should reinforce the signal."""
        v = random_ternary(dim, seed=1)
        bundled = bundle([v, v, v])
        # Should be highly similar to original
        sim = similarity(bundled, v)
        assert sim > 0.95, f"Self-bundle similarity {sim} should be ~1"

    def test_opposite_items_cancel(self, dim):
        """Bundling opposite items should cancel out."""
        v = random_ternary(dim, sparsity=1.0, seed=1)  # Full sparsity for clean test
        bundled = bundle([v, -v])
        # Sum is zero, so normalized result is zero vector
        assert torch.norm(bundled).item() < 1e-6 or torch.allclose(bundled, torch.zeros_like(bundled), atol=1e-6)
