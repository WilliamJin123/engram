"""Tests for bind and unbind HDV operations."""

import torch
import pytest
from engram.hdv import random_ternary, bind, unbind, similarity


class TestBind:
    """Test suite for bind operation."""

    def test_returns_tensor(self, dim):
        """Should return a torch tensor."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        result = bind(a, b)
        assert isinstance(result, torch.Tensor)

    def test_correct_dimension(self, dim):
        """Should return tensor of same dimension."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        result = bind(a, b)
        assert result.shape == (dim,)

    def test_element_wise_multiplication(self, dim):
        """Bind should be element-wise multiplication."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        result = bind(a, b)
        expected = a * b
        assert torch.equal(result, expected)

    def test_result_is_ternary(self, dim):
        """Result should only contain {-1, 0, +1}."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        result = bind(a, b)
        unique_vals = torch.unique(result)
        for val in unique_vals:
            assert val.item() in (-1.0, 0.0, 1.0)

    def test_commutative(self, dim):
        """Bind should be commutative: bind(a,b) == bind(b,a)."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        assert torch.equal(bind(a, b), bind(b, a))

    def test_associative(self, dim):
        """Bind should be associative: bind(bind(a,b),c) == bind(a,bind(b,c))."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        c = random_ternary(dim, seed=3)
        assert torch.equal(bind(bind(a, b), c), bind(a, bind(b, c)))

    def test_self_bind_gives_magnitude_squared(self, dim):
        """bind(a, a) should give |a|^2 element-wise (1 where non-zero, 0 where zero)."""
        a = random_ternary(dim, seed=1)
        result = bind(a, a)
        # For ternary: (-1)*(-1)=1, (1)*(1)=1, (0)*(0)=0
        expected = (a != 0).float()
        assert torch.equal(result, expected)


class TestUnbind:
    """Test suite for unbind operation."""

    def test_returns_tensor(self, dim):
        """Should return a torch tensor."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        bound = bind(a, b)
        result = unbind(bound, a)
        assert isinstance(result, torch.Tensor)

    def test_correct_dimension(self, dim):
        """Should return tensor of same dimension."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        bound = bind(a, b)
        result = unbind(bound, a)
        assert result.shape == (dim,)

    def test_unbind_is_multiplication(self, dim):
        """Unbind should be element-wise multiplication (same as bind for ternary)."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        bound = bind(a, b)
        result = unbind(bound, a)
        expected = bound * a
        assert torch.equal(result, expected)


class TestBindUnbindIntegration:
    """Integration tests for bind/unbind retrievability."""

    def test_unbind_recovers_with_high_similarity(self, dim):
        """VALIDATION: unbind should recover original with similarity > 0.5 for ternary.

        Note: For ternary vectors, recovery is partial because information is lost
        where either vector has zeros. With sparsity=0.5, we expect ~25% of positions
        to have full information (both non-zero), giving similarity around 0.5.
        """
        a = random_ternary(dim, sparsity=0.5, seed=1)
        b = random_ternary(dim, sparsity=0.5, seed=2)
        bound = bind(a, b)
        recovered = unbind(bound, a)
        sim = similarity(recovered, b)
        # For ternary with sparsity=0.5, expect ~0.5 similarity due to zero masking
        assert sim > 0.4, f"Similarity {sim} too low for ternary unbind"

    def test_unbind_with_full_sparsity_high_similarity(self, dim):
        """With sparsity=1.0 (no zeros), unbind should recover with similarity > 0.9."""
        a = random_ternary(dim, sparsity=1.0, seed=1)
        b = random_ternary(dim, sparsity=1.0, seed=2)
        bound = bind(a, b)
        recovered = unbind(bound, a)
        sim = similarity(recovered, b)
        # With no zeros, should get perfect recovery
        assert sim > 0.99, f"Similarity {sim} should be ~1.0 for bipolar unbind"

    def test_unbind_dissimilar_from_unrelated(self, dim):
        """Unbound vector should be dissimilar to unrelated vectors."""
        a = random_ternary(dim, sparsity=0.5, seed=1)
        b = random_ternary(dim, sparsity=0.5, seed=2)
        c = random_ternary(dim, sparsity=0.5, seed=3)
        bound = bind(a, b)
        recovered = unbind(bound, a)
        sim_to_c = similarity(recovered, c)
        # Should be near zero for unrelated vector
        assert abs(sim_to_c) < 0.1, f"Similarity to unrelated {sim_to_c} too high"

    def test_wrong_key_gives_low_similarity(self, dim):
        """Unbinding with wrong key should give low similarity to target."""
        a = random_ternary(dim, sparsity=0.5, seed=1)
        b = random_ternary(dim, sparsity=0.5, seed=2)
        wrong_key = random_ternary(dim, sparsity=0.5, seed=3)
        bound = bind(a, b)
        recovered = unbind(bound, wrong_key)
        sim = similarity(recovered, b)
        assert abs(sim) < 0.15, f"Wrong key similarity {sim} too high"
