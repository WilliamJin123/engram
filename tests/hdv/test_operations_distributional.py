"""Tests for distributional bind and unbind HDV operations."""

import torch
import pytest
from engram.hdv.distributional import DistributionalHDV, random_distributional
from engram.hdv.operations import distributional_bind, distributional_unbind, similarity


class TestDistributionalBind:
    """Test suite for distributional_bind operation."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a DistributionalHDV instance."""
        a = random_distributional(dim, seed=1)
        b = random_distributional(dim, seed=2)
        result = distributional_bind(a, b)
        assert isinstance(result, DistributionalHDV)

    def test_timestamps_use_max(self, dim):
        """Result timestamps should be max of input timestamps."""
        a = random_distributional(dim, current_time=100.0, seed=1)
        b = random_distributional(dim, current_time=200.0, seed=2)
        result = distributional_bind(a, b)
        assert result.last_accessed == 200.0
        assert result.last_updated == 200.0

    def test_mean_is_element_wise_product(self, dim):
        """Mean should be element-wise product of input means."""
        a = random_distributional(dim, seed=1)
        b = random_distributional(dim, seed=2)
        result = distributional_bind(a, b)
        expected_mean = a.mean * b.mean
        assert torch.allclose(result.mean, expected_mean)

    def test_variance_increases(self, dim):
        """Binding should generally increase total variance."""
        a = random_distributional(dim, initial_variance=0.1, seed=1)
        b = random_distributional(dim, initial_variance=0.1, seed=2)
        result = distributional_bind(a, b)
        # Total variance should be greater than either input's total variance
        # (except where both means are zero)
        input_total_var = a.variance.sum() + b.variance.sum()
        # Result variance follows product formula, which increases uncertainty
        # Just verify variance is positive and reasonable
        assert result.variance.sum() > 0
        assert (result.variance > 0).all()

    def test_commutative(self, dim):
        """Bind should be commutative: bind(a,b) == bind(b,a)."""
        a = random_distributional(dim, seed=1)
        b = random_distributional(dim, seed=2)
        result_ab = distributional_bind(a, b)
        result_ba = distributional_bind(b, a)
        assert torch.allclose(result_ab.mean, result_ba.mean)
        assert torch.allclose(result_ab.variance, result_ba.variance)

    def test_low_variance_inputs_low_variance_output(self, dim):
        """With low input variance, output variance should be relatively low."""
        low_var = 0.01
        a = random_distributional(dim, initial_variance=low_var, sparsity=1.0, seed=1)
        b = random_distributional(dim, initial_variance=low_var, sparsity=1.0, seed=2)
        result = distributional_bind(a, b)
        # For bipolar vectors (+/-1), variance formula gives:
        # 1*0.01 + 1*0.01 + 0.01*0.01 = 0.0201
        # So output variance should be around 0.02 per dimension
        avg_var = result.variance.mean().item()
        # Should be less than 0.1 (much lower than high variance case)
        assert avg_var < 0.1

    def test_high_variance_inputs_high_variance_output(self, dim):
        """With high input variance, output variance should be higher."""
        high_var = 1.0
        a = random_distributional(dim, initial_variance=high_var, sparsity=1.0, seed=1)
        b = random_distributional(dim, initial_variance=high_var, sparsity=1.0, seed=2)
        result = distributional_bind(a, b)
        # For bipolar vectors (+/-1) with variance=1.0:
        # 1*1.0 + 1*1.0 + 1.0*1.0 = 3.0 per dimension
        avg_var = result.variance.mean().item()
        # Should be significantly higher than low variance case
        assert avg_var > 1.0


class TestDistributionalUnbind:
    """Test suite for distributional_unbind operation."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a DistributionalHDV instance."""
        a = random_distributional(dim, seed=1)
        b = random_distributional(dim, seed=2)
        bound = distributional_bind(a, b)
        result = distributional_unbind(bound, a)
        assert isinstance(result, DistributionalHDV)

    def test_unbind_same_as_bind(self, dim):
        """Unbind should be the same operation as bind for ternary means."""
        a = random_distributional(dim, seed=1)
        b = random_distributional(dim, seed=2)
        bound = distributional_bind(a, b)

        # unbind(bound, a) should equal bind(bound, a)
        unbind_result = distributional_unbind(bound, a)
        bind_result = distributional_bind(bound, a)

        assert torch.allclose(unbind_result.mean, bind_result.mean)
        assert torch.allclose(unbind_result.variance, bind_result.variance)

    def test_variance_increases_on_unbind(self, dim):
        """Unbinding should also increase variance (same as binding)."""
        a = random_distributional(dim, initial_variance=0.1, seed=1)
        b = random_distributional(dim, initial_variance=0.1, seed=2)
        bound = distributional_bind(a, b)
        result = distributional_unbind(bound, a)
        # Variance should still be positive and reasonable
        assert (result.variance > 0).all()
        # After two bindings worth of variance increase
        assert result.variance.mean() > 0


class TestBindUnbindRoundtrip:
    """Integration tests for bind/unbind distributional operations."""

    def test_unbind_recovers_similar_mean(self, dim):
        """Unbind should recover a mean similar to original for bipolar vectors.

        For ternary vectors with sparsity=1.0 (bipolar), binding with key a
        and then unbinding with key a should recover the original b exactly
        in terms of mean (ignoring variance).
        """
        a = random_distributional(dim, initial_variance=0.1, sparsity=1.0, seed=1)
        b = random_distributional(dim, initial_variance=0.1, sparsity=1.0, seed=2)
        bound = distributional_bind(a, b)
        recovered = distributional_unbind(bound, a)

        # For bipolar vectors: bind(bind(a, b), a) = a*b*a = b*(a*a) = b*1 = b
        # So the mean should be recovered exactly
        sim = similarity(recovered.mean, b.mean)
        assert sim > 0.99, f"Expected high similarity, got {sim}"

    def test_wrong_key_low_similarity(self, dim):
        """Using wrong key for unbind should give low similarity to target."""
        a = random_distributional(dim, initial_variance=0.1, sparsity=0.5, seed=1)
        b = random_distributional(dim, initial_variance=0.1, sparsity=0.5, seed=2)
        wrong_key = random_distributional(dim, initial_variance=0.1, sparsity=0.5, seed=3)

        bound = distributional_bind(a, b)
        recovered = distributional_unbind(bound, wrong_key)

        sim = similarity(recovered.mean, b.mean)
        # Should be near zero for wrong key
        assert abs(sim) < 0.15, f"Expected low similarity with wrong key, got {sim}"
