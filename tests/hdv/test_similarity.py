"""Tests for similarity and normalize HDV operations."""

import torch
import pytest
from engram.hdv import random_ternary, similarity, normalize


class TestSimilarity:
    """Test suite for similarity function."""

    def test_returns_float(self, dim):
        """Should return a Python float."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        result = similarity(a, b)
        assert isinstance(result, float)

    def test_range_bounds(self, dim):
        """Should return value in [-1, 1]."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        result = similarity(a, b)
        assert -1.0 <= result <= 1.0

    def test_self_similarity_is_one(self, dim):
        """Similarity of vector with itself should be 1."""
        v = random_ternary(dim, seed=1)
        result = similarity(v, v)
        assert abs(result - 1.0) < 1e-6

    def test_opposite_similarity_is_negative_one(self, dim):
        """Similarity of vector with its negation should be -1."""
        v = random_ternary(dim, sparsity=1.0, seed=1)  # Full sparsity for clean test
        result = similarity(v, -v)
        assert abs(result - (-1.0)) < 1e-6

    def test_symmetric(self, dim):
        """Similarity should be symmetric: sim(a,b) == sim(b,a)."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        assert abs(similarity(a, b) - similarity(b, a)) < 1e-6

    def test_orthogonal_vectors_near_zero(self, dim):
        """Random high-dimensional vectors should be nearly orthogonal."""
        a = random_ternary(dim, seed=1)
        b = random_ternary(dim, seed=2)
        result = similarity(a, b)
        # For 10K dimensions, expect |sim| < 0.1
        assert abs(result) < 0.1

    def test_zero_vector_similarity(self, dim):
        """Similarity with zero vector should be 0."""
        v = random_ternary(dim, seed=1)
        zero = torch.zeros(dim)
        assert similarity(v, zero) == 0.0
        assert similarity(zero, v) == 0.0


class TestRandomOrthogonality:
    """Test orthogonality properties of random vectors."""

    def test_mean_orthogonality(self, dim):
        """Mean similarity of random pairs should be near zero."""
        n_pairs = 50
        sims = []
        for i in range(n_pairs):
            a = random_ternary(dim, seed=i * 2)
            b = random_ternary(dim, seed=i * 2 + 1)
            sims.append(similarity(a, b))
        mean_sim = sum(sims) / len(sims)
        assert abs(mean_sim) < 0.05, f"Mean similarity {mean_sim} too far from 0"

    def test_orthogonality_std(self, dim):
        """Std of random similarities should be ~ 1/sqrt(dim) ~ 0.01."""
        n_pairs = 50
        sims = []
        for i in range(n_pairs):
            a = random_ternary(dim, seed=i * 2)
            b = random_ternary(dim, seed=i * 2 + 1)
            sims.append(similarity(a, b))
        sims_tensor = torch.tensor(sims)
        std = sims_tensor.std().item()
        expected_std = 1 / (dim ** 0.5)  # ~0.01 for dim=10000
        # Allow 5x tolerance since ternary sparsity affects this
        assert std < expected_std * 5, f"Std {std} too high (expected ~{expected_std})"
        print(f"\n  Orthogonality std: {std:.4f}, expected ~{expected_std:.4f}")


class TestNormalize:
    """Test suite for normalize function."""

    def test_returns_tensor(self, dim):
        """Should return a torch tensor."""
        v = random_ternary(dim, seed=1)
        result = normalize(v)
        assert isinstance(result, torch.Tensor)

    def test_correct_dimension(self, dim):
        """Should return tensor of same dimension."""
        v = random_ternary(dim, seed=1)
        result = normalize(v)
        assert result.shape == (dim,)

    def test_unit_length(self, dim):
        """Normalized vector should have unit length."""
        v = random_ternary(dim, seed=1)
        result = normalize(v)
        norm = torch.norm(result)
        assert abs(norm.item() - 1.0) < 1e-5

    def test_same_direction(self, dim):
        """Normalized vector should point in same direction."""
        v = random_ternary(dim, seed=1)
        result = normalize(v)
        # Check that v and result are parallel (same direction)
        sim = similarity(v, result)
        assert sim > 0.999

    def test_zero_vector_stays_zero(self, dim):
        """Normalizing zero vector should return zero vector."""
        zero = torch.zeros(dim)
        result = normalize(zero)
        assert torch.equal(result, zero)

    def test_already_normalized_unchanged(self, dim):
        """Normalizing a unit vector should be idempotent."""
        v = random_ternary(dim, seed=1)
        normalized = normalize(v)
        double_normalized = normalize(normalized)
        assert torch.allclose(normalized, double_normalized, atol=1e-6)
