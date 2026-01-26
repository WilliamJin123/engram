"""Exploration tests documenting noise characteristics of HDV operations.

These tests serve as documentation for expected behavior including
capacity limits and noise floors.
"""

import torch
import pytest
from engram.hdv import random_ternary, bind, unbind, similarity


class TestOrthogonalityNoise:
    """Document the noise floor from random vector orthogonality."""

    def test_orthogonality_distribution(self, dim):
        """Random vectors have similarity ~ N(0, 1/sqrt(dim)).

        For dim=10000, this means std ~ 0.01, so |similarity| < 0.03 with 99% confidence.
        """
        n_samples = 100
        sims = []
        for i in range(n_samples):
            a = random_ternary(dim, sparsity=0.5, seed=i * 2)
            b = random_ternary(dim, sparsity=0.5, seed=i * 2 + 1)
            sims.append(similarity(a, b))

        sims_tensor = torch.tensor(sims)
        mean = sims_tensor.mean().item()
        std = sims_tensor.std().item()
        max_abs = sims_tensor.abs().max().item()

        print(f"\n  === Orthogonality Noise (dim={dim}) ===")
        print(f"  Mean:    {mean:.4f} (expected: ~0)")
        print(f"  Std:     {std:.4f} (expected: ~{1/dim**0.5:.4f})")
        print(f"  Max|sim|: {max_abs:.4f}")

        assert abs(mean) < 0.02
        assert std < 0.05
        assert max_abs < 0.1


class TestBindNoiseFloor:
    """Document noise characteristics of bind/unbind operations."""

    def test_unbind_noise_with_wrong_key(self, dim):
        """Document noise when unbinding with wrong key."""
        n_samples = 50
        sims = []
        for i in range(n_samples):
            a = random_ternary(dim, sparsity=0.5, seed=i * 3)
            b = random_ternary(dim, sparsity=0.5, seed=i * 3 + 1)
            wrong_key = random_ternary(dim, sparsity=0.5, seed=i * 3 + 2)
            bound = bind(a, b)
            recovered = unbind(bound, wrong_key)
            sims.append(similarity(recovered, b))

        sims_tensor = torch.tensor(sims)
        max_abs = sims_tensor.abs().max().item()

        print(f"\n  === Wrong-Key Unbind Noise ===")
        print(f"  Max|sim|: {max_abs:.4f}")

        assert max_abs < 0.15

    def test_correct_unbind_vs_noise(self, dim):
        """Compare correct unbind similarity to wrong-key noise."""
        n_samples = 30
        correct_sims = []
        wrong_sims = []

        for i in range(n_samples):
            a = random_ternary(dim, sparsity=0.5, seed=i * 3)
            b = random_ternary(dim, sparsity=0.5, seed=i * 3 + 1)
            wrong_key = random_ternary(dim, sparsity=0.5, seed=i * 3 + 2)

            bound = bind(a, b)
            correct_recovered = unbind(bound, a)
            wrong_recovered = unbind(bound, wrong_key)

            correct_sims.append(similarity(correct_recovered, b))
            wrong_sims.append(similarity(wrong_recovered, b))

        min_correct = min(correct_sims)
        max_wrong = max(abs(s) for s in wrong_sims)

        print(f"\n  === Correct vs Wrong Unbind ===")
        print(f"  Correct min: {min_correct:.3f}")
        print(f"  Wrong max:   {max_wrong:.3f}")
        print(f"  Signal/Noise: {min_correct/max_wrong:.1f}x")

        assert min_correct > max_wrong * 2


class TestSparsityEffects:
    """Document how sparsity affects HDV properties."""

    @pytest.mark.parametrize("sparsity", [0.25, 0.5, 0.75, 1.0])
    def test_unbind_recovery_by_sparsity(self, dim, sparsity):
        """Document how sparsity affects unbind recovery.

        Higher sparsity = better recovery because less information loss.
        """
        n_samples = 30
        sims = []

        for i in range(n_samples):
            a = random_ternary(dim, sparsity=sparsity, seed=i * 2)
            b = random_ternary(dim, sparsity=sparsity, seed=i * 2 + 1)
            bound = bind(a, b)
            recovered = unbind(bound, a)
            sims.append(similarity(recovered, b))

        avg_sim = sum(sims) / len(sims)
        min_sim = min(sims)

        print(f"\n  sparsity={sparsity}: avg_recovery={avg_sim:.3f}, min={min_sim:.3f}")

        if sparsity >= 0.75:
            assert avg_sim > 0.7
        elif sparsity >= 0.5:
            assert avg_sim > 0.4
