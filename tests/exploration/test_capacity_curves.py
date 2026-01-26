"""Exploration tests documenting bundle capacity and degradation.

These tests serve as documentation for system behavior and capacity limits.
They are not required for regression testing.
"""

import torch
import pytest
from engram.hdv import random_ternary, bundle, similarity


class TestBundleCapacity:
    """Document bundle capacity and degradation."""

    @pytest.mark.parametrize("n_items", [5, 10, 25, 50, 100, 200])
    def test_bundle_capacity_curve(self, dim, n_items):
        """Document how similarity degrades as bundle size increases.

        Expected: similarity ~ 1/sqrt(n)
        """
        items = [random_ternary(dim, sparsity=0.5, seed=i) for i in range(n_items)]
        bundled = bundle(items)

        sims = [similarity(bundled, item) for item in items]
        avg_sim = sum(sims) / len(sims)
        min_sim = min(sims)
        expected = 1 / (n_items ** 0.5)

        print(f"\n  n={n_items:3d}: avg_sim={avg_sim:.3f}, min_sim={min_sim:.3f}, expected~{expected:.3f}")

        # Should be in reasonable range of theoretical value
        tolerance = max(0.5, 0.3 + n_items / 500)
        assert avg_sim > expected * tolerance

    def test_bundle_vs_noise_floor(self, dim):
        """Compare bundle queryability against noise floor."""
        n_items = 50
        items = [random_ternary(dim, sparsity=0.5, seed=i) for i in range(n_items)]
        bundled = bundle(items)

        # Member similarities
        member_sims = [similarity(bundled, item) for item in items]
        avg_member = sum(member_sims) / len(member_sims)
        min_member = min(member_sims)

        # Non-member similarities (noise floor)
        non_member_sims = [
            similarity(bundled, random_ternary(dim, sparsity=0.5, seed=1000 + i))
            for i in range(50)
        ]
        max_noise = max(abs(s) for s in non_member_sims)

        print(f"\n  === Bundle Capacity vs Noise (n={n_items}) ===")
        print(f"  Member avg:     {avg_member:.3f}")
        print(f"  Member min:     {min_member:.3f}")
        print(f"  Noise max|sim|: {max_noise:.3f}")
        print(f"  Signal/Noise:   {min_member/max_noise:.1f}x")

        assert min_member > max_noise * 1.5
