"""Test phase interference at scale.

The claim: With many patterns having random phases, retrieval
should still find the correct match.

Success criteria:
- Correct match in top-3 at 1000 patterns
- Document degradation curve
"""

import torch
import pytest
import math

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.interference import interference_retrieval, jaccard_retrieval


class TestPhaseInterferenceAtScale:
    """Test retrieval accuracy as pattern count increases."""

    @pytest.mark.parametrize("n_patterns", [100, 500, 1000, 2000, 5000])
    def test_retrieval_accuracy_vs_scale(self, dim, rng, n_patterns):
        """Track if correct match stays in top-K as scale increases."""
        k = 50  # sparsity

        # Create query
        query = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        # Create memory with one exact match and rest random
        patterns = []

        # Pattern 0: exact match
        exact_match = ComplexSparsePattern(
            dim=dim,
            indices=query.indices,
            magnitudes=query.magnitudes,
            phases=query.phases,
        )
        patterns.append(exact_match)

        # Rest are random (different indices, random phases)
        for _ in range(n_patterns - 1):
            patterns.append(ComplexSparsePattern.random(dim=dim, k=k, generator=rng))

        # Test interference retrieval
        results = interference_retrieval(query, patterns)

        # Check if exact match is in top-3
        top_3_indices = [r[0] for r in results[:3]]
        in_top_3 = 0 in top_3_indices

        # Check rank of exact match
        exact_rank = next(i for i, r in enumerate(results) if r[0] == 0) + 1

        print(f"\n{n_patterns} patterns:")
        print(f"  Exact match rank: {exact_rank}")
        print(f"  In top-3: {in_top_3}")
        print(f"  Top-5 scores: {[(r[0], f'{r[1]:.3f}') for r in results[:5]]}")

        # Should find exact match at rank 1
        assert exact_rank == 1, f"Exact match should be rank 1, got {exact_rank}"

    def test_random_phase_patterns(self, dim, rng):
        """Test with patterns that share indices but have random phases."""
        k = 50
        n_patterns = 100

        # Create base pattern
        query = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        patterns = []

        # Pattern 0: same indices, same phases (should win)
        patterns.append(ComplexSparsePattern(
            dim=dim,
            indices=query.indices,
            magnitudes=query.magnitudes,
            phases=query.phases,
        ))

        # Patterns 1-10: same indices, random phases (phase interference)
        for _ in range(10):
            random_phases = tuple(
                (torch.rand(1, generator=rng).item() * 2 * math.pi)
                for _ in range(k)
            )
            patterns.append(ComplexSparsePattern(
                dim=dim,
                indices=query.indices,
                magnitudes=query.magnitudes,
                phases=random_phases,
            ))

        # Rest: random patterns
        for _ in range(n_patterns - 11):
            patterns.append(ComplexSparsePattern.random(dim=dim, k=k, generator=rng))

        # Interference should distinguish same-phase from random-phase
        int_results = interference_retrieval(query, patterns)

        # Jaccard can't distinguish (all have same indices for 0-10)
        jac_results = jaccard_retrieval(query, patterns)

        print("\nSame indices, different phases test:")
        print(f"  Interference top-5: {[(r[0], f'{r[1]:.3f}') for r in int_results[:5]]}")
        print(f"  Jaccard top-5: {[(r[0], f'{r[1]:.3f}') for r in jac_results[:5]]}")

        # Pattern 0 (same phase) should be top for interference
        assert int_results[0][0] == 0, "Same-phase pattern should rank first"

        # Interference score for pattern 0 should be much higher than patterns 1-10
        same_phase_score = int_results[0][1]
        random_phase_scores = [r[1] for r in int_results if r[0] in range(1, 11)]
        avg_random_phase = sum(random_phase_scores) / len(random_phase_scores) if random_phase_scores else 0

        print(f"  Same-phase score: {same_phase_score:.3f}")
        print(f"  Avg random-phase score: {avg_random_phase:.3f}")

        assert same_phase_score > avg_random_phase + 0.1, "Same-phase should score higher"
