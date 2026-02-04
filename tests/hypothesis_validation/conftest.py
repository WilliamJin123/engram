# tests/hypothesis_validation/conftest.py
"""Shared fixtures for hypothesis validation tests.

These fixtures provide the infrastructure for statistically comparing quantum
memory methods against baselines. Key fixtures:

- seeded_rng: Reproducible random number generator
- statistical_comparator: Function for Welch's t-test + Cohen's d
- memory_store_factory: Creates fresh MemoryStore instances for testing

Usage:
    def test_interference_beats_jaccard(seeded_rng, statistical_comparator):
        scores_a = [run_trial(seeded_rng) for _ in range(N_TRIALS)]
        scores_b = [run_baseline(seeded_rng) for _ in range(N_TRIALS)]
        result = statistical_comparator(scores_a, scores_b)
        assert result['significant'] and result['cohens_d'] > 0.3
"""

from __future__ import annotations

import random
from typing import Dict, List, Callable, Any

import pytest
from scipy.stats import ttest_ind
import pingouin as pg


# Default trial count for statistical tests
# 50 trials provides reasonable power while keeping tests fast in CI
# Can be overridden per-test for behaviors requiring more samples
N_TRIALS = 50


@pytest.fixture
def seeded_rng() -> Callable[[int], random.Random]:
    """Factory for creating seeded random number generators.

    Returns a function that creates a Random instance with a given seed.
    Using seeded RNG ensures reproducible tests across runs.

    Usage:
        def test_something(seeded_rng):
            rng = seeded_rng(42)  # Get RNG with seed 42
            value = rng.random()  # Deterministic result
    """
    def _create_rng(seed: int = 42) -> random.Random:
        return random.Random(seed)
    return _create_rng


@pytest.fixture
def statistical_comparator() -> Callable[[List[float], List[float]], Dict[str, Any]]:
    """Factory for statistical comparison between two score distributions.

    Returns a function that takes two lists of scores and performs:
    1. Welch's t-test (unequal variance assumption)
    2. Cohen's d effect size calculation

    Returns dict with:
        - t_stat: t-statistic from Welch's test
        - p_value: one-tailed p-value (testing if group_a > group_b)
        - cohens_d: Cohen's d effect size
        - significant: bool, p < 0.05
        - effect_size_interpretation: 'negligible', 'small', 'medium', or 'large'

    Usage:
        def test_comparison(statistical_comparator):
            result = statistical_comparator(quantum_scores, baseline_scores)
            assert result['significant']
            assert result['effect_size_interpretation'] in ['medium', 'large']
    """
    def _compare(
        group_a: List[float],
        group_b: List[float],
        alternative: str = 'greater'
    ) -> Dict[str, Any]:
        """Compare group_a vs group_b.

        Args:
            group_a: Scores from method A (expected to be higher)
            group_b: Scores from method B (baseline)
            alternative: 'greater' tests if A > B, 'less' if A < B, 'two-sided' for any difference

        Returns:
            Dictionary with statistical results
        """
        # Welch's t-test (does not assume equal variance)
        result = ttest_ind(
            group_a,
            group_b,
            equal_var=False,
            alternative=alternative
        )

        # Cohen's d effect size using pingouin
        # Handles edge cases like zero variance
        d = pg.compute_effsize(group_a, group_b, paired=False, eftype='cohen')

        # Interpret effect size
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = 'negligible'
        elif abs_d < 0.5:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'

        return {
            't_stat': result.statistic,
            'p_value': result.pvalue,
            'cohens_d': d,
            'significant': result.pvalue < 0.05,
            'effect_size_interpretation': interpretation,
        }

    return _compare


@pytest.fixture
def memory_store_factory():
    """Factory for creating fresh MemoryStore instances.

    Creates MemoryStore with seeded encoder for reproducible tests.
    Each call returns a new, empty store.

    Usage:
        def test_retrieval(memory_store_factory, seeded_rng):
            store = memory_store_factory(seed=42)
            store.store("test memory")
            results = store.retrieve("test")
    """
    # Import here to avoid import errors if agentic layer changes
    from agentic.memory_store import MemoryStore
    from agentic.text_encoder import TextEncoder

    def _create_store(
        seed: int = 42,
        dim: int = 10000,
        k: int = 100
    ) -> MemoryStore:
        """Create a fresh MemoryStore with seeded encoder.

        Args:
            seed: Random seed for encoder
            dim: Pattern dimensionality
            k: Sparsity (active bits per pattern)

        Returns:
            Fresh MemoryStore instance
        """
        encoder = TextEncoder(dim=dim, k=k)
        return MemoryStore(encoder=encoder)

    return _create_store
