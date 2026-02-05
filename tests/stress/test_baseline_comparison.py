# tests/stress/test_baseline_comparison.py
"""Baseline comparison tests for Phase 8 metrics validation.

This module validates METR-01 and METR-02 requirements:
- METR-01: Interference retrieval can be measured against cosine baseline
- METR-02: Interference retrieval can be measured against random baseline

Additional comparison against recency baseline per CONTEXT.md.

Per CONTEXT.md: "Soft metrics only - document results without failing tests
(Phase 8 is measurement, not enforcement)"

Statistical significance testing uses Mann-Whitney U (non-parametric, rank-based).
30 trials per comparison for statistical stability per RESEARCH.md.
"""

from __future__ import annotations

import random

import pytest

from tests.conftest import NoiseLevel
from tests.stress.conftest import STRESS_NOISE_LEVELS, StressMetrics
from tests.stress.metrics import (
    compute_mrr,
    compute_recall_at_k,
    compare_methods,
    get_target_rank,
)
from tests.hypothesis_validation.baselines import (
    cosine_retrieval,
    random_retrieval,
    recency_retrieval,
)


def _get_rank_from_tuples(
    results: list[tuple[str, float]],
    target_id: str,
    top_k: int,
) -> int:
    """Get target rank from baseline results (tuple format).

    Baseline functions return list of (pattern_id, score) tuples,
    while interference retrieval returns RetrievalResult objects.
    This helper handles the tuple format.

    Args:
        results: List of (pattern_id, score) tuples, sorted by score descending
        target_id: Pattern ID to find
        top_k: Sentinel value for not-found case

    Returns:
        1-indexed rank if found, top_k + 1 if not found
    """
    for i, (pid, _score) in enumerate(results):
        if pid == target_id:
            return i + 1  # 1-indexed
    return top_k + 1  # Not found sentinel


def run_comparison_trials(
    noisy_memory,
    noise_level: NoiseLevel,
    n_trials: int = 30,
    top_k: int = 10,
) -> dict:
    """Run multi-trial comparison of interference vs baselines.

    Per CONTEXT.md: Test baselines on BOTH clean and noisy memory.
    Per RESEARCH.md: 30 trials for Mann-Whitney stability.

    Each trial creates a fresh noisy memory with a different seed,
    then retrieves the target pattern using each method.

    Args:
        noisy_memory: Fixture function for creating noisy MemoryStore
        noise_level: NoiseLevel preset for noise injection
        n_trials: Number of independent trials (default 30 per RESEARCH.md)
        top_k: Number of results to retrieve (default 10)

    Returns:
        Dict with rank lists for each method:
        - "interference": Ranks from MemoryStore.retrieve
        - "cosine": Ranks from cosine_retrieval baseline
        - "random": Ranks from random_retrieval baseline
        - "recency": Ranks from recency_retrieval baseline
    """
    interference_ranks = []
    cosine_ranks = []
    random_ranks = []
    recency_ranks = []

    for trial in range(n_trials):
        # Create noisy memory with trial-specific seed for reproducibility
        store, target_ids, noise_result = noisy_memory(
            ["The quick brown fox jumps over the lazy dog"],
            level=noise_level,
            seed=trial * 100,
        )
        target_id = target_ids[0]
        target_pattern = store.get(target_id)

        # Interference retrieval (system under test)
        int_results = store.retrieve(target_pattern.text, top_k=top_k)
        int_rank = get_target_rank(int_results, target_id)
        interference_ranks.append(int_rank if int_rank else top_k + 1)

        # Cosine baseline (classical sparse vector similarity)
        cos_results = cosine_retrieval(target_pattern.bits, store.patterns, top_k=top_k)
        cos_rank = _get_rank_from_tuples(cos_results, target_id, top_k)
        cosine_ranks.append(cos_rank)

        # Random baseline (floor comparison - different seed per trial)
        rng = random.Random(trial * 100 + 1)
        rand_results = random_retrieval(store.patterns, top_k=top_k, rng=rng)
        rand_rank = _get_rank_from_tuples(rand_results, target_id, top_k)
        random_ranks.append(rand_rank)

        # Recency baseline (ranks by last access time)
        current_tick = store.current_tick
        rec_results = recency_retrieval(store.patterns, current_tick, top_k=top_k)
        rec_rank = _get_rank_from_tuples(rec_results, target_id, top_k)
        recency_ranks.append(rec_rank)

    return {
        "interference": interference_ranks,
        "cosine": cosine_ranks,
        "random": random_ranks,
        "recency": recency_ranks,
    }
