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


# =============================================================================
# METR-01: Interference vs Cosine Baseline
# =============================================================================


@pytest.mark.stress
@pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
def test_interference_vs_cosine_baseline(noisy_memory, noise_level, update_report):
    """METR-01: Differentiation test - interference vs cosine similarity.

    Per CONTEXT.md: Document results without failing tests.
    Per RESEARCH.md: Use Mann-Whitney U test for significance.

    Cosine baseline uses sparse binary vector similarity.
    Interference retrieval uses quantum-inspired amplitude interference.
    """
    # Reduce trials if not generating report (faster CI)
    n_trials = 30 if update_report else 10

    results = run_comparison_trials(noisy_memory, noise_level, n_trials=n_trials)

    # Compare interference vs cosine using Mann-Whitney U
    comparison = compare_methods(
        results["interference"],
        results["cosine"],
        "interference",
        "cosine",
    )

    # Compute MRR for each method
    int_mrr = compute_mrr(results["interference"])
    cos_mrr = compute_mrr(results["cosine"])

    # Convert numpy bool to Python bool for JSON serialization
    is_significant = bool(comparison["significant_at_0.05"])

    # Document results (soft metrics - no assertions that fail)
    metrics = StressMetrics(
        test_name="interference_vs_cosine",
        noise_level=noise_level.value,
        success=is_significant,
        metrics={
            "interference_mrr": int_mrr,
            "cosine_mrr": cos_mrr,
            "interference_mean_rank": sum(results["interference"]) / len(results["interference"]),
            "cosine_mean_rank": sum(results["cosine"]) / len(results["cosine"]),
            "p_value": comparison["p_value"],
            "statistically_significant": is_significant,
            "interpretation": comparison["interpretation"],
            "n_trials": n_trials,
        },
    )

    print(f"\nMETRICS: {metrics.to_json()}")


# =============================================================================
# METR-02: Interference vs Random Baseline
# =============================================================================


@pytest.mark.stress
@pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
def test_interference_vs_random_baseline(noisy_memory, noise_level, update_report):
    """METR-02: Differentiation test - interference vs random retrieval.

    Random is floor baseline - interference SHOULD significantly outperform.
    Any meaningful retrieval method should beat random selection.
    """
    n_trials = 30 if update_report else 10

    results = run_comparison_trials(noisy_memory, noise_level, n_trials=n_trials)

    # Compare interference vs random
    comparison = compare_methods(
        results["interference"],
        results["random"],
        "interference",
        "random",
    )

    int_mrr = compute_mrr(results["interference"])
    rand_mrr = compute_mrr(results["random"])

    # Convert numpy bool to Python bool for JSON serialization
    is_significant = bool(comparison["significant_at_0.05"])

    metrics = StressMetrics(
        test_name="interference_vs_random",
        noise_level=noise_level.value,
        success=is_significant,
        metrics={
            "interference_mrr": int_mrr,
            "random_mrr": rand_mrr,
            "interference_mean_rank": sum(results["interference"]) / len(results["interference"]),
            "random_mean_rank": sum(results["random"]) / len(results["random"]),
            "p_value": comparison["p_value"],
            "statistically_significant": is_significant,
            "interpretation": comparison["interpretation"],
            "n_trials": n_trials,
        },
    )

    print(f"\nMETRICS: {metrics.to_json()}")


# =============================================================================
# Bonus: Interference vs Recency Baseline
# =============================================================================


@pytest.mark.stress
@pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
def test_interference_vs_recency_baseline(noisy_memory, noise_level, update_report):
    """Bonus: Differentiation test - interference vs recency baseline.

    Recency baseline ranks by access time (most recent first).
    Per CONTEXT.md: Include recency as third baseline comparison.
    """
    n_trials = 30 if update_report else 10

    results = run_comparison_trials(noisy_memory, noise_level, n_trials=n_trials)

    # Compare interference vs recency
    comparison = compare_methods(
        results["interference"],
        results["recency"],
        "interference",
        "recency",
    )

    # Convert numpy bool to Python bool for JSON serialization
    is_significant = bool(comparison["significant_at_0.05"])

    metrics = StressMetrics(
        test_name="interference_vs_recency",
        noise_level=noise_level.value,
        success=is_significant,
        metrics={
            "interference_mrr": compute_mrr(results["interference"]),
            "recency_mrr": compute_mrr(results["recency"]),
            "p_value": comparison["p_value"],
            "statistically_significant": is_significant,
            "n_trials": n_trials,
        },
    )

    print(f"\nMETRICS: {metrics.to_json()}")


# =============================================================================
# Comprehensive Comparison: All Methods Summary
# =============================================================================


@pytest.mark.stress
@pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
def test_all_methods_comparison(noisy_memory, noise_level, update_report):
    """Comprehensive comparison of interference vs all baselines.

    Produces summary metrics for report generation with:
    - MRR, Recall@3, Recall@5 for each method
    - Statistical significance vs each baseline
    - Overall success metrics
    """
    n_trials = 30 if update_report else 10

    results = run_comparison_trials(noisy_memory, noise_level, n_trials=n_trials)

    # Compute metrics for all methods
    summary = {
        "noise_level": noise_level.value,
        "n_trials": n_trials,
        "methods": {},
    }

    for method_name, ranks in results.items():
        valid_ranks = [r for r in ranks if r is not None]
        summary["methods"][method_name] = {
            "mean_rank": sum(valid_ranks) / len(valid_ranks) if valid_ranks else None,
            "mrr": compute_mrr(ranks),
            "recall_at_3": compute_recall_at_k(ranks, k=3),
            "recall_at_5": compute_recall_at_k(ranks, k=5),
        }

    # Statistical comparisons: interference vs each baseline
    comparisons = {}
    for baseline in ["cosine", "random", "recency"]:
        comp = compare_methods(
            results["interference"],
            results[baseline],
            "interference",
            baseline,
        )
        comparisons[f"interference_vs_{baseline}"] = {
            "p_value": comp["p_value"],
            "significant": bool(comp["significant_at_0.05"]),
        }

    summary["comparisons"] = comparisons

    # Determine overall success: interference significantly better than at least one baseline
    any_significant = any(c["significant"] for c in comparisons.values())
    all_significant = all(c["significant"] for c in comparisons.values())

    summary["overall"] = {
        "any_baseline_beaten": any_significant,
        "all_baselines_beaten": all_significant,
    }

    metrics = StressMetrics(
        test_name="all_methods_comparison",
        noise_level=noise_level.value,
        success=any_significant,
        metrics=summary,
    )

    print(f"\nMETRICS: {metrics.to_json()}")


# =============================================================================
# Clean Memory Baseline: Reference Performance
# =============================================================================


@pytest.mark.stress
def test_clean_memory_baseline_comparison(noisy_memory, update_report):
    """Baseline comparison on CLEAN memory (NoiseLevel.NONE).

    Per CONTEXT.md: Test baselines on BOTH clean and noisy memory.
    This establishes ceiling performance for each method.
    """
    n_trials = 30 if update_report else 10

    results = run_comparison_trials(
        noisy_memory,
        NoiseLevel.NONE,
        n_trials=n_trials,
    )

    summary = {
        "noise_level": "none",
        "n_trials": n_trials,
        "methods": {},
    }

    for method_name, ranks in results.items():
        valid_ranks = [r for r in ranks if r is not None]
        summary["methods"][method_name] = {
            "mean_rank": sum(valid_ranks) / len(valid_ranks) if valid_ranks else None,
            "mrr": compute_mrr(ranks),
            "recall_at_3": compute_recall_at_k(ranks, k=3),
        }

    metrics = StressMetrics(
        test_name="clean_memory_baseline",
        noise_level="none",
        success=True,  # Clean memory is reference, not pass/fail
        metrics=summary,
    )

    print(f"\nMETRICS: {metrics.to_json()}")
