# tests/stress/test_degradation_curves.py
"""Degradation curve tests for METR-03 requirement.

METR-03: Degradation curves with X-axis = near-miss count (not general noise level).

This module provides:
- 4-level summary degradation (NONE/LOW/MEDIUM/HIGH noise presets)
- Fine-grained degradation curves (10+ near-miss count data points)
- Multi-method comparison (interference vs cosine vs random vs recency)

All tests output METRICS JSON for Phase 8 analysis and can generate
plots and reports with --update-report flag.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pytest

from tests.conftest import NoiseLevel, NoiseConfig
from tests.stress.conftest import StressMetrics, update_report
from tests.stress.metrics import (
    compute_mrr,
    compute_recall_at_k,
    get_target_rank,
    plot_degradation_curve,
    plot_method_comparison,
)
from tests.hypothesis_validation.baselines import (
    cosine_retrieval,
    random_retrieval,
    recency_retrieval,
)


# Directory for reports output
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


def collect_4level_degradation(noisy_memory, n_trials: int = 10) -> dict:
    """Collect performance metrics at each NoiseLevel preset.

    Per CONTEXT.md: Both 4-level summary AND fine-grained curves.

    Args:
        noisy_memory: Factory fixture for creating noisy memory scenarios.
        n_trials: Number of trials per noise level.

    Returns:
        Dict with noise_level keys ('none', 'low', 'medium', 'high'),
        each containing:
        - mean_rank, std_rank, mrr, recall_at_3, recall_at_5, raw_ranks
    """
    levels = [NoiseLevel.NONE, NoiseLevel.LOW, NoiseLevel.MEDIUM, NoiseLevel.HIGH]
    data = {}

    for level in levels:
        ranks = []
        for trial in range(n_trials):
            store, target_ids, _ = noisy_memory(
                ["The quick brown fox jumps"],
                level=level,
                seed=trial * 100,
            )
            target_id = target_ids[0]
            target_pattern = store.get(target_id)

            results = store.retrieve(target_pattern.text, top_k=10)
            rank = get_target_rank(results, target_id)
            ranks.append(rank if rank else 11)

        data[level.value] = {
            "mean_rank": float(np.mean(ranks)),
            "std_rank": float(np.std(ranks)),
            "mrr": compute_mrr(ranks),
            "recall_at_3": compute_recall_at_k(ranks, k=3),
            "recall_at_5": compute_recall_at_k(ranks, k=5),
            "raw_ranks": ranks,
        }

    return data


def collect_fine_grained_degradation(
    noisy_memory,
    near_miss_counts: list[int] | None = None,
    n_trials: int = 10,
) -> dict:
    """Collect performance at varying near-miss counts.

    Per CONTEXT.md: Fine-grained curves with 10+ data points.
    Per RESEARCH.md: Linear spacing [0, 1, 2, 3, 4, 5, 7, 10, 15, 20].

    Args:
        noisy_memory: Factory fixture for creating noisy memory scenarios.
        near_miss_counts: List of near-miss counts to test. Defaults to 10 points.
        n_trials: Number of trials per near-miss count.

    Returns:
        Dict with keys: noise_level, mean_rank, std_rank, recall_at_3
        (lists aligned by index)
    """
    if near_miss_counts is None:
        near_miss_counts = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]

    data = {"noise_level": [], "mean_rank": [], "std_rank": [], "recall_at_3": []}

    for nm_count in near_miss_counts:
        ranks = []
        for trial in range(n_trials):
            # Custom config with specific near-miss count
            config = NoiseConfig(
                near_miss_ratio=float(nm_count),
                clutter_ratio=3.0,  # Fixed clutter
                seed=trial * 100,
            )
            store, target_ids, _ = noisy_memory(
                ["The quick brown fox jumps"],
                config=config,
            )
            target_id = target_ids[0]
            target_pattern = store.get(target_id)

            results = store.retrieve(target_pattern.text, top_k=20)
            rank = get_target_rank(results, target_id)
            ranks.append(rank if rank else 21)

        data["noise_level"].append(nm_count)
        data["mean_rank"].append(float(np.mean(ranks)))
        data["std_rank"].append(float(np.std(ranks)))
        data["recall_at_3"].append(compute_recall_at_k(ranks, k=3))

    return data


def collect_method_comparison_degradation(
    noisy_memory,
    near_miss_counts: list[int] | None = None,
    n_trials: int = 10,
) -> tuple[dict[str, dict], list[int]]:
    """Collect degradation data for all retrieval methods.

    Compares interference retrieval against classical baselines:
    - cosine: Sparse binary vector cosine similarity
    - random: Random selection (floor baseline)
    - recency: Ranks by last access time

    Args:
        noisy_memory: Factory fixture for creating noisy memory scenarios.
        near_miss_counts: List of near-miss counts to test.
        n_trials: Number of trials per condition.

    Returns:
        Tuple of (methods_data, near_miss_counts) where methods_data maps
        method_name to {mean_rank: [], std_rank: []}.
    """
    if near_miss_counts is None:
        near_miss_counts = [0, 1, 2, 3, 5, 7, 10]

    methods_data = {
        "interference": {"mean_rank": [], "std_rank": []},
        "cosine": {"mean_rank": [], "std_rank": []},
        "random": {"mean_rank": [], "std_rank": []},
        "recency": {"mean_rank": [], "std_rank": []},
    }

    for nm_count in near_miss_counts:
        method_ranks: dict[str, list[int]] = {m: [] for m in methods_data.keys()}

        for trial in range(n_trials):
            config = NoiseConfig(
                near_miss_ratio=float(nm_count),
                clutter_ratio=3.0,
                seed=trial * 100,
            )
            store, target_ids, _ = noisy_memory(
                ["The quick brown fox"],
                config=config,
            )
            target_id = target_ids[0]
            target_pattern = store.get(target_id)
            top_k = 20

            # Interference retrieval
            int_results = store.retrieve(target_pattern.text, top_k=top_k)
            int_rank = get_target_rank(int_results, target_id)
            method_ranks["interference"].append(int_rank if int_rank else top_k + 1)

            # Cosine retrieval
            cos_results = cosine_retrieval(target_pattern.bits, store.patterns, top_k=top_k)
            cos_rank = _get_tuple_rank(cos_results, target_id, top_k)
            method_ranks["cosine"].append(cos_rank)

            # Random retrieval
            rng = random.Random(trial * 100 + 1)
            rand_results = random_retrieval(store.patterns, top_k=top_k, rng=rng)
            rand_rank = _get_tuple_rank(rand_results, target_id, top_k)
            method_ranks["random"].append(rand_rank)

            # Recency retrieval
            rec_results = recency_retrieval(store.patterns, store.current_tick, top_k=top_k)
            rec_rank = _get_tuple_rank(rec_results, target_id, top_k)
            method_ranks["recency"].append(rec_rank)

        for method_name, ranks in method_ranks.items():
            methods_data[method_name]["mean_rank"].append(float(np.mean(ranks)))
            methods_data[method_name]["std_rank"].append(float(np.std(ranks)))

    return methods_data, near_miss_counts


def _get_tuple_rank(
    results: list[tuple[str, float]], target_id: str, top_k: int
) -> int:
    """Get rank of target from tuple-format results (baseline methods).

    Args:
        results: List of (pattern_id, score) tuples
        target_id: Pattern ID to find
        top_k: Default rank if not found

    Returns:
        1-indexed rank if found, top_k + 1 if not found
    """
    for i, (pid, _) in enumerate(results):
        if pid == target_id:
            return i + 1
    return top_k + 1
