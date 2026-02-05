# tests/stress/metrics/retrieval_metrics.py
"""Retrieval metrics for evaluating search quality.

Provides standard Information Retrieval metrics:
- MRR (Mean Reciprocal Rank): Average of 1/rank for each query
- Recall@K: Fraction of targets found in top-K results
- get_target_rank: Extract rank of specific pattern from results

All ranks are 1-indexed (rank 1 = best).
"""

from __future__ import annotations

from typing import Any


def compute_mrr(ranks: list[int | None]) -> float:
    """Compute Mean Reciprocal Rank.

    MRR = (1/Q) * sum(1/rank_i) for each query i

    Args:
        ranks: List of target ranks (1-indexed), None if not found

    Returns:
        MRR score in [0, 1]. Higher is better.
    """
    if not ranks:
        return 0.0

    reciprocals = []
    for rank in ranks:
        if rank is not None:
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)  # Not found contributes 0

    return sum(reciprocals) / len(reciprocals)


def compute_recall_at_k(ranks: list[int | None], k: int = 3) -> float:
    """Compute Recall@K (fraction of targets found in top-K).

    Args:
        ranks: List of target ranks (1-indexed), None if not found
        k: Cutoff for top-K

    Returns:
        Recall@K score in [0, 1]. Higher is better.
    """
    if not ranks:
        return 0.0

    found_in_top_k = sum(1 for r in ranks if r is not None and r <= k)
    return found_in_top_k / len(ranks)


def get_target_rank(results: list[Any], target_id: str) -> int | None:
    """Extract rank of target pattern from retrieval results.

    Args:
        results: List of RetrievalResult objects with pattern_id attribute
        target_id: Pattern ID to find

    Returns:
        1-indexed rank if found, None if not in results
    """
    for i, result in enumerate(results):
        if result.pattern_id == target_id:
            return i + 1  # 1-indexed

    return None
