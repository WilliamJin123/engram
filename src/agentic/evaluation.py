# src/agentic/evaluation.py
"""Evaluation metrics for measuring emergent semantic clustering.

This module provides metrics to evaluate:
1. Semantic separation - whether patterns cluster by semantic group
2. Pattern obesity - tracking pattern growth over time
3. Retrieval precision - whether coactivation improves retrieval accuracy

These metrics help validate that the agentic memory system is learning
meaningful associations without explicit semantic supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.evolving_pattern import EvolvingPattern
    from agentic.memory_store import MemoryStore


@dataclass
class SemanticSeparationResult:
    """Result of semantic separation analysis.

    Attributes:
        within_group_similarity: Average Jaccard similarity within groups.
        cross_group_similarity: Average Jaccard similarity across groups.
        separation_ratio: Ratio of within to cross (higher = better clustering).
        group_similarities: Per-group average similarities.
    """

    within_group_similarity: float
    cross_group_similarity: float
    separation_ratio: float
    group_similarities: dict[str, float]


@dataclass
class ObesityStats:
    """Statistics about pattern obesity (bit count growth).

    Attributes:
        mean_obesity: Average obesity across all patterns.
        max_obesity: Maximum obesity observed.
        min_obesity: Minimum obesity observed.
        mean_acquired_ratio: Average ratio of acquired to total bits.
        total_patterns: Number of patterns analyzed.
    """

    mean_obesity: float
    max_obesity: float
    min_obesity: float
    mean_acquired_ratio: float
    total_patterns: int


@dataclass
class RetrievalPrecisionResult:
    """Result of retrieval precision analysis.

    Attributes:
        precision_at_k: Fraction of top-k results in expected group.
        mean_reciprocal_rank: MRR for first relevant result.
        total_queries: Number of queries evaluated.
        per_query_results: Detailed per-query metrics.
    """

    precision_at_k: float
    mean_reciprocal_rank: float
    total_queries: int
    per_query_results: list[dict]


def semantic_separation(
    patterns_by_group: dict[str, Sequence["EvolvingPattern"]],
) -> SemanticSeparationResult:
    """Measure semantic separation between pattern groups.

    Computes the ratio of within-group similarity to cross-group similarity.
    Higher ratios indicate better clustering (patterns are more similar to
    members of their own group than to members of other groups).

    Args:
        patterns_by_group: Dictionary mapping group names to pattern lists.

    Returns:
        SemanticSeparationResult with similarity metrics.
    """
    # Filter out empty groups
    non_empty_groups = {k: v for k, v in patterns_by_group.items() if len(v) > 0}

    if not non_empty_groups:
        return SemanticSeparationResult(
            within_group_similarity=0,
            cross_group_similarity=0,
            separation_ratio=0,
            group_similarities={},
        )

    # Compute within-group similarities
    group_similarities: dict[str, float] = {}
    all_within_sims: list[float] = []

    for group_name, patterns in non_empty_groups.items():
        if len(patterns) < 2:
            group_similarities[group_name] = 0
            continue

        sims = []
        for i, p1 in enumerate(patterns):
            for p2 in patterns[i + 1:]:
                sim = _jaccard_similarity(p1.bits, p2.bits)
                sims.append(sim)

        avg_sim = sum(sims) / len(sims) if sims else 0
        group_similarities[group_name] = avg_sim
        all_within_sims.extend(sims)

    within_group_similarity = (
        sum(all_within_sims) / len(all_within_sims)
        if all_within_sims
        else 0
    )

    # Compute cross-group similarities
    all_cross_sims: list[float] = []
    group_names = list(non_empty_groups.keys())

    for i, group1_name in enumerate(group_names):
        for group2_name in group_names[i + 1:]:
            group1 = non_empty_groups[group1_name]
            group2 = non_empty_groups[group2_name]

            for p1 in group1:
                for p2 in group2:
                    sim = _jaccard_similarity(p1.bits, p2.bits)
                    all_cross_sims.append(sim)

    cross_group_similarity = (
        sum(all_cross_sims) / len(all_cross_sims)
        if all_cross_sims
        else 0
    )

    # Compute separation ratio (avoid division by zero)
    if cross_group_similarity > 0:
        separation_ratio = within_group_similarity / cross_group_similarity
    elif within_group_similarity > 0:
        separation_ratio = float("inf")
    else:
        separation_ratio = 1.0

    return SemanticSeparationResult(
        within_group_similarity=within_group_similarity,
        cross_group_similarity=cross_group_similarity,
        separation_ratio=separation_ratio,
        group_similarities=group_similarities,
    )


def pattern_obesity_stats(patterns: Sequence["EvolvingPattern"]) -> ObesityStats:
    """Compute statistics about pattern obesity (bit count growth).

    Obesity is the ratio of current bits to original bits.
    Patterns start with obesity 1.0 and grow through coactivation.

    Args:
        patterns: List of patterns to analyze.

    Returns:
        ObesityStats with obesity metrics.
    """
    if not patterns:
        return ObesityStats(
            mean_obesity=0,
            max_obesity=0,
            min_obesity=0,
            mean_acquired_ratio=0,
            total_patterns=0,
        )

    obesities = [p.obesity for p in patterns]
    acquired_ratios = [p.acquired_ratio for p in patterns]

    return ObesityStats(
        mean_obesity=sum(obesities) / len(obesities),
        max_obesity=max(obesities),
        min_obesity=min(obesities),
        mean_acquired_ratio=sum(acquired_ratios) / len(acquired_ratios),
        total_patterns=len(patterns),
    )


def retrieval_precision(
    store: "MemoryStore",
    queries: list[tuple[str, str]],
    top_k: int = 5,
    group_key: str = "group",
) -> RetrievalPrecisionResult:
    """Measure retrieval precision for queries with known expected groups.

    For each query, retrieves top-k results and computes:
    - Precision@k: fraction of results in the expected group
    - MRR: reciprocal rank of first result in expected group

    Args:
        store: MemoryStore to query.
        queries: List of (query_text, expected_group) tuples.
        top_k: Number of results to retrieve per query.
        group_key: Metadata key for group labels.

    Returns:
        RetrievalPrecisionResult with precision metrics.
    """
    if not queries:
        return RetrievalPrecisionResult(
            precision_at_k=0,
            mean_reciprocal_rank=0,
            total_queries=0,
            per_query_results=[],
        )

    precisions = []
    reciprocal_ranks = []
    per_query_results = []

    for query_text, expected_group in queries:
        results = store.retrieve(query_text, top_k=top_k, method="jaccard")

        # Count how many results are in expected group
        in_group = 0
        first_relevant_rank = None

        for rank, result in enumerate(results, start=1):
            result_group = result.metadata.get(group_key)
            if result_group == expected_group:
                in_group += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        # Compute precision@k
        precision = in_group / top_k if top_k > 0 else 0

        # Compute reciprocal rank
        rr = 1 / first_relevant_rank if first_relevant_rank else 0

        precisions.append(precision)
        reciprocal_ranks.append(rr)

        per_query_results.append({
            "query": query_text,
            "expected_group": expected_group,
            "precision": precision,
            "reciprocal_rank": rr,
            "results_in_group": in_group,
        })

    return RetrievalPrecisionResult(
        precision_at_k=sum(precisions) / len(precisions),
        mean_reciprocal_rank=sum(reciprocal_ranks) / len(reciprocal_ranks),
        total_queries=len(queries),
        per_query_results=per_query_results,
    )


def _jaccard_similarity(bits1: set[int], bits2: set[int]) -> float:
    """Compute Jaccard similarity between two bit sets."""
    intersection = len(bits1 & bits2)
    union = len(bits1 | bits2)
    return intersection / union if union > 0 else 0.0
