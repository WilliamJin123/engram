# tests/hypothesis_validation/baselines.py
"""Baseline retrieval methods for hypothesis validation.

These are classical (non-quantum) baselines that quantum retrieval methods
must outperform to validate the hypothesis that interference effects provide
meaningful advantage.

Baselines:
- jaccard_similarity: Set overlap / set union
- cosine_similarity_sparse: Sparse vector cosine similarity
- jaccard_retrieval: Rank patterns by Jaccard to query
- cosine_retrieval: Rank patterns by cosine to query
- random_retrieval: Random baseline (floor comparison)

Return format matches interference_retrieval: List[Tuple[str, float]]
where each tuple is (pattern_id, similarity_score) sorted descending.
"""

from __future__ import annotations

import random
import math
from typing import Set, Dict, List, Tuple, Optional

from quantum_substrate.protocols import SubstratePattern


def jaccard_similarity(bits_a: Set[int], bits_b: Set[int]) -> float:
    """Compute Jaccard similarity between two bit sets.

    Jaccard(A, B) = |A intersection B| / |A union B|

    Args:
        bits_a: First set of active bit indices
        bits_b: Second set of active bit indices

    Returns:
        Jaccard similarity in [0, 1]. Returns 0 if both sets empty.
    """
    if not bits_a and not bits_b:
        return 0.0

    intersection = len(bits_a & bits_b)
    union = len(bits_a | bits_b)

    return intersection / union if union > 0 else 0.0


def cosine_similarity_sparse(
    bits_a: Set[int],
    bits_b: Set[int],
    dim: int = 10000
) -> float:
    """Compute cosine similarity treating bit sets as sparse binary vectors.

    For binary vectors, cosine simplifies to:
    cosine(A, B) = |A intersection B| / sqrt(|A| * |B|)

    This is equivalent to dot(a, b) / (norm(a) * norm(b)) where
    a and b are binary vectors with 1s at the indices in bits_a/bits_b.

    Args:
        bits_a: First set of active bit indices
        bits_b: Second set of active bit indices
        dim: Vector dimensionality (unused but kept for API compatibility)

    Returns:
        Cosine similarity in [0, 1]. Returns 0 if either set empty.
    """
    if not bits_a or not bits_b:
        return 0.0

    intersection = len(bits_a & bits_b)
    norm_product = math.sqrt(len(bits_a) * len(bits_b))

    return intersection / norm_product if norm_product > 0 else 0.0


def jaccard_retrieval(
    query_bits: Set[int],
    patterns: Dict[str, SubstratePattern],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Rank patterns by Jaccard similarity to query.

    Classical set-based retrieval baseline. Only considers which bits are
    active, ignoring phase information.

    Args:
        query_bits: Query pattern's active bit indices
        patterns: Dictionary mapping pattern_id to SubstratePattern
        top_k: Number of top results to return

    Returns:
        List of (pattern_id, similarity) tuples, sorted by similarity descending.
        Format matches interference_retrieval for fair comparison.
    """
    results = []

    for pattern_id, pattern in patterns.items():
        similarity = jaccard_similarity(query_bits, pattern.bits)
        results.append((pattern_id, similarity))

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]


def cosine_retrieval(
    query_bits: Set[int],
    patterns: Dict[str, SubstratePattern],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Rank patterns by cosine similarity to query.

    Sparse binary vector cosine similarity baseline.

    Args:
        query_bits: Query pattern's active bit indices
        patterns: Dictionary mapping pattern_id to SubstratePattern
        top_k: Number of top results to return

    Returns:
        List of (pattern_id, similarity) tuples, sorted by similarity descending.
        Format matches interference_retrieval for fair comparison.
    """
    results = []

    for pattern_id, pattern in patterns.items():
        similarity = cosine_similarity_sparse(query_bits, pattern.bits)
        results.append((pattern_id, similarity))

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]


def random_retrieval(
    patterns: Dict[str, SubstratePattern],
    top_k: int = 5,
    rng: Optional[random.Random] = None
) -> List[Tuple[str, float]]:
    """Random baseline - picks top_k patterns at random.

    This is the floor baseline. Any meaningful retrieval method should
    significantly outperform random selection.

    Args:
        patterns: Dictionary mapping pattern_id to SubstratePattern
        top_k: Number of results to return
        rng: Optional seeded random generator for reproducibility

    Returns:
        List of (pattern_id, random_score) tuples.
        Scores are random for consistency with other baselines' format.
    """
    if rng is None:
        rng = random.Random()

    pattern_ids = list(patterns.keys())
    rng.shuffle(pattern_ids)

    selected = pattern_ids[:top_k]

    # Assign random scores (descending so first is "best")
    results = []
    for i, pattern_id in enumerate(selected):
        # Score decreases with rank to mimic sorted output
        score = 1.0 - (i / max(top_k, 1))
        results.append((pattern_id, score))

    return results
