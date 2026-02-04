"""Surprise detection for quantum-inspired memory substrate.

Surprise emerges from mismatch between expected and actual retrieval results.
This is not externally injected but measured as the distance between:
- Expected superposition (built from query + coherence-weighted patterns)
- Actual top result (what retrieval returned)

Surprise magnitude determines re-coherence strength for involved patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from quantum_substrate.protocols import SubstratePattern

if TYPE_CHECKING:
    from quantum_substrate.coherence import CoherenceManager


@dataclass
class SurpriseResult:
    """Result of surprise detection.

    Attributes:
        magnitude: Surprise level 0-1 (0 = expected, 1 = completely unexpected).
        surprising_pattern_id: ID of unexpected top result (None if expected matched).
        expected_pattern_id: ID we expected as top (None if prediction was correct).
        participant_scores: All patterns involved in retrieval and their contribution.
    """

    magnitude: float
    surprising_pattern_id: str | None
    expected_pattern_id: str | None
    participant_scores: dict[str, float]


def compute_surprise_magnitude(
    expected_bits: set[int],
    actual_bits: set[int],
) -> float:
    """Compute surprise as normalized Hamming distance.

    Formula: (bits_in_A + bits_in_B - 2 * overlap) / total_active_bits

    This is the symmetric difference normalized by union size.
    Native to SDR representation, gives 0-1 regardless of pattern size.

    Args:
        expected_bits: Bit indices we expected to be active.
        actual_bits: Bit indices that were actually active.

    Returns:
        Surprise magnitude in [0, 1].
        - 0.0: Identical patterns (no surprise)
        - 1.0: No overlap (maximum surprise)
    """
    if not expected_bits and not actual_bits:
        return 0.0

    overlap = len(expected_bits & actual_bits)
    total_unique = len(expected_bits | actual_bits)

    if total_unique == 0:
        return 0.0

    # Symmetric difference = |A| + |B| - 2*|A & B|
    symmetric_diff = len(expected_bits) + len(actual_bits) - 2 * overlap

    return symmetric_diff / total_unique


class SurpriseDetector:
    """Detects surprise in retrieval results.

    Surprise is emergent: we build an expectation from current coherence
    state, then measure how much the actual result differs.

    Usage:
        detector = SurpriseDetector()
        expected_bits, weights = detector.build_expectation(query, patterns, coherence_mgr)
        result = detector.detect(expected_bits, weights, actual_pattern, actual_id, retrieval_results)
    """

    def __init__(self, expectation_threshold: float = 0.1):
        """Initialize detector.

        Args:
            expectation_threshold: Minimum (coherence * overlap_ratio) to include
                a pattern in expectation building. Default 0.1.
        """
        self.expectation_threshold = expectation_threshold

    def build_expectation(
        self,
        query_bits: set[int],
        patterns: dict[str, "SubstratePattern"],
        coherence_manager: "CoherenceManager",
    ) -> tuple[set[int], dict[str, float]]:
        """Build expected bit pattern from query and coherence-weighted patterns.

        For each pattern, compute its contribution weight as:
            coherence * overlap_ratio
        where overlap_ratio = overlap_with_query / query_size

        Patterns exceeding threshold contribute their bits to expectation.

        Args:
            query_bits: Active bits in the query.
            patterns: All patterns by ID.
            coherence_manager: For computing decayed coherence.

        Returns:
            Tuple of (expected_bits, pattern_weights):
            - expected_bits: Union of query bits + weighted pattern bits
            - pattern_weights: Dict of pattern_id -> contribution weight
        """
        expected_bits = set(query_bits)
        pattern_weights: dict[str, float] = {}

        if not query_bits:
            return expected_bits, pattern_weights

        query_size = len(query_bits)

        for pattern_id, pattern in patterns.items():
            # Compute decayed coherence
            coherence = coherence_manager.compute_decayed_coherence(pattern)

            # Overlap ratio with query
            overlap = len(pattern.bits & query_bits)
            overlap_ratio = overlap / query_size if query_size > 0 else 0.0

            # Contribution weight
            weight = coherence * overlap_ratio
            pattern_weights[pattern_id] = weight

            # Include pattern bits if above threshold
            if weight > self.expectation_threshold:
                expected_bits |= pattern.bits

        return expected_bits, pattern_weights

    def detect(
        self,
        expected_bits: set[int],
        expected_weights: dict[str, float],
        actual_pattern: "SubstratePattern",
        actual_pattern_id: str,
        retrieval_results: Sequence[tuple[str, float]],
    ) -> SurpriseResult:
        """Detect surprise by comparing expected vs actual.

        Args:
            expected_bits: Bits we expected to see (from build_expectation).
            expected_weights: Pattern contribution weights (from build_expectation).
            actual_pattern: The top retrieval result pattern.
            actual_pattern_id: ID of actual top result.
            retrieval_results: List of (pattern_id, score) from retrieval.

        Returns:
            SurpriseResult with magnitude and involved patterns.
        """
        # Compute surprise magnitude
        actual_bits = actual_pattern.bits
        magnitude = compute_surprise_magnitude(expected_bits, actual_bits)

        # Find expected top pattern (highest weight)
        expected_top_id: str | None = None
        if expected_weights:
            expected_top_id = max(expected_weights.keys(), key=lambda k: expected_weights[k])

        # Determine if result was surprising
        surprising_pattern_id: str | None = None
        expected_pattern_id: str | None = None

        if expected_top_id and expected_top_id != actual_pattern_id:
            # Prediction was wrong
            surprising_pattern_id = actual_pattern_id
            expected_pattern_id = expected_top_id

        # Build participant scores from retrieval results
        participant_scores: dict[str, float] = {}
        for pattern_id, score in retrieval_results:
            participant_scores[pattern_id] = score

        return SurpriseResult(
            magnitude=magnitude,
            surprising_pattern_id=surprising_pattern_id,
            expected_pattern_id=expected_pattern_id,
            participant_scores=participant_scores,
        )
