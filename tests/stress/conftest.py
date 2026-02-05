# tests/stress/conftest.py
"""Stress test infrastructure and fixtures.

Provides:
- STRESS_NOISE_LEVELS: Noise levels for stress testing (MEDIUM, HIGH)
- StressMetrics: Dataclass for capturing test metrics
- capture_retrieval_metrics(): Extract metrics from retrieval results
- evaluate_success_criteria(): Evaluate test success against criteria

All stress tests skip NONE/LOW noise levels per CONTEXT.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

import pytest

from tests.conftest import NoiseLevel


# Skip NONE/LOW per CONTEXT.md - stress tests need meaningful noise
STRESS_NOISE_LEVELS = [NoiseLevel.MEDIUM, NoiseLevel.HIGH]


def pytest_addoption(parser: Any) -> None:
    """Add custom command-line options for stress tests."""
    parser.addoption(
        "--update-report",
        action="store_true",
        default=False,
        help="Generate/update metrics report (slow, runs many trials)",
    )


def pytest_configure(config: Any) -> None:
    """Register the stress test marker."""
    config.addinivalue_line(
        "markers",
        "stress: mark test as a stress test (runs with noisy memory)",
    )


@dataclass
class StressMetrics:
    """Metrics captured during stress test execution.

    Attributes:
        test_name: Name of the test that produced these metrics.
        noise_level: Noise level preset used (MEDIUM, HIGH).
        success: Whether the test passed its success criteria.
        metrics: Flexible dict for test-specific metrics.
    """

    test_name: str
    noise_level: str
    success: bool
    metrics: dict[str, Any]

    def to_json(self) -> str:
        """Serialize metrics to JSON for Phase 8 analysis."""
        return json.dumps(asdict(self), indent=2)


def capture_retrieval_metrics(
    results: list,  # list[RetrievalResult]
    target_id: str,
    store: Any,  # MemoryStore
) -> dict[str, Any]:
    """Extract metrics from retrieval results.

    Args:
        results: List of RetrievalResult from retrieval operation.
        target_id: Pattern ID of the expected target.
        store: MemoryStore instance.

    Returns:
        Dict with:
        - target_rank: Position of target in results (1-indexed), None if not found
        - target_score: Score of target if found
        - top_score: Best score in results
        - total_patterns: Total pattern count in store
    """
    target_rank = None
    target_score = None
    top_score = results[0].score if results else None

    for i, result in enumerate(results):
        if result.pattern_id == target_id:
            target_rank = i + 1  # 1-indexed
            target_score = result.score
            break

    return {
        "target_rank": target_rank,
        "target_score": target_score,
        "top_score": top_score,
        "total_patterns": store.pattern_count,
    }


@pytest.fixture
def update_report(request: Any) -> bool:
    """Check if --update-report flag is set."""
    return request.config.getoption("--update-report")


def evaluate_success_criteria(
    results: list,  # list[RetrievalResult]
    target_id: str,
    near_miss_ids: list[str] | None = None,
    top_k: int = 3,
) -> dict[str, bool]:
    """Evaluate retrieval results against success criteria.

    Args:
        results: List of RetrievalResult from retrieval operation.
        target_id: Pattern ID of the expected target.
        near_miss_ids: Optional list of near-miss pattern IDs to check.
        top_k: K for relaxed top-K criterion.

    Returns:
        Dict with:
        - strict_top_1: True if target is rank 1
        - relaxed_top_k: True if target is in top-K
        - relative_to_near_misses: True if target ranks above all near-misses
    """
    result_ids = [r.pattern_id for r in results]

    # Find target position (0-indexed)
    target_pos = None
    if target_id in result_ids:
        target_pos = result_ids.index(target_id)

    # Strict: target is rank 1
    strict_top_1 = target_pos == 0 if target_pos is not None else False

    # Relaxed: target in top-K
    relaxed_top_k = target_pos is not None and target_pos < top_k

    # Relative: target ranks above all near-misses
    relative_to_near_misses = True
    if near_miss_ids and target_pos is not None:
        for nm_id in near_miss_ids:
            if nm_id in result_ids:
                nm_pos = result_ids.index(nm_id)
                if nm_pos < target_pos:
                    # Near-miss ranked higher than target
                    relative_to_near_misses = False
                    break
    elif near_miss_ids and target_pos is None:
        # Target not found at all
        relative_to_near_misses = False

    return {
        "strict_top_1": strict_top_1,
        "relaxed_top_k": relaxed_top_k,
        "relative_to_near_misses": relative_to_near_misses,
    }
