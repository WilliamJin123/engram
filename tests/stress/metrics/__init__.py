# tests/stress/metrics/__init__.py
"""Metrics module for Phase 8 validation.

Provides:
- Retrieval metrics: MRR, Recall@K, rank extraction
- Statistical testing: Mann-Whitney U comparison
- Visualization: Degradation curves with error bands
"""

from tests.stress.metrics.retrieval_metrics import (
    compute_mrr,
    compute_recall_at_k,
    get_target_rank,
)
from tests.stress.metrics.statistical import compare_methods
from tests.stress.metrics.visualization import (
    plot_degradation_curve,
    plot_method_comparison,
)

__all__ = [
    "compute_mrr",
    "compute_recall_at_k",
    "get_target_rank",
    "compare_methods",
    "plot_degradation_curve",
    "plot_method_comparison",
]
