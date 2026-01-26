"""Engram: Self-organizing memory framework with epistemic uncertainty.

A memory system using distributional High-Dimensional Vectors (HDVs)
for principled uncertainty tracking in knowledge graphs.
"""

__version__ = "0.2.0"  # Bump for distributional HDV release

from .hdv import (
    DistributionalHDV,
    UncertaintyParams,
    random_distributional,
    distributional_bind,
    distributional_unbind,
    distributional_similarity,
    bayesian_update,
    temporal_decay,
    human_confirm,
    handle_contradiction,
    bundle_observations,
    propagate_through_edge,
)
from .graph import Edge

__all__ = [
    # Version
    "__version__",
    # Core types
    "DistributionalHDV",
    "UncertaintyParams",
    "Edge",
    # Factory
    "random_distributional",
    # Operations
    "distributional_bind",
    "distributional_unbind",
    "distributional_similarity",
    # Uncertainty management
    "bayesian_update",
    "temporal_decay",
    "human_confirm",
    "handle_contradiction",
    "bundle_observations",
    "propagate_through_edge",
]
