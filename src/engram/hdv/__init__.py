"""High-Dimensional Vector (HDV) operations with epistemic uncertainty.

This package provides distributional HDVs (diagonal Gaussians over HDV space)
for principled uncertainty tracking in knowledge graphs.
"""

from .distributional import DistributionalHDV, random_distributional
from .uncertainty import (
    UncertaintyParams,
    kl_divergence,
    symmetric_kl,
    distributional_similarity,
    bayesian_update,
    temporal_decay,
    human_confirm,
    handle_contradiction,
    bundle_observations,
    propagate_through_edge,
)
from .operations import (
    distributional_bind,
    distributional_unbind,
    # Legacy exports (deprecated)
    random_ternary,
    bind,
    unbind,
    bundle,
    similarity,
    normalize,
)

__all__ = [
    # Core types
    "DistributionalHDV",
    "UncertaintyParams",
    # Factory
    "random_distributional",
    # Distributional operations
    "distributional_bind",
    "distributional_unbind",
    "distributional_similarity",
    # Uncertainty management
    "kl_divergence",
    "symmetric_kl",
    "bayesian_update",
    "temporal_decay",
    "human_confirm",
    "handle_contradiction",
    "bundle_observations",
    "propagate_through_edge",
    # Legacy (deprecated)
    "random_ternary",
    "bind",
    "unbind",
    "bundle",
    "similarity",
    "normalize",
]
