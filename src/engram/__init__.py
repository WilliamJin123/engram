"""Engram: Self-organizing memory framework with epistemic uncertainty.

A memory system using distributional High-Dimensional Vectors (HDVs)
for principled uncertainty tracking in knowledge graphs.
"""

__version__ = "0.4.0"  # Bump for architecture implementation

from .hdv import (
    DistributionalHDV,
    UncertaintyParams,
    random_distributional,
    distributional_bind,
    distributional_unbind,
    distributional_similarity,
    bayesian_update,
    bayesian_update_with_mass,
    temporal_decay,
    human_confirm,
    handle_contradiction,
    bundle_observations,
    propagate_through_edge,
)
from .graph import Edge, Node, KnowledgeGraph
from .sleep import Clusterer, Abstractor, ContradictionDetector
from .drive import CuriosityDrive
from .strategy import create_strategy_node, is_strategy_node, StrategyMatcher

__all__ = [
    # Version
    "__version__",
    # Core types
    "DistributionalHDV",
    "UncertaintyParams",
    "Edge",
    "Node",
    "KnowledgeGraph",
    # Factory
    "random_distributional",
    # Operations
    "distributional_bind",
    "distributional_unbind",
    "distributional_similarity",
    # Uncertainty management
    "bayesian_update",
    "bayesian_update_with_mass",
    "temporal_decay",
    "human_confirm",
    "handle_contradiction",
    "bundle_observations",
    "propagate_through_edge",
    # Sleep agents (hardcoded)
    "Clusterer",
    "Abstractor",
    "ContradictionDetector",
    # Drives (hardcoded)
    "CuriosityDrive",
    # Strategies (emergent)
    "create_strategy_node",
    "is_strategy_node",
    "StrategyMatcher",
]
