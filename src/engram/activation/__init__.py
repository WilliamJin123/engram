"""Spreading activation for knowledge graph queries.

Activation is TRANSIENT - computed during queries, not stored on nodes.
This matches biological neural networks where activation is computed
each forward pass, while weights (strength) are stored.
"""

from .spreading import spread_activation, query, multi_query

__all__ = [
    "spread_activation",
    "query",
    "multi_query",
]
