"""Sleep agents for graph reorganization.

Sleep agents are hardcoded processes (DNA-level) that operate on the
knowledge graph during consolidation. They are not nodes themselves
and don't follow node physics.
"""

from .clusterer import Clusterer

__all__ = ["Clusterer"]
