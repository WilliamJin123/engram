"""Sleep agents for graph reorganization.

Sleep agents are hardcoded processes (DNA-level) that operate on the
knowledge graph during consolidation. They are not nodes themselves
and don't follow node physics.
"""

from .clusterer import Clusterer
from .abstractor import Abstractor
from .contradiction import ContradictionDetector

__all__ = ["Clusterer", "Abstractor", "ContradictionDetector"]
