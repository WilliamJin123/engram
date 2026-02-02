"""Quantum-inspired substrate for memory systems."""

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity
from quantum_substrate.interference import (
    jaccard_retrieval,
    interference_retrieval,
    create_related_pattern,
)
from quantum_substrate.coherence import CoherenceManager, CoherenceConfig
from quantum_substrate.surprise import (
    SurpriseDetector,
    SurpriseResult,
    compute_surprise_magnitude,
)
from quantum_substrate.tunneling import (
    TunnelingConfig,
    TunnelingResult,
    CreativeModeTracker,
    attempt_tunneling,
)

__version__ = "0.1.0"
__all__ = [
    "ComplexSparsePattern",
    "bind_hrr",
    "unbind_hrr",
    "similarity",
    "jaccard_retrieval",
    "interference_retrieval",
    "create_related_pattern",
    "CoherenceManager",
    "CoherenceConfig",
    "SurpriseDetector",
    "SurpriseResult",
    "compute_surprise_magnitude",
    "TunnelingConfig",
    "TunnelingResult",
    "CreativeModeTracker",
    "attempt_tunneling",
]
