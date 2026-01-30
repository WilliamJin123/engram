"""Quantum-inspired substrate for memory systems."""

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity
from quantum_substrate.interference import (
    jaccard_retrieval,
    interference_retrieval,
    create_related_pattern,
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
]
