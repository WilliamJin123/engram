"""Quantum-inspired substrate for memory systems."""

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity

__version__ = "0.1.0"
__all__ = ["ComplexSparsePattern", "bind_hrr", "unbind_hrr", "similarity"]
