# src/quantum_substrate/protocols.py
"""Protocol definitions for quantum-inspired memory substrate.

This module defines the interfaces that substrate operations depend on.
By using protocols (structural subtyping), the substrate layer avoids
importing from the agentic layer while still being able to type-check
against pattern implementations.

Key insight: Substrate defines WHAT it needs; agentic layer provides HOW.
This inverts the dependency direction, achieving clean architecture.
"""

from __future__ import annotations

from typing import Protocol, Set, FrozenSet, Dict, runtime_checkable


@runtime_checkable
class SubstratePattern(Protocol):
    """Protocol for patterns usable by substrate operations.

    This is the interface that coherence, surprise, and tunneling modules
    depend on. Any class implementing these attributes works with substrate
    operations via structural subtyping (duck typing with type checking).

    The EvolvingPattern class in the agentic layer implements this protocol
    by having all the required attributes. No explicit inheritance needed -
    Python's Protocol system checks structural compatibility.

    Usage in substrate modules:
        from quantum_substrate.protocols import SubstratePattern

        def apply_decay(pattern: SubstratePattern) -> float:
            # Works with any object having the required attributes
            return pattern.coherence * pattern.embeddedness

    Required attributes:
        dim: Pattern dimensionality (vector space size)
        bits: Current active bit indices (mutable, includes acquired bits)
        original_bits: Frozen original bits (immutable, for binding operations)
        phases: Phase angle at each active bit (for interference effects)
        coherence: Current coherence 0-1 (read/write for decay/refresh)
        last_access_tick: Global tick of last access (read/write for decay timing)
        embeddedness: Structural embeddedness (read-only, affects decay rate)
        stability_score: Stability for crystallization (read-only, affects decay rate)

    Required methods:
        record_access(): Record access without modification (for stability tracking)
    """

    @property
    def dim(self) -> int:
        """Pattern dimensionality (total bits in vector space)."""
        ...

    @property
    def bits(self) -> Set[int]:
        """Current active bit indices (original + acquired)."""
        ...

    @property
    def original_bits(self) -> FrozenSet[int]:
        """Frozen original bits from initial encoding.

        Used for binding operations to prevent transitive pollution.
        """
        ...

    @property
    def phases(self) -> Dict[int, float]:
        """Phase angle (0 to 2pi) at each active bit.

        Used for interference effects in retrieval.
        """
        ...

    @property
    def coherence(self) -> float:
        """Current coherence value 0-1.

        High coherence (near 1.0): pattern participates in interference effects.
        Low coherence (near 0.01): pattern behaves classically.
        """
        ...

    @coherence.setter
    def coherence(self, value: float) -> None:
        """Set coherence value (used by decay/refresh operations)."""
        ...

    @property
    def last_access_tick(self) -> int:
        """Global tick when pattern was last accessed.

        Used for lazy decay calculation.
        """
        ...

    @last_access_tick.setter
    def last_access_tick(self, value: int) -> None:
        """Set last access tick (used by refresh operations)."""
        ...

    @property
    def embeddedness(self) -> float:
        """Structural embeddedness based on connection count.

        Higher embeddedness = slower decay rate.
        Formula: 1.0 + 0.1 * connection_count
        """
        ...

    @property
    def stability_score(self) -> float:
        """Stability score for crystallization dynamics.

        Based on access_count_since_modification:
        - 0 accesses = 0.0 (just modified, fully malleable)
        - 10+ accesses = 1.0 (stable, crystallizes faster)

        Higher stability = pattern hasn't changed despite repeated access.
        """
        ...

    def record_access(self) -> None:
        """Record an access without modification.

        Increments access counter for stability tracking.
        Call this when pattern is retrieved/used but not modified.
        """
        ...
