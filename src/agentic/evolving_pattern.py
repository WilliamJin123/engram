# src/agentic/evolving_pattern.py
"""Evolving pattern with coactivation learning support.

Patterns track their original bits separately from acquired bits,
enabling coactivation-based learning while preserving binding integrity.

Key insight: For HRR binding operations, we use ORIGINAL bits only.
For retrieval, we use ALL bits (original + acquired).
This prevents transitive pollution while enabling learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.text_encoder import EncodedPattern


@dataclass
class EvolvingPattern:
    """A sparse pattern that can evolve through coactivation learning.

    Attributes:
        dim: Total dimensionality of the pattern space.
        bits: Current active bit indices (original + acquired).
        original_bits: Frozen bits from initial encoding (used for binding).
        acquired_bits: Bits gained through coactivation.
        phases: Phase (0 to 2pi) at each active bit.
        text: Source text that was encoded.
        metadata: Arbitrary metadata (speaker, time, topic, etc).
        acquisition_count: Number of coactivation events.
        coherence: Current coherence value (0-1). New patterns start at 1.0.
        last_access_tick: Global tick when pattern was last accessed.
        connection_count: Number of bindings/coactivations (for embeddedness).
    """

    dim: int
    bits: set[int]
    original_bits: frozenset[int]
    acquired_bits: set[int]
    phases: dict[int, float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    acquisition_count: int = 0
    coherence: float = 1.0
    last_access_tick: int = 0
    connection_count: int = 0
    last_modified_tick: int = 0
    access_count_since_modification: int = 0

    @classmethod
    def from_encoded(cls, encoded: "EncodedPattern") -> EvolvingPattern:
        """Create from an EncodedPattern."""
        return cls(
            dim=encoded.dim,
            bits=set(encoded.bits),
            original_bits=frozenset(encoded.bits),
            acquired_bits=set(),
            phases=dict(encoded.phases),
            text=encoded.text,
            metadata=dict(encoded.metadata),
            acquisition_count=0,
            coherence=1.0,
            last_access_tick=0,
            connection_count=0,
            last_modified_tick=0,
            access_count_since_modification=0,
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        dim: int = 1024,
        k: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> EvolvingPattern:
        """Create a pattern directly from text."""
        from agentic.text_encoder import TextEncoder

        encoder = TextEncoder(dim=dim, k=k)
        encoded = encoder.encode(text, metadata=metadata)
        return cls.from_encoded(encoded)

    def to_dense(self) -> "torch.Tensor":
        """Convert to dense complex tensor for HRR operations.

        Uses ORIGINAL bits only to preserve binding integrity.
        """
        import torch

        dense = torch.zeros(self.dim, dtype=torch.complex64)
        for bit in self.original_bits:
            phase = self.phases.get(bit, 0.0)
            dense[bit] = torch.exp(torch.tensor(1j * phase))
        return dense

    def to_dense_evolved(self) -> "torch.Tensor":
        """Convert to dense complex tensor using ALL bits.

        Includes both original and acquired bits.
        Use this for retrieval (similarity search).
        """
        import torch

        dense = torch.zeros(self.dim, dtype=torch.complex64)
        for bit in self.bits:
            phase = self.phases.get(bit, 0.0)
            dense[bit] = torch.exp(torch.tensor(1j * phase))
        return dense

    @property
    def obesity(self) -> float:
        """Ratio of current bits to original bits."""
        if not self.original_bits:
            return 0.0
        return len(self.bits) / len(self.original_bits)

    @property
    def acquired_ratio(self) -> float:
        """Fraction of current bits that were acquired."""
        if not self.bits:
            return 0.0
        return len(self.acquired_bits) / len(self.bits)

    @property
    def embeddedness(self) -> float:
        """Structural embeddedness based on connection count.

        Higher embeddedness = slower decay. Scale factor 0.1 means
        10 connections halves the effective decay rate.
        """
        return 1.0 + 0.1 * self.connection_count

    def clamp_coherence(self, floor: float = 0.01) -> None:
        """Ensure coherence stays within valid bounds."""
        self.coherence = max(floor, min(1.0, self.coherence))

    @property
    def stability_score(self) -> float:
        """Stability score for crystallization dynamics.

        Based on access_count_since_modification:
        - 0 accesses = 0.0 (just modified, fully malleable)
        - 10+ accesses = 1.0 (stable, crystallizes faster)

        Higher stability = pattern hasn't changed despite repeated access,
        so it should crystallize (coherence decays faster).
        """
        return min(1.0, self.access_count_since_modification / 10.0)

    def mark_modified(self, tick: int) -> None:
        """Mark pattern as modified at given tick.

        Resets stability tracking since pattern bits have changed.

        Args:
            tick: Current global tick when modification occurred.
        """
        self.last_modified_tick = tick
        self.access_count_since_modification = 0

    def record_access(self) -> None:
        """Record an access without modification.

        Increments access counter for stability tracking.
        Call this when pattern is retrieved/used but not modified.
        """
        self.access_count_since_modification += 1
