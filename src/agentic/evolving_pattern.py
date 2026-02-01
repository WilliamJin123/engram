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
    """

    dim: int
    bits: set[int]
    original_bits: frozenset[int]
    acquired_bits: set[int]
    phases: dict[int, float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    acquisition_count: int = 0

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
