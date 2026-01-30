"""Complex sparse pattern representation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class ComplexSparsePattern:
    """A sparse pattern with complex amplitudes.

    Each active dimension has a magnitude (strength) and phase (timing/relation).
    Stored sparsely for efficiency - only active dimensions are tracked.

    Attributes:
        dim: Total dimensionality of the pattern space.
        indices: Which dimensions are active (sorted).
        magnitudes: Magnitude at each active dimension.
        phases: Phase (0 to 2pi) at each active dimension.
    """

    dim: int
    indices: tuple[int, ...]
    magnitudes: tuple[float, ...]
    phases: tuple[float, ...]

    def __post_init__(self):
        """Validate and normalize."""
        assert len(self.indices) == len(self.magnitudes) == len(self.phases)
        assert all(0 <= i < self.dim for i in self.indices)
        # Ensure indices are sorted for consistent operations
        if self.indices != tuple(sorted(self.indices)):
            sorted_order = sorted(range(len(self.indices)), key=lambda i: self.indices[i])
            object.__setattr__(self, "indices", tuple(self.indices[i] for i in sorted_order))
            object.__setattr__(self, "magnitudes", tuple(self.magnitudes[i] for i in sorted_order))
            object.__setattr__(self, "phases", tuple(self.phases[i] for i in sorted_order))

    @classmethod
    def random(
        cls,
        dim: int,
        k: int,
        generator: torch.Generator | None = None,
    ) -> ComplexSparsePattern:
        """Create a random sparse pattern.

        Args:
            dim: Dimensionality of pattern space.
            k: Number of active dimensions.
            generator: Optional RNG for reproducibility.

        Returns:
            Random pattern with k active dimensions, unit magnitudes, random phases.
        """
        # Random indices
        indices = torch.randperm(dim, generator=generator)[:k].sort().values.tolist()

        # Unit magnitudes (can be varied later if needed)
        magnitudes = [1.0] * k

        # Random phases in [0, 2pi)
        phases = (torch.rand(k, generator=generator) * 2 * math.pi).tolist()

        return cls(dim=dim, indices=tuple(indices), magnitudes=tuple(magnitudes), phases=tuple(phases))

    def to_dense(self) -> torch.Tensor:
        """Convert to dense complex tensor.

        Returns:
            Complex tensor of shape (dim,) with non-zero entries at active indices.
        """
        dense = torch.zeros(self.dim, dtype=torch.complex64)
        for idx, mag, phase in zip(self.indices, self.magnitudes, self.phases):
            # complex = magnitude * e^(i*phase)
            dense[idx] = mag * torch.exp(torch.tensor(1j * phase))
        return dense

    @classmethod
    def from_dense(cls, dense: torch.Tensor, threshold: float = 1e-6) -> ComplexSparsePattern:
        """Create from dense complex tensor.

        Args:
            dense: Complex tensor to sparsify.
            threshold: Minimum magnitude to consider non-zero.

        Returns:
            Sparse pattern with only significant entries.
        """
        magnitudes = dense.abs()
        active_mask = magnitudes > threshold
        indices = torch.where(active_mask)[0].tolist()

        mags = magnitudes[active_mask].tolist()
        phases = torch.angle(dense[active_mask]).tolist()
        # Normalize phases to [0, 2pi)
        phases = [(p + 2 * math.pi) % (2 * math.pi) for p in phases]

        return cls(
            dim=len(dense),
            indices=tuple(indices),
            magnitudes=tuple(mags),
            phases=tuple(phases),
        )

    @property
    def k(self) -> int:
        """Number of active dimensions."""
        return len(self.indices)
