"""Distributional HDV representation using diagonal Gaussians."""

from dataclasses import dataclass
import torch


@dataclass
class DistributionalHDV:
    """A probability distribution over HDV space (diagonal Gaussian).

    Represents epistemic uncertainty about a concept's HDV representation.
    Each dimension has independent variance, allowing per-feature uncertainty
    (e.g., confident about 'has legs', uncertain about 'size').

    Attributes:
        mean: Expected HDV location (mu).
        variance: Per-dimension uncertainty (sigma squared).
        last_accessed: Timestamp of last access (for decay calculation).
        last_updated: Timestamp of last evidence update.
    """
    mean: torch.Tensor
    variance: torch.Tensor
    last_accessed: float = 0.0
    last_updated: float = 0.0

    def __post_init__(self):
        """Validate inputs after dataclass initialization."""
        if self.mean.shape != self.variance.shape:
            raise ValueError(
                f"Mean and variance must have same shape. "
                f"Got mean={self.mean.shape}, variance={self.variance.shape}"
            )
        if (self.variance <= 0).any():
            raise ValueError("Variance must be positive (> 0) for all dimensions")

    @property
    def dim(self) -> int:
        """Return the dimensionality of this HDV."""
        return self.mean.shape[0]
