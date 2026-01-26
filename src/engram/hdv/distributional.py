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


def random_distributional(
    dim: int,
    initial_variance: float = 0.5,
    sparsity: float = 0.5,
    current_time: float = 0.0,
    seed: int | None = None,
) -> DistributionalHDV:
    """Create a random DistributionalHDV with ternary mean.

    Args:
        dim: Dimension of the HDV.
        initial_variance: Uniform variance for all dimensions.
        sparsity: Fraction of non-zero elements in mean (0-1).
        current_time: Timestamp for initialization.
        seed: Random seed for reproducibility.

    Returns:
        A new DistributionalHDV with random ternary mean.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None

    # Generate ternary mean
    mask = torch.rand(dim, generator=gen) < sparsity
    signs = 2 * torch.randint(0, 2, (dim,), generator=gen, dtype=torch.float32) - 1
    mean = signs * mask.float()

    # Uniform variance
    variance = torch.full((dim,), initial_variance)

    return DistributionalHDV(
        mean=mean,
        variance=variance,
        last_accessed=current_time,
        last_updated=current_time,
    )
