"""High-Dimensional Vector (HDV) operations using ternary encoding.

This module implements the core HDV primitives for hyperdimensional computing
using ternary vectors with values in {-1, 0, +1}.
"""

import torch

from engram.hdv.distributional import DistributionalHDV


def random_ternary(
    dim: int, sparsity: float = 0.5, seed: int | None = None
) -> torch.Tensor:
    """Generate a random ternary vector with values in {-1, 0, +1}.

    Args:
        dim: Dimension of the vector.
        sparsity: Fraction of non-zero elements (default 0.5).
        seed: Optional random seed for reproducibility.

    Returns:
        Tensor of shape (dim,) with values in {-1.0, 0.0, +1.0}.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None

    # Create mask for non-zero positions
    mask = torch.rand(dim, generator=gen) < sparsity

    # Generate random signs for non-zero positions
    signs = 2 * torch.randint(0, 2, (dim,), generator=gen, dtype=torch.float32) - 1

    # Apply mask: non-zero positions get their sign, others get 0
    return signs * mask.float()


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bind two vectors using element-wise multiplication.

    For ternary vectors, this creates an association between a and b.
    The result is also ternary: product of {-1,0,+1} values stays in {-1,0,+1}.

    Args:
        a: First ternary vector.
        b: Second ternary vector.

    Returns:
        Element-wise product a * b.
    """
    return a * b


def unbind(bound: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Unbind a vector by removing the contribution of another vector.

    For ternary vectors, unbind(bind(a,b), a) recovers b where both a and b
    were non-zero. Information is lost where a was zero.

    Args:
        bound: A bound vector (result of bind operation).
        a: The vector to unbind with.

    Returns:
        Element-wise product bound * a (inverse of bind for non-zero elements).
    """
    return bound * a


def bundle(items: list[torch.Tensor]) -> torch.Tensor:
    """Bundle multiple vectors into one via element-wise sum + normalize.

    Creates a superposition of vectors where each item remains queryable
    via similarity. Capacity degrades as ~1/sqrt(n) with n items.

    Args:
        items: List of vectors to bundle.

    Returns:
        Normalized sum of all vectors.

    Raises:
        ValueError: If items list is empty.
    """
    if len(items) == 0:
        raise ValueError("Cannot bundle empty list")

    summed = torch.stack(items).sum(dim=0)
    return normalize(summed)


def similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in range [-1, 1].
    """
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return (torch.dot(a, b) / (norm_a * norm_b)).item()


def normalize(v: torch.Tensor) -> torch.Tensor:
    """Normalize a vector to unit length.

    Args:
        v: Vector to normalize.

    Returns:
        Unit vector in same direction, or zero vector if input is zero.
    """
    norm = torch.norm(v)
    if norm == 0:
        return torch.zeros_like(v)
    return v / norm


def distributional_bind(a: DistributionalHDV, b: DistributionalHDV) -> DistributionalHDV:
    """Bind two distributional HDVs.

    Creates an association between two concepts. For the mean, this is
    element-wise multiplication. Variances compound according to the
    product of random variables formula.

    For product of random variables X*Y:
    Var(XY) ≈ E[X]²Var(Y) + E[Y]²Var(X) + Var(X)Var(Y)

    Args:
        a: First distributional HDV.
        b: Second distributional HDV.

    Returns:
        A new DistributionalHDV representing the bound association.
    """
    # Mean: element-wise product
    new_mean = a.mean * b.mean

    # Variance: product of random variables formula
    new_variance = (
        a.mean ** 2 * b.variance +
        b.mean ** 2 * a.variance +
        a.variance * b.variance
    )

    # Ensure positive variance
    new_variance = torch.clamp(new_variance, min=1e-10)

    return DistributionalHDV(
        mean=new_mean,
        variance=new_variance,
        last_accessed=max(a.last_accessed, b.last_accessed),
        last_updated=max(a.last_updated, b.last_updated),
    )


def distributional_unbind(bound: DistributionalHDV, key: DistributionalHDV) -> DistributionalHDV:
    """Unbind a distributional HDV with a key.

    For ternary-mean vectors, unbinding is the same operation as binding.
    This is because for values in {-1, 0, +1}, the multiplicative inverse
    equals the value itself (x * x = 1 for x in {-1, +1}).

    Args:
        bound: A bound distributional HDV.
        key: The key to unbind with.

    Returns:
        A new DistributionalHDV representing the unbound result.
    """
    return distributional_bind(bound, key)
