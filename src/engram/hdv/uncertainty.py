"""Uncertainty calculations for distributional HDVs."""

import torch
from .distributional import DistributionalHDV


def kl_divergence(p: DistributionalHDV, q: DistributionalHDV) -> float:
    """Compute KL divergence KL(P || Q) for diagonal Gaussians.

    For diagonal Gaussians:
    KL(P||Q) = 0.5 * sum_i [
        log(sigma^2_q,i / sigma^2_p,i)
        + sigma^2_p,i / sigma^2_q,i
        + (mu_p,i - mu_q,i)^2 / sigma^2_q,i
        - 1
    ]

    Args:
        p: First distribution (the "true" distribution).
        q: Second distribution (the "approximating" distribution).

    Returns:
        KL divergence (non-negative float).
    """
    # Avoid division by zero / log(0) with small epsilon
    eps = 1e-10
    var_p = p.variance + eps
    var_q = q.variance + eps

    # Per-dimension KL terms
    log_term = torch.log(var_q / var_p)
    ratio_term = var_p / var_q
    mean_diff_sq = (p.mean - q.mean) ** 2
    mahalanobis_term = mean_diff_sq / var_q

    # Sum over dimensions
    kl = 0.5 * torch.sum(log_term + ratio_term + mahalanobis_term - 1)

    return max(0.0, kl.item())  # Ensure non-negative due to numerical issues


def symmetric_kl(p: DistributionalHDV, q: DistributionalHDV) -> float:
    """Compute symmetric (Jeffrey's) KL divergence.

    J(P, Q) = (KL(P||Q) + KL(Q||P)) / 2

    This is symmetric: J(P, Q) == J(Q, P).

    Args:
        p: First distribution.
        q: Second distribution.

    Returns:
        Symmetric KL divergence (non-negative float).
    """
    return (kl_divergence(p, q) + kl_divergence(q, p)) / 2
