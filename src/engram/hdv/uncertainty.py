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


def distributional_similarity(
    a: DistributionalHDV,
    b: DistributionalHDV,
    kl_scale: float = 1.0,
) -> tuple[float, float]:
    """Compute similarity between distributional HDVs with uncertainty.

    Uses hyperbolic transformation of symmetric KL divergence:
    similarity = 1 / (1 + kl_scale * KL)

    This preserves weak similarity for distant distributions (never reaches 0),
    matching the intuition that most concepts have some connection.

    Args:
        a: First distributional HDV.
        b: Second distributional HDV.
        kl_scale: Scaling factor for KL divergence (higher = stricter).

    Returns:
        Tuple of (similarity, uncertainty):
        - similarity: Value in (0, 1], where 1 = identical distributions.
        - uncertainty: Estimate of how uncertain this similarity is.
    """
    # Compute symmetric KL divergence
    skl = symmetric_kl(a, b)

    # Hyperbolic transformation: 1 / (1 + scale * KL)
    similarity = 1.0 / (1.0 + kl_scale * skl)

    # Uncertainty in the similarity estimate
    # Higher variance in either distribution = less confident about similarity
    avg_var_a = a.variance.mean().item()
    avg_var_b = b.variance.mean().item()
    combined_var = (avg_var_a + avg_var_b) / 2

    # Scale uncertainty to be interpretable (0 = very confident, 1 = very uncertain)
    # Use sigmoid-like scaling based on combined variance
    uncertainty = combined_var / (1.0 + combined_var)

    return similarity, uncertainty


def bayesian_update(
    prior: DistributionalHDV,
    observation: torch.Tensor,
    obs_variance: torch.Tensor,
    current_time: float,
) -> DistributionalHDV:
    """Perform Kalman-style Bayesian update with new observation.

    Updates the distribution based on new evidence. The posterior:
    - Has variance that is always <= prior variance (information gain)
    - Has mean shifted toward observation (weighted by relative confidence)

    Args:
        prior: Current belief distribution.
        observation: New observed HDV.
        obs_variance: Variance/uncertainty of the observation (per-dimension).
        current_time: Current timestamp.

    Returns:
        New DistributionalHDV representing posterior belief.
    """
    # Kalman gain: how much to trust observation vs prior
    # K = prior_var / (prior_var + obs_var)
    # Where K close to 1 means trust observation, K close to 0 means trust prior
    K = prior.variance / (prior.variance + obs_variance)

    # Update mean: move toward observation, weighted by gain
    posterior_mean = prior.mean + K * (observation - prior.mean)

    # Update variance: always decreases (we gained information)
    # posterior_var = (1 - K) * prior_var
    posterior_variance = (1 - K) * prior.variance

    return DistributionalHDV(
        mean=posterior_mean,
        variance=posterior_variance,
        last_accessed=current_time,
        last_updated=current_time,
    )
