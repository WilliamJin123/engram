"""Uncertainty calculations for distributional HDVs."""

import torch
from dataclasses import dataclass
from .distributional import DistributionalHDV


@dataclass
class UncertaintyParams:
    """Configuration parameters for uncertainty dynamics.

    Attributes:
        base_drift_rate: Variance increase per time unit (always applies).
        access_drift_rate: Additional variance increase when not accessed.
        access_grace_period: Time before access drift kicks in.
        min_variance: Floor for variance (never perfectly certain).
        max_variance: Ceiling for variance (complete uncertainty).
        human_confirmation_factor: Multiply variance by this on confirmation.
        contradiction_scale: How much contradictions increase variance.
        kl_scale: Scaling factor for similarity calculation.
        base_edge_variance: Variance added by uncertain edges.
    """
    base_drift_rate: float = 0.001
    access_drift_rate: float = 0.01
    access_grace_period: float = 100.0
    min_variance: float = 0.01
    max_variance: float = 2.0
    human_confirmation_factor: float = 0.1
    contradiction_scale: float = 0.5
    kl_scale: float = 1.0
    base_edge_variance: float = 0.1


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


def bayesian_update_with_strength(
    prior: DistributionalHDV,
    observation: torch.Tensor,
    obs_variance: torch.Tensor,
    strength: float,
    current_time: float,
) -> DistributionalHDV:
    """Perform strength-modulated Bayesian update.

    High-strength nodes resist change more than low-strength nodes.
    This implements "certainty resistance" where well-established
    beliefs are harder to shift.

    The strength modulates the effective prior variance:
    - High strength = artificially lower effective variance = trust prior more
    - Low strength = normal behavior = observation has more influence

    Args:
        prior: Current belief distribution.
        observation: New observed HDV.
        obs_variance: Variance/uncertainty of the observation.
        strength: Node strength (importance/confidence, > 0).
        current_time: Current timestamp.

    Returns:
        New DistributionalHDV representing posterior belief.
    """
    # Strength resistance: higher strength = prior is treated as more certain
    # Using log scale to prevent extreme values
    strength_resistance = 1.0 + torch.log1p(torch.tensor(strength)).item()

    # Effective prior variance is reduced by strength (prior appears more certain)
    effective_prior_var = prior.variance / strength_resistance

    # Kalman gain with strength-adjusted variance
    K = effective_prior_var / (effective_prior_var + obs_variance)

    # Update mean: high strength = smaller K = less movement
    posterior_mean = prior.mean + K * (observation - prior.mean)

    # Update variance: high strength = less reduction
    posterior_variance = (1 - K) * prior.variance

    return DistributionalHDV(
        mean=posterior_mean,
        variance=posterior_variance,
        last_accessed=current_time,
        last_updated=current_time,
    )


# Alias for backwards compatibility
bayesian_update_with_mass = bayesian_update_with_strength


def temporal_decay(
    hdv: DistributionalHDV,
    current_time: float,
    params: UncertaintyParams,
) -> DistributionalHDV:
    """Apply temporal variance decay (memories become uncertain over time).

    Two decay components:
    1. Base drift: Always applies, based on time since last update
    2. Access drift: Additional decay when not accessed recently

    Args:
        hdv: The distributional HDV to decay.
        current_time: Current timestamp.
        params: Uncertainty parameters.

    Returns:
        New DistributionalHDV with increased variance.
    """
    # Time since last update (for base drift)
    dt_update = max(0.0, current_time - hdv.last_updated)

    # Time since last access (for access drift)
    dt_access = max(0.0, current_time - hdv.last_accessed)

    # Base drift: always applies
    base_drift = params.base_drift_rate * dt_update

    # Access drift: kicks in after grace period
    if dt_access > params.access_grace_period:
        excess_time = dt_access - params.access_grace_period
        access_drift = params.access_drift_rate * excess_time
    else:
        access_drift = 0.0

    # Apply multiplicatively (variance grows)
    total_drift = 1.0 + base_drift + access_drift
    new_variance = hdv.variance * total_drift

    # Clamp to bounds
    new_variance = torch.clamp(new_variance, min=params.min_variance, max=params.max_variance)

    return DistributionalHDV(
        mean=hdv.mean,  # Mean unchanged
        variance=new_variance,
        last_accessed=current_time,  # Update access time
        last_updated=hdv.last_updated,  # Keep original update time
    )


def human_confirm(
    hdv: DistributionalHDV,
    current_time: float,
    params: UncertaintyParams,
    confirmed_mean: torch.Tensor | None = None,
) -> DistributionalHDV:
    """Apply human confirmation (dramatically reduces uncertainty).

    Human confirmation represents authoritative knowledge that collapses
    uncertainty. This is the strongest form of evidence in the system.

    Args:
        hdv: The distributional HDV to confirm.
        current_time: Current timestamp.
        params: Uncertainty parameters.
        confirmed_mean: Optional explicit mean value (if human corrects it).

    Returns:
        New DistributionalHDV with collapsed variance.
    """
    # Use provided mean or keep existing
    new_mean = confirmed_mean if confirmed_mean is not None else hdv.mean

    # Dramatically reduce variance
    new_variance = hdv.variance * params.human_confirmation_factor

    # Clamp to minimum (never perfectly certain)
    new_variance = torch.clamp(new_variance, min=params.min_variance)

    return DistributionalHDV(
        mean=new_mean,
        variance=new_variance,
        last_accessed=current_time,
        last_updated=current_time,
    )


def handle_contradiction(
    existing: DistributionalHDV,
    contradicting: torch.Tensor,
    current_time: float,
    params: UncertaintyParams,
) -> DistributionalHDV:
    """Handle contradictory evidence by increasing uncertainty.

    When new evidence contradicts existing beliefs, we:
    1. Increase variance in conflicting dimensions (we're less sure now)
    2. Shift mean slightly toward the new evidence (hedge our bets)

    This is different from Bayesian update which assumes evidence is correct.
    Contradiction handling acknowledges we don't know which is right.

    Args:
        existing: Current belief distribution.
        contradicting: The contradicting evidence (HDV).
        current_time: Current timestamp.
        params: Uncertainty parameters.

    Returns:
        New DistributionalHDV with increased uncertainty.
    """
    # Compute conflict strength per dimension
    # High conflict where signs differ and magnitudes are large
    conflict_strength = torch.abs(existing.mean - contradicting)

    # Increase variance proportionally to conflict
    variance_increase = conflict_strength * params.contradiction_scale
    new_variance = existing.variance + variance_increase

    # Clamp to bounds
    new_variance = torch.clamp(new_variance, min=params.min_variance, max=params.max_variance)

    # Shift mean slightly toward contradiction (10% blend)
    # We don't know which is right, so we hedge
    blend_factor = 0.1
    new_mean = (1 - blend_factor) * existing.mean + blend_factor * contradicting

    return DistributionalHDV(
        mean=new_mean,
        variance=new_variance,
        last_accessed=current_time,
        last_updated=current_time,
    )


def bundle_observations(
    observations: list[tuple[torch.Tensor, torch.Tensor]],
    current_time: float,
) -> DistributionalHDV:
    """Bundle multiple observations into a single distribution.

    Uses Bayesian fusion of independent Gaussians:
    - Combined precision = sum of individual precisions
    - Combined mean = precision-weighted average of means

    More observations -> lower variance (more confident).
    This implements "seeing 10 dogs gives more confident 'dog' concept than 1".

    Args:
        observations: List of (mean, variance) tuples.
        current_time: Current timestamp.

    Returns:
        Fused DistributionalHDV.

    Raises:
        ValueError: If observations list is empty or dimensions mismatch.
    """
    if not observations:
        raise ValueError("Cannot bundle empty observation list")

    # Check dimensions match
    dim = observations[0][0].shape[0]
    for i, (mean, variance) in enumerate(observations):
        if mean.shape[0] != dim or variance.shape[0] != dim:
            raise ValueError(
                f"Observation {i} dimension mismatch: expected {dim}, "
                f"got mean={mean.shape[0]}, var={variance.shape[0]}"
            )

    # Compute precisions (1/variance) with numerical stability
    eps = 1e-10
    precisions = [1.0 / (var + eps) for _, var in observations]

    # Total precision = sum of precisions
    total_precision = sum(precisions)

    # Combined variance = 1 / total_precision
    combined_variance = 1.0 / total_precision

    # Combined mean = precision-weighted average
    weighted_sum = sum(p * m for (m, _), p in zip(observations, precisions))
    combined_mean = weighted_sum / total_precision

    return DistributionalHDV(
        mean=combined_mean,
        variance=combined_variance,
        last_accessed=current_time,
        last_updated=current_time,
    )


def propagate_through_edge(
    source: DistributionalHDV,
    edge_hdv: torch.Tensor,
    edge_variance: float,
    params: UncertaintyParams,
) -> DistributionalHDV:
    """Propagate a distributional signal through an edge.

    Signal traveling through an edge:
    1. Mean transforms via binding with edge type HDV
    2. Variance increases based on edge uncertainty

    This implements "propagation compounds uncertainty"—information
    traveling through the graph accumulates variance at each hop.

    Args:
        source: Source distributional HDV.
        edge_hdv: The HDV for the relationship node this edge passes through.
        edge_variance: Uncertainty of this edge (0=certain, higher=uncertain).
        params: Uncertainty parameters.

    Returns:
        Propagated distributional HDV at the target.
    """
    # Mean transforms via bind (element-wise multiply with edge HDV)
    new_mean = source.mean * edge_hdv

    # Edge uncertainty contribution
    edge_uncertainty = edge_variance * params.base_edge_variance

    # Variance compounds: source variance + edge uncertainty
    new_variance = source.variance + edge_uncertainty

    # Clamp to bounds
    new_variance = torch.clamp(new_variance, min=params.min_variance, max=params.max_variance)

    return DistributionalHDV(
        mean=new_mean,
        variance=new_variance,
        last_accessed=source.last_accessed,
        last_updated=source.last_updated,
    )
