# Distributional HDV Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace point-estimate HDVs with probability distributions (diagonal Gaussians) to enable principled epistemic uncertainty tracking.

**Architecture:** Each node stores a mean vector and per-dimension variance vector, forming a diagonal Gaussian over HDV space. Similarity uses hyperbolic KL divergence (`1/(1+KL)`). Variance increases over time (temporal decay), decreases with observations (Bayesian updates), and collapses with human confirmation. Edges carry a confidence scalar that affects uncertainty propagation.

**Tech Stack:** PyTorch tensors, pytest for TDD, existing engram package structure.

---

## Task 1: DistributionalHDV Data Class

**Files:**
- Create: `src/engram/hdv/distributional.py`
- Test: `tests/hdv/test_distributional.py`

**Step 1: Write the failing test**

Create the test file:

```python
"""Tests for DistributionalHDV data class."""

import torch
import pytest
from engram.hdv.distributional import DistributionalHDV


class TestDistributionalHDVCreation:
    """Test suite for DistributionalHDV instantiation."""

    def test_create_with_mean_and_variance(self, dim):
        """Should create instance with mean and variance tensors."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5
        hdv = DistributionalHDV(mean=mean, variance=variance)
        assert hdv.mean is not None
        assert hdv.variance is not None

    def test_mean_shape_preserved(self, dim):
        """Mean tensor should preserve shape."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5
        hdv = DistributionalHDV(mean=mean, variance=variance)
        assert hdv.mean.shape == (dim,)

    def test_variance_shape_preserved(self, dim):
        """Variance tensor should preserve shape."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5
        hdv = DistributionalHDV(mean=mean, variance=variance)
        assert hdv.variance.shape == (dim,)

    def test_timestamps_default_to_zero(self, dim):
        """Timestamps should default to 0.0 if not provided."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5
        hdv = DistributionalHDV(mean=mean, variance=variance)
        assert hdv.last_accessed == 0.0
        assert hdv.last_updated == 0.0

    def test_timestamps_can_be_set(self, dim):
        """Should accept custom timestamps."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5
        hdv = DistributionalHDV(
            mean=mean, variance=variance,
            last_accessed=100.0, last_updated=50.0
        )
        assert hdv.last_accessed == 100.0
        assert hdv.last_updated == 50.0

    def test_dim_property(self, dim):
        """Should expose dimension as property."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5
        hdv = DistributionalHDV(mean=mean, variance=variance)
        assert hdv.dim == dim


class TestDistributionalHDVValidation:
    """Test variance constraints."""

    def test_variance_must_be_positive(self, dim):
        """Variance values must be positive."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * -0.5  # Invalid
        with pytest.raises(ValueError, match="positive"):
            DistributionalHDV(mean=mean, variance=variance)

    def test_zero_variance_not_allowed(self, dim):
        """Variance of exactly zero is not allowed."""
        mean = torch.randn(dim)
        variance = torch.zeros(dim)  # Invalid
        with pytest.raises(ValueError, match="positive"):
            DistributionalHDV(mean=mean, variance=variance)

    def test_shape_mismatch_raises(self, dim):
        """Mean and variance must have same shape."""
        mean = torch.randn(dim)
        variance = torch.ones(dim // 2)  # Wrong shape
        with pytest.raises(ValueError, match="shape"):
            DistributionalHDV(mean=mean, variance=variance)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hdv/test_distributional.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'engram.hdv.distributional'"

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hdv/test_distributional.py -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add src/engram/hdv/distributional.py tests/hdv/test_distributional.py
git commit -m "$(cat <<'EOF'
feat(hdv): add DistributionalHDV data class

Introduces the core data structure for epistemic uncertainty:
- Diagonal Gaussian with per-dimension variance
- Validation for positive variance and shape matching
- Timestamps for temporal decay tracking

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Random Distributional HDV Factory

**Files:**
- Modify: `src/engram/hdv/distributional.py`
- Modify: `tests/hdv/test_distributional.py`

**Step 1: Write the failing test**

Add to `tests/hdv/test_distributional.py`:

```python
from engram.hdv.distributional import DistributionalHDV, random_distributional


class TestRandomDistributional:
    """Test suite for random_distributional factory."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a DistributionalHDV instance."""
        hdv = random_distributional(dim)
        assert isinstance(hdv, DistributionalHDV)

    def test_correct_dimension(self, dim):
        """Should create HDV with requested dimension."""
        hdv = random_distributional(dim)
        assert hdv.dim == dim

    def test_mean_is_ternary(self, dim):
        """Mean should contain only {-1, 0, +1} values."""
        hdv = random_distributional(dim, seed=42)
        unique_vals = torch.unique(hdv.mean)
        for val in unique_vals:
            assert val.item() in (-1.0, 0.0, 1.0)

    def test_variance_is_uniform_by_default(self, dim):
        """Default variance should be uniform across dimensions."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=42)
        assert torch.allclose(hdv.variance, torch.full((dim,), 0.5))

    def test_custom_initial_variance(self, dim):
        """Should accept custom initial variance."""
        hdv = random_distributional(dim, initial_variance=0.8, seed=42)
        assert torch.allclose(hdv.variance, torch.full((dim,), 0.8))

    def test_seed_reproducibility(self, dim):
        """Same seed should produce same mean."""
        hdv1 = random_distributional(dim, seed=123)
        hdv2 = random_distributional(dim, seed=123)
        assert torch.equal(hdv1.mean, hdv2.mean)

    def test_different_seeds_differ(self, dim):
        """Different seeds should produce different means."""
        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=2)
        assert not torch.equal(hdv1.mean, hdv2.mean)

    def test_sparsity_controls_zeros(self, dim):
        """Sparsity parameter should control fraction of non-zeros in mean."""
        hdv = random_distributional(dim, sparsity=0.3, seed=42)
        non_zero_frac = (hdv.mean != 0).float().mean().item()
        assert abs(non_zero_frac - 0.3) < 0.05  # Within 5%

    def test_timestamps_initialized_with_current_time(self, dim):
        """Should initialize timestamps with provided current_time."""
        hdv = random_distributional(dim, current_time=1000.0, seed=42)
        assert hdv.last_accessed == 1000.0
        assert hdv.last_updated == 1000.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hdv/test_distributional.py::TestRandomDistributional -v`
Expected: FAIL with "cannot import name 'random_distributional'"

**Step 3: Write minimal implementation**

Add to `src/engram/hdv/distributional.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hdv/test_distributional.py::TestRandomDistributional -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add src/engram/hdv/distributional.py tests/hdv/test_distributional.py
git commit -m "$(cat <<'EOF'
feat(hdv): add random_distributional factory function

Creates random DistributionalHDVs with ternary means and
configurable initial variance, sparsity, and timestamps.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: KL Divergence Calculation

**Files:**
- Create: `src/engram/hdv/uncertainty.py`
- Create: `tests/hdv/test_uncertainty.py`

**Step 1: Write the failing test**

```python
"""Tests for uncertainty calculations (KL divergence, etc.)."""

import torch
import pytest
import math
from engram.hdv.distributional import DistributionalHDV, random_distributional
from engram.hdv.uncertainty import kl_divergence, symmetric_kl


class TestKLDivergence:
    """Test suite for KL divergence between diagonal Gaussians."""

    def test_identical_distributions_zero_kl(self, dim):
        """KL divergence of identical distributions is 0."""
        hdv = random_distributional(dim, seed=42)
        kl = kl_divergence(hdv, hdv)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_kl_is_non_negative(self, dim):
        """KL divergence is always >= 0."""
        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=2)
        kl = kl_divergence(hdv1, hdv2)
        assert kl >= 0

    def test_kl_is_asymmetric(self, dim):
        """KL(P||Q) != KL(Q||P) in general."""
        # Create distributions with different variances
        mean1 = torch.randn(dim)
        mean2 = mean1 + 0.5  # Slightly different mean
        var1 = torch.ones(dim) * 0.3
        var2 = torch.ones(dim) * 0.7

        hdv1 = DistributionalHDV(mean=mean1, variance=var1)
        hdv2 = DistributionalHDV(mean=mean2, variance=var2)

        kl_12 = kl_divergence(hdv1, hdv2)
        kl_21 = kl_divergence(hdv2, hdv1)

        assert kl_12 != pytest.approx(kl_21, rel=0.1)

    def test_larger_mean_diff_larger_kl(self, dim):
        """Larger mean difference should give larger KL."""
        base_mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5

        hdv_base = DistributionalHDV(mean=base_mean, variance=variance)
        hdv_close = DistributionalHDV(mean=base_mean + 0.1, variance=variance)
        hdv_far = DistributionalHDV(mean=base_mean + 1.0, variance=variance)

        kl_close = kl_divergence(hdv_base, hdv_close)
        kl_far = kl_divergence(hdv_base, hdv_far)

        assert kl_far > kl_close

    def test_larger_variance_ratio_larger_kl(self, dim):
        """Larger variance ratio should give larger KL."""
        mean = torch.randn(dim)
        var_base = torch.ones(dim) * 0.5

        hdv_base = DistributionalHDV(mean=mean, variance=var_base)
        hdv_similar_var = DistributionalHDV(mean=mean, variance=var_base * 1.1)
        hdv_diff_var = DistributionalHDV(mean=mean, variance=var_base * 3.0)

        kl_similar = kl_divergence(hdv_base, hdv_similar_var)
        kl_diff = kl_divergence(hdv_base, hdv_diff_var)

        assert kl_diff > kl_similar

    def test_kl_formula_validation(self):
        """Validate KL formula against known analytical result.

        For 1D Gaussians: KL(N(μ1,σ1²) || N(μ2,σ2²)) =
        log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        """
        # Simple 1D case
        mean1 = torch.tensor([1.0])
        mean2 = torch.tensor([2.0])
        var1 = torch.tensor([0.5])
        var2 = torch.tensor([1.0])

        hdv1 = DistributionalHDV(mean=mean1, variance=var1)
        hdv2 = DistributionalHDV(mean=mean2, variance=var2)

        # Analytical KL
        sigma1, sigma2 = math.sqrt(0.5), math.sqrt(1.0)
        expected = (
            math.log(sigma2 / sigma1)
            + (var1[0].item() + (mean1[0] - mean2[0])**2) / (2 * var2[0].item())
            - 0.5
        )

        kl = kl_divergence(hdv1, hdv2)
        assert kl == pytest.approx(expected.item(), rel=0.01)


class TestSymmetricKL:
    """Test suite for symmetric (Jeffrey's) KL divergence."""

    def test_symmetric_kl_is_symmetric(self, dim):
        """Symmetric KL should give same result both ways."""
        hdv1 = random_distributional(dim, initial_variance=0.3, seed=1)
        hdv2 = random_distributional(dim, initial_variance=0.7, seed=2)

        skl_12 = symmetric_kl(hdv1, hdv2)
        skl_21 = symmetric_kl(hdv2, hdv1)

        assert skl_12 == pytest.approx(skl_21, rel=1e-6)

    def test_identical_distributions_zero_symmetric_kl(self, dim):
        """Symmetric KL of identical distributions is 0."""
        hdv = random_distributional(dim, seed=42)
        skl = symmetric_kl(hdv, hdv)
        assert skl == pytest.approx(0.0, abs=1e-6)

    def test_symmetric_kl_is_average_of_kl(self, dim):
        """Symmetric KL = (KL(P||Q) + KL(Q||P)) / 2."""
        hdv1 = random_distributional(dim, initial_variance=0.3, seed=1)
        hdv2 = random_distributional(dim, initial_variance=0.7, seed=2)

        kl_12 = kl_divergence(hdv1, hdv2)
        kl_21 = kl_divergence(hdv2, hdv1)
        expected = (kl_12 + kl_21) / 2

        skl = symmetric_kl(hdv1, hdv2)
        assert skl == pytest.approx(expected, rel=1e-6)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hdv/test_uncertainty.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'engram.hdv.uncertainty'"

**Step 3: Write minimal implementation**

```python
"""Uncertainty calculations for distributional HDVs."""

import torch
from .distributional import DistributionalHDV


def kl_divergence(p: DistributionalHDV, q: DistributionalHDV) -> float:
    """Compute KL divergence KL(P || Q) for diagonal Gaussians.

    For diagonal Gaussians:
    KL(P||Q) = 0.5 * sum_i [
        log(σ²_q,i / σ²_p,i)
        + σ²_p,i / σ²_q,i
        + (μ_p,i - μ_q,i)² / σ²_q,i
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hdv/test_uncertainty.py -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add src/engram/hdv/uncertainty.py tests/hdv/test_uncertainty.py
git commit -m "$(cat <<'EOF'
feat(hdv): add KL divergence for diagonal Gaussians

Implements KL divergence and symmetric KL (Jeffrey's divergence)
for comparing distributional HDVs. Foundation for uncertainty-aware
similarity calculations.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Distributional Similarity (Hyperbolic KL)

**Files:**
- Modify: `src/engram/hdv/uncertainty.py`
- Modify: `tests/hdv/test_uncertainty.py`

**Step 1: Write the failing test**

Add to `tests/hdv/test_uncertainty.py`:

```python
from engram.hdv.uncertainty import (
    kl_divergence, symmetric_kl, distributional_similarity
)


class TestDistributionalSimilarity:
    """Test suite for hyperbolic KL-based similarity."""

    def test_identical_distributions_similarity_one(self, dim):
        """Identical distributions should have similarity 1.0."""
        hdv = random_distributional(dim, seed=42)
        sim, uncertainty = distributional_similarity(hdv, hdv)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_returns_tuple_of_floats(self, dim):
        """Should return (similarity, uncertainty) tuple."""
        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=2)
        result = distributional_similarity(hdv1, hdv2)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_similarity_bounded_zero_to_one(self, dim):
        """Similarity should be in range [0, 1]."""
        for seed in range(10):
            hdv1 = random_distributional(dim, seed=seed)
            hdv2 = random_distributional(dim, seed=seed + 100)
            sim, _ = distributional_similarity(hdv1, hdv2)
            assert 0.0 <= sim <= 1.0

    def test_similarity_is_symmetric(self, dim):
        """Similarity should be symmetric."""
        hdv1 = random_distributional(dim, initial_variance=0.3, seed=1)
        hdv2 = random_distributional(dim, initial_variance=0.7, seed=2)

        sim_12, _ = distributional_similarity(hdv1, hdv2)
        sim_21, _ = distributional_similarity(hdv2, hdv1)

        assert sim_12 == pytest.approx(sim_21, rel=1e-6)

    def test_closer_means_higher_similarity(self, dim):
        """Distributions with closer means should have higher similarity."""
        base = random_distributional(dim, seed=42)

        # Create distributions at different distances
        close_mean = base.mean + 0.1 * torch.randn(dim)
        far_mean = base.mean + 2.0 * torch.randn(dim)

        close = DistributionalHDV(mean=close_mean, variance=base.variance.clone())
        far = DistributionalHDV(mean=far_mean, variance=base.variance.clone())

        sim_close, _ = distributional_similarity(base, close)
        sim_far, _ = distributional_similarity(base, far)

        assert sim_close > sim_far

    def test_hyperbolic_formula(self, dim):
        """Should use hyperbolic formula: 1 / (1 + kl_scale * KL)."""
        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=2)

        skl = symmetric_kl(hdv1, hdv2)
        expected_sim = 1.0 / (1.0 + skl)  # default kl_scale=1.0

        sim, _ = distributional_similarity(hdv1, hdv2)
        assert sim == pytest.approx(expected_sim, rel=1e-6)

    def test_custom_kl_scale(self, dim):
        """Should respect custom kl_scale parameter."""
        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=2)

        skl = symmetric_kl(hdv1, hdv2)
        kl_scale = 2.0
        expected_sim = 1.0 / (1.0 + kl_scale * skl)

        sim, _ = distributional_similarity(hdv1, hdv2, kl_scale=kl_scale)
        assert sim == pytest.approx(expected_sim, rel=1e-6)

    def test_uncertainty_higher_with_high_variance(self, dim):
        """Uncertainty should be higher when distributions have high variance."""
        low_var = random_distributional(dim, initial_variance=0.1, seed=1)
        high_var = random_distributional(dim, initial_variance=1.0, seed=1)  # Same mean

        other = random_distributional(dim, initial_variance=0.5, seed=2)

        _, unc_low = distributional_similarity(low_var, other)
        _, unc_high = distributional_similarity(high_var, other)

        assert unc_high > unc_low

    def test_uncertainty_non_negative(self, dim):
        """Uncertainty should always be >= 0."""
        for seed in range(10):
            hdv1 = random_distributional(dim, seed=seed)
            hdv2 = random_distributional(dim, seed=seed + 100)
            _, uncertainty = distributional_similarity(hdv1, hdv2)
            assert uncertainty >= 0.0

    def test_very_different_distributions_low_similarity(self, dim):
        """Very different distributions should have low (but positive) similarity."""
        # Create maximally different distributions
        mean1 = torch.ones(dim)
        mean2 = -torch.ones(dim)
        var = torch.ones(dim) * 0.1  # Low variance = confident they're different

        hdv1 = DistributionalHDV(mean=mean1, variance=var)
        hdv2 = DistributionalHDV(mean=mean2, variance=var)

        sim, _ = distributional_similarity(hdv1, hdv2)

        # Should be low but positive (hyperbolic never reaches 0)
        assert 0.0 < sim < 0.1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hdv/test_uncertainty.py::TestDistributionalSimilarity -v`
Expected: FAIL with "cannot import name 'distributional_similarity'"

**Step 3: Write minimal implementation**

Add to `src/engram/hdv/uncertainty.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hdv/test_uncertainty.py::TestDistributionalSimilarity -v`
Expected: PASS (10 tests)

**Step 5: Commit**

```bash
git add src/engram/hdv/uncertainty.py tests/hdv/test_uncertainty.py
git commit -m "$(cat <<'EOF'
feat(hdv): add distributional_similarity with hyperbolic KL

Implements the core similarity function for distributional HDVs:
- Hyperbolic formula 1/(1+KL) preserves weak similarity
- Returns (similarity, uncertainty) tuple
- Uncertainty derived from combined variance

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Bayesian Update (Kalman-style)

**Files:**
- Modify: `src/engram/hdv/uncertainty.py`
- Modify: `tests/hdv/test_uncertainty.py`

**Step 1: Write the failing test**

Add to `tests/hdv/test_uncertainty.py`:

```python
from engram.hdv.uncertainty import (
    kl_divergence, symmetric_kl, distributional_similarity, bayesian_update
)


class TestBayesianUpdate:
    """Test suite for Kalman-style Bayesian updates."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a new DistributionalHDV."""
        prior = random_distributional(dim, initial_variance=0.5, seed=1)
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        posterior = bayesian_update(prior, observation, obs_variance, current_time=1.0)
        assert isinstance(posterior, DistributionalHDV)

    def test_preserves_dimension(self, dim):
        """Posterior should have same dimension as prior."""
        prior = random_distributional(dim, initial_variance=0.5, seed=1)
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        posterior = bayesian_update(prior, observation, obs_variance, current_time=1.0)
        assert posterior.dim == dim

    def test_variance_always_decreases(self, dim):
        """Posterior variance should be <= prior variance (we gained information)."""
        prior = random_distributional(dim, initial_variance=0.5, seed=1)
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        posterior = bayesian_update(prior, observation, obs_variance, current_time=1.0)

        # Every dimension should have reduced or equal variance
        assert (posterior.variance <= prior.variance + 1e-6).all()

    def test_mean_moves_toward_observation(self, dim):
        """Posterior mean should move toward observation."""
        prior = random_distributional(dim, initial_variance=0.5, seed=1)
        observation = torch.randn(dim) * 2  # Far from origin
        obs_variance = torch.ones(dim) * 0.3

        posterior = bayesian_update(prior, observation, obs_variance, current_time=1.0)

        # Distance to observation should decrease
        dist_before = torch.norm(prior.mean - observation)
        dist_after = torch.norm(posterior.mean - observation)
        assert dist_after < dist_before

    def test_low_obs_variance_bigger_shift(self, dim):
        """Lower observation variance (high confidence) should shift mean more."""
        prior = random_distributional(dim, initial_variance=0.5, seed=1)
        observation = torch.ones(dim)  # Same observation

        # High confidence observation
        posterior_confident = bayesian_update(
            prior, observation, torch.ones(dim) * 0.1, current_time=1.0
        )
        # Low confidence observation
        posterior_uncertain = bayesian_update(
            prior, observation, torch.ones(dim) * 1.0, current_time=1.0
        )

        # Confident observation should move mean more
        shift_confident = torch.norm(posterior_confident.mean - prior.mean)
        shift_uncertain = torch.norm(posterior_uncertain.mean - prior.mean)
        assert shift_confident > shift_uncertain

    def test_low_obs_variance_bigger_variance_reduction(self, dim):
        """Lower observation variance should reduce posterior variance more."""
        prior = random_distributional(dim, initial_variance=0.5, seed=1)
        observation = torch.ones(dim)

        posterior_confident = bayesian_update(
            prior, observation, torch.ones(dim) * 0.1, current_time=1.0
        )
        posterior_uncertain = bayesian_update(
            prior, observation, torch.ones(dim) * 1.0, current_time=1.0
        )

        # Confident observation should reduce variance more
        assert posterior_confident.variance.mean() < posterior_uncertain.variance.mean()

    def test_kalman_gain_formula(self):
        """Validate against analytical Kalman filter formula.

        K = prior_var / (prior_var + obs_var)
        posterior_var = (1 - K) * prior_var
        posterior_mean = prior_mean + K * (obs - prior_mean)
        """
        prior_var = 0.5
        obs_var = 0.3
        prior_mean_val = 1.0
        obs_val = 2.0

        prior = DistributionalHDV(
            mean=torch.tensor([prior_mean_val]),
            variance=torch.tensor([prior_var])
        )
        observation = torch.tensor([obs_val])
        obs_variance = torch.tensor([obs_var])

        posterior = bayesian_update(prior, observation, obs_variance, current_time=1.0)

        # Analytical solution
        K = prior_var / (prior_var + obs_var)
        expected_var = (1 - K) * prior_var
        expected_mean = prior_mean_val + K * (obs_val - prior_mean_val)

        assert posterior.variance[0].item() == pytest.approx(expected_var, rel=0.01)
        assert posterior.mean[0].item() == pytest.approx(expected_mean, rel=0.01)

    def test_multiple_updates_converge(self, dim):
        """Multiple observations should converge mean and reduce variance."""
        prior = random_distributional(dim, initial_variance=1.0, seed=1)
        target = torch.ones(dim) * 0.5  # True value we're observing

        current = prior
        for i in range(10):
            # Observe with some noise
            noisy_obs = target + torch.randn(dim) * 0.1
            current = bayesian_update(
                current, noisy_obs, torch.ones(dim) * 0.2, current_time=float(i)
            )

        # Should be close to target with low variance
        assert torch.norm(current.mean - target) < torch.norm(prior.mean - target)
        assert current.variance.mean() < prior.variance.mean()

    def test_timestamps_updated(self, dim):
        """Should update timestamps to current_time."""
        prior = random_distributional(dim, current_time=0.0, seed=1)
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        posterior = bayesian_update(prior, observation, obs_variance, current_time=100.0)

        assert posterior.last_accessed == 100.0
        assert posterior.last_updated == 100.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hdv/test_uncertainty.py::TestBayesianUpdate -v`
Expected: FAIL with "cannot import name 'bayesian_update'"

**Step 3: Write minimal implementation**

Add to `src/engram/hdv/uncertainty.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hdv/test_uncertainty.py::TestBayesianUpdate -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add src/engram/hdv/uncertainty.py tests/hdv/test_uncertainty.py
git commit -m "$(cat <<'EOF'
feat(hdv): add Kalman-style Bayesian update

Implements bayesian_update for incorporating new evidence:
- Variance always decreases (information gain)
- Mean shifts toward observation weighted by Kalman gain
- Low observation variance = bigger update

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Temporal Variance Decay

**Files:**
- Modify: `src/engram/hdv/uncertainty.py`
- Modify: `tests/hdv/test_uncertainty.py`

**Step 1: Write the failing test**

Add to `tests/hdv/test_uncertainty.py`:

```python
from engram.hdv.uncertainty import (
    kl_divergence, symmetric_kl, distributional_similarity,
    bayesian_update, temporal_decay, UncertaintyParams
)


class TestUncertaintyParams:
    """Test suite for UncertaintyParams configuration."""

    def test_default_values(self):
        """Should have sensible defaults."""
        params = UncertaintyParams()
        assert params.base_drift_rate > 0
        assert params.access_drift_rate > 0
        assert params.access_grace_period > 0
        assert params.min_variance > 0
        assert params.max_variance > params.min_variance

    def test_custom_values(self):
        """Should accept custom values."""
        params = UncertaintyParams(
            base_drift_rate=0.01,
            access_drift_rate=0.05,
            access_grace_period=50.0,
        )
        assert params.base_drift_rate == 0.01
        assert params.access_drift_rate == 0.05
        assert params.access_grace_period == 50.0


class TestTemporalDecay:
    """Test suite for temporal variance decay."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a new DistributionalHDV."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        params = UncertaintyParams()

        decayed = temporal_decay(hdv, current_time=100.0, params=params)
        assert isinstance(decayed, DistributionalHDV)

    def test_variance_increases_over_time(self, dim):
        """Variance should increase as time passes."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        params = UncertaintyParams()

        decayed = temporal_decay(hdv, current_time=1000.0, params=params)

        assert decayed.variance.mean() > hdv.variance.mean()

    def test_base_drift_always_applies(self, dim):
        """Base drift should increase variance even if recently accessed."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        # Set last_accessed to current time (just accessed)
        hdv.last_accessed = 100.0
        params = UncertaintyParams(base_drift_rate=0.01, access_drift_rate=0.05)

        decayed = temporal_decay(hdv, current_time=100.0, params=params)

        # Should still have some increase from base drift
        # (based on time since last_updated, not last_accessed)
        assert decayed.variance.mean() >= hdv.variance.mean()

    def test_access_drift_after_grace_period(self, dim):
        """Additional drift should apply after access grace period."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        params = UncertaintyParams(
            base_drift_rate=0.001,
            access_drift_rate=0.01,
            access_grace_period=50.0
        )

        # Within grace period
        decayed_early = temporal_decay(hdv, current_time=30.0, params=params)

        # After grace period
        decayed_late = temporal_decay(hdv, current_time=200.0, params=params)

        # Late should have more variance increase per unit time
        early_increase = decayed_early.variance.mean() - hdv.variance.mean()
        late_increase = decayed_late.variance.mean() - hdv.variance.mean()

        # Late has much more time AND extra access drift
        assert late_increase > early_increase * 3  # More than just proportional

    def test_variance_capped_at_max(self, dim):
        """Variance should not exceed max_variance."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        params = UncertaintyParams(
            base_drift_rate=0.1,  # Very high drift
            max_variance=1.0
        )

        # Very long time
        decayed = temporal_decay(hdv, current_time=10000.0, params=params)

        assert (decayed.variance <= params.max_variance + 1e-6).all()

    def test_variance_clamped_at_min(self, dim):
        """Variance should not go below min_variance."""
        hdv = random_distributional(dim, initial_variance=0.01, current_time=0.0, seed=1)
        params = UncertaintyParams(min_variance=0.05)

        decayed = temporal_decay(hdv, current_time=0.0, params=params)  # No time passed

        assert (decayed.variance >= params.min_variance - 1e-6).all()

    def test_mean_unchanged(self, dim):
        """Mean should not change during temporal decay."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        params = UncertaintyParams()

        decayed = temporal_decay(hdv, current_time=1000.0, params=params)

        assert torch.equal(decayed.mean, hdv.mean)

    def test_timestamps_updated(self, dim):
        """last_accessed should update to current_time."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        params = UncertaintyParams()

        decayed = temporal_decay(hdv, current_time=500.0, params=params)

        assert decayed.last_accessed == 500.0
        # last_updated stays the same (no new evidence)
        assert decayed.last_updated == hdv.last_updated

    def test_no_time_passed_no_change(self, dim):
        """If no time has passed, variance should stay the same."""
        hdv = random_distributional(dim, initial_variance=0.3, current_time=100.0, seed=1)
        params = UncertaintyParams()

        decayed = temporal_decay(hdv, current_time=100.0, params=params)

        assert torch.allclose(decayed.variance, hdv.variance)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hdv/test_uncertainty.py::TestUncertaintyParams -v`
Run: `pytest tests/hdv/test_uncertainty.py::TestTemporalDecay -v`
Expected: FAIL with "cannot import name 'temporal_decay'"

**Step 3: Write minimal implementation**

Add to `src/engram/hdv/uncertainty.py`:

```python
from dataclasses import dataclass


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
    human_confirmation_factor: float = 0.01
    contradiction_scale: float = 0.5
    kl_scale: float = 1.0
    base_edge_variance: float = 0.1


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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hdv/test_uncertainty.py::TestUncertaintyParams -v`
Run: `pytest tests/hdv/test_uncertainty.py::TestTemporalDecay -v`
Expected: PASS (11 tests total)

**Step 5: Commit**

```bash
git add src/engram/hdv/uncertainty.py tests/hdv/test_uncertainty.py
git commit -m "$(cat <<'EOF'
feat(hdv): add temporal variance decay

Implements memory uncertainty growth over time:
- Base drift always applies (slow)
- Access drift accelerates when neglected
- Variance clamped to min/max bounds
- UncertaintyParams for configuration

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Human Confirmation

**Files:**
- Modify: `src/engram/hdv/uncertainty.py`
- Modify: `tests/hdv/test_uncertainty.py`

**Step 1: Write the failing test**

Add to `tests/hdv/test_uncertainty.py`:

```python
from engram.hdv.uncertainty import (
    kl_divergence, symmetric_kl, distributional_similarity,
    bayesian_update, temporal_decay, UncertaintyParams, human_confirm
)


class TestHumanConfirm:
    """Test suite for human confirmation (variance collapse)."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a new DistributionalHDV."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=1)
        params = UncertaintyParams()

        confirmed = human_confirm(hdv, current_time=1.0, params=params)
        assert isinstance(confirmed, DistributionalHDV)

    def test_variance_dramatically_reduced(self, dim):
        """Variance should be reduced by human_confirmation_factor."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=1)
        params = UncertaintyParams(human_confirmation_factor=0.01)

        confirmed = human_confirm(hdv, current_time=1.0, params=params)

        expected_var = hdv.variance * 0.01
        assert torch.allclose(confirmed.variance, expected_var.clamp(min=params.min_variance))

    def test_variance_respects_minimum(self, dim):
        """Variance should not go below min_variance."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=1)
        params = UncertaintyParams(
            human_confirmation_factor=0.001,  # Would push below min
            min_variance=0.02
        )

        confirmed = human_confirm(hdv, current_time=1.0, params=params)

        assert (confirmed.variance >= params.min_variance - 1e-6).all()

    def test_mean_unchanged_without_explicit(self, dim):
        """Mean should stay same if no confirmed_mean provided."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=1)
        params = UncertaintyParams()

        confirmed = human_confirm(hdv, current_time=1.0, params=params)

        assert torch.equal(confirmed.mean, hdv.mean)

    def test_mean_updated_with_explicit(self, dim):
        """Mean should change to confirmed_mean if provided."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=1)
        params = UncertaintyParams()
        new_mean = torch.randn(dim)

        confirmed = human_confirm(
            hdv, current_time=1.0, params=params, confirmed_mean=new_mean
        )

        assert torch.equal(confirmed.mean, new_mean)

    def test_timestamps_updated(self, dim):
        """Should update both timestamps."""
        hdv = random_distributional(dim, initial_variance=0.5, current_time=0.0, seed=1)
        params = UncertaintyParams()

        confirmed = human_confirm(hdv, current_time=100.0, params=params)

        assert confirmed.last_accessed == 100.0
        assert confirmed.last_updated == 100.0

    def test_high_variance_becomes_low(self, dim):
        """Even very uncertain concepts become confident after confirmation."""
        hdv = random_distributional(dim, initial_variance=1.5, seed=1)  # High uncertainty
        params = UncertaintyParams(human_confirmation_factor=0.01)

        confirmed = human_confirm(hdv, current_time=1.0, params=params)

        # Should be dramatically lower
        assert confirmed.variance.mean() < hdv.variance.mean() * 0.1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hdv/test_uncertainty.py::TestHumanConfirm -v`
Expected: FAIL with "cannot import name 'human_confirm'"

**Step 3: Write minimal implementation**

Add to `src/engram/hdv/uncertainty.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hdv/test_uncertainty.py::TestHumanConfirm -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/engram/hdv/uncertainty.py tests/hdv/test_uncertainty.py
git commit -m "$(cat <<'EOF'
feat(hdv): add human_confirm for variance collapse

Human confirmation dramatically reduces uncertainty:
- Variance multiplied by human_confirmation_factor (default 0.01)
- Optional explicit mean correction
- Represents authoritative knowledge

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Contradiction Handling

**Files:**
- Modify: `src/engram/hdv/uncertainty.py`
- Modify: `tests/hdv/test_uncertainty.py`

**Step 1: Write the failing test**

Add to `tests/hdv/test_uncertainty.py`:

```python
from engram.hdv.uncertainty import (
    kl_divergence, symmetric_kl, distributional_similarity,
    bayesian_update, temporal_decay, UncertaintyParams,
    human_confirm, handle_contradiction
)


class TestHandleContradiction:
    """Test suite for contradiction handling."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a new DistributionalHDV."""
        existing = random_distributional(dim, initial_variance=0.3, seed=1)
        contradicting = random_distributional(dim, seed=2)
        params = UncertaintyParams()

        result = handle_contradiction(
            existing, contradicting.mean, current_time=1.0, params=params
        )
        assert isinstance(result, DistributionalHDV)

    def test_variance_increases_on_conflict(self, dim):
        """Variance should increase in dimensions where values conflict."""
        # Create clearly opposing values
        existing = DistributionalHDV(
            mean=torch.ones(dim),
            variance=torch.ones(dim) * 0.3
        )
        contradicting = -torch.ones(dim)  # Opposite sign everywhere
        params = UncertaintyParams(contradiction_scale=0.5)

        result = handle_contradiction(
            existing, contradicting, current_time=1.0, params=params
        )

        # Variance should increase
        assert result.variance.mean() > existing.variance.mean()

    def test_no_conflict_minimal_change(self, dim):
        """Similar values should not increase variance much."""
        existing = random_distributional(dim, initial_variance=0.3, seed=1)
        similar = existing.mean + torch.randn(dim) * 0.1  # Very similar
        params = UncertaintyParams(contradiction_scale=0.5)

        result = handle_contradiction(
            existing, similar, current_time=1.0, params=params
        )

        # Variance should increase only slightly
        var_increase = result.variance.mean() - existing.variance.mean()
        assert var_increase < 0.1

    def test_mean_moves_slightly_toward_contradiction(self, dim):
        """Mean should shift slightly toward contradicting value."""
        existing = DistributionalHDV(
            mean=torch.ones(dim),
            variance=torch.ones(dim) * 0.3
        )
        contradicting = torch.zeros(dim)  # Different
        params = UncertaintyParams()

        result = handle_contradiction(
            existing, contradicting, current_time=1.0, params=params
        )

        # Should be closer to contradicting than before, but not by much
        dist_before = torch.norm(existing.mean - contradicting)
        dist_after = torch.norm(result.mean - contradicting)
        assert dist_after < dist_before
        # But shouldn't move all the way (still closer to original)
        assert torch.norm(result.mean - existing.mean) < dist_before * 0.3

    def test_variance_respects_maximum(self, dim):
        """Variance should not exceed max_variance."""
        existing = random_distributional(dim, initial_variance=1.8, seed=1)
        contradicting = -existing.mean  # Maximum conflict
        params = UncertaintyParams(
            contradiction_scale=1.0,
            max_variance=2.0
        )

        result = handle_contradiction(
            existing, contradicting, current_time=1.0, params=params
        )

        assert (result.variance <= params.max_variance + 1e-6).all()

    def test_conflict_strength_affects_variance_increase(self, dim):
        """Larger conflicts should increase variance more."""
        existing = DistributionalHDV(
            mean=torch.ones(dim),
            variance=torch.ones(dim) * 0.3
        )
        params = UncertaintyParams(contradiction_scale=0.5)

        # Small conflict
        small_conflict = existing.mean * 0.8  # 20% different
        result_small = handle_contradiction(
            existing, small_conflict, current_time=1.0, params=params
        )

        # Large conflict
        large_conflict = -existing.mean  # Opposite
        result_large = handle_contradiction(
            existing, large_conflict, current_time=1.0, params=params
        )

        assert result_large.variance.mean() > result_small.variance.mean()

    def test_timestamps_updated(self, dim):
        """Should update timestamps."""
        existing = random_distributional(dim, initial_variance=0.3, current_time=0.0, seed=1)
        contradicting = random_distributional(dim, seed=2)
        params = UncertaintyParams()

        result = handle_contradiction(
            existing, contradicting.mean, current_time=100.0, params=params
        )

        assert result.last_accessed == 100.0
        assert result.last_updated == 100.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hdv/test_uncertainty.py::TestHandleContradiction -v`
Expected: FAIL with "cannot import name 'handle_contradiction'"

**Step 3: Write minimal implementation**

Add to `src/engram/hdv/uncertainty.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hdv/test_uncertainty.py::TestHandleContradiction -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/engram/hdv/uncertainty.py tests/hdv/test_uncertainty.py
git commit -m "$(cat <<'EOF'
feat(hdv): add handle_contradiction for conflicting evidence

Contradictions increase uncertainty instead of overwriting:
- Variance increases proportionally to conflict strength
- Mean shifts slightly toward contradiction (hedging)
- Different from Bayesian update (doesn't assume evidence is right)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Bundle Multiple Observations

**Files:**
- Modify: `src/engram/hdv/uncertainty.py`
- Modify: `tests/hdv/test_uncertainty.py`

**Step 1: Write the failing test**

Add to `tests/hdv/test_uncertainty.py`:

```python
from engram.hdv.uncertainty import (
    kl_divergence, symmetric_kl, distributional_similarity,
    bayesian_update, temporal_decay, UncertaintyParams,
    human_confirm, handle_contradiction, bundle_observations
)


class TestBundleObservations:
    """Test suite for bundling multiple observations."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a DistributionalHDV."""
        observations = [
            (torch.randn(dim), torch.ones(dim) * 0.3)
            for _ in range(5)
        ]

        result = bundle_observations(observations, current_time=1.0)
        assert isinstance(result, DistributionalHDV)

    def test_single_observation_same_as_input(self, dim):
        """Single observation should produce same mean and variance."""
        mean = torch.randn(dim)
        variance = torch.ones(dim) * 0.5

        result = bundle_observations([(mean, variance)], current_time=1.0)

        assert torch.allclose(result.mean, mean)
        assert torch.allclose(result.variance, variance)

    def test_more_observations_lower_variance(self, dim):
        """More observations should reduce variance."""
        # All observe same underlying value with noise
        true_value = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.5

        few_obs = [
            (true_value + torch.randn(dim) * 0.1, obs_variance)
            for _ in range(3)
        ]
        many_obs = [
            (true_value + torch.randn(dim) * 0.1, obs_variance)
            for _ in range(30)
        ]

        result_few = bundle_observations(few_obs, current_time=1.0)
        result_many = bundle_observations(many_obs, current_time=1.0)

        # Many observations should have lower variance
        assert result_many.variance.mean() < result_few.variance.mean()

    def test_variance_inversely_proportional_to_count(self, dim):
        """Variance should scale roughly as 1/n for identical observations."""
        mean = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.5

        # 1 observation
        result_1 = bundle_observations([(mean, obs_variance)], current_time=1.0)

        # 4 observations (should have ~1/4 variance)
        result_4 = bundle_observations(
            [(mean, obs_variance) for _ in range(4)],
            current_time=1.0
        )

        ratio = result_1.variance.mean() / result_4.variance.mean()
        # Should be approximately 4 (with some tolerance for numerical issues)
        assert 3.0 < ratio < 5.0

    def test_mean_is_precision_weighted_average(self, dim):
        """Mean should be weighted by precision (1/variance)."""
        # Two observations: one confident, one uncertain
        confident_mean = torch.ones(dim)
        confident_var = torch.ones(dim) * 0.1  # High precision

        uncertain_mean = torch.zeros(dim)
        uncertain_var = torch.ones(dim) * 1.0  # Low precision

        result = bundle_observations([
            (confident_mean, confident_var),
            (uncertain_mean, uncertain_var),
        ], current_time=1.0)

        # Result should be closer to confident observation
        dist_to_confident = torch.norm(result.mean - confident_mean)
        dist_to_uncertain = torch.norm(result.mean - uncertain_mean)
        assert dist_to_confident < dist_to_uncertain

    def test_empty_raises_error(self):
        """Should raise error for empty observation list."""
        with pytest.raises(ValueError, match="empty"):
            bundle_observations([], current_time=1.0)

    def test_timestamps_set(self, dim):
        """Should set timestamps to current_time."""
        observations = [(torch.randn(dim), torch.ones(dim) * 0.5)]

        result = bundle_observations(observations, current_time=123.0)

        assert result.last_accessed == 123.0
        assert result.last_updated == 123.0

    def test_dimension_mismatch_raises(self, dim):
        """Should raise error if observations have different dimensions."""
        observations = [
            (torch.randn(dim), torch.ones(dim) * 0.5),
            (torch.randn(dim // 2), torch.ones(dim // 2) * 0.5),  # Wrong dim
        ]

        with pytest.raises(ValueError, match="dimension"):
            bundle_observations(observations, current_time=1.0)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hdv/test_uncertainty.py::TestBundleObservations -v`
Expected: FAIL with "cannot import name 'bundle_observations'"

**Step 3: Write minimal implementation**

Add to `src/engram/hdv/uncertainty.py`:

```python
def bundle_observations(
    observations: list[tuple[torch.Tensor, torch.Tensor]],
    current_time: float,
) -> DistributionalHDV:
    """Bundle multiple observations into a single distribution.

    Uses Bayesian fusion of independent Gaussians:
    - Combined precision = sum of individual precisions
    - Combined mean = precision-weighted average of means

    More observations → lower variance (more confident).
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hdv/test_uncertainty.py::TestBundleObservations -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add src/engram/hdv/uncertainty.py tests/hdv/test_uncertainty.py
git commit -m "$(cat <<'EOF'
feat(hdv): add bundle_observations for Bayesian fusion

Implements precision-weighted bundling of multiple observations:
- More observations → lower variance
- Confident observations weighted more in mean
- Variance scales as ~1/n for identical observations

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Distributional Bind Operation

**Files:**
- Create: `src/engram/hdv/operations.py` (replace existing)
- Create: `tests/hdv/test_operations_distributional.py`

**Step 1: Write the failing test**

```python
"""Tests for distributional HDV bind/unbind operations."""

import torch
import pytest
from engram.hdv.distributional import DistributionalHDV, random_distributional
from engram.hdv.operations import distributional_bind, distributional_unbind


class TestDistributionalBind:
    """Test suite for distributional bind operation."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a DistributionalHDV."""
        a = random_distributional(dim, seed=1)
        b = random_distributional(dim, seed=2)

        result = distributional_bind(a, b)
        assert isinstance(result, DistributionalHDV)

    def test_mean_is_element_wise_product(self, dim):
        """Mean should be element-wise product of input means."""
        a = random_distributional(dim, seed=1)
        b = random_distributional(dim, seed=2)

        result = distributional_bind(a, b)

        expected_mean = a.mean * b.mean
        assert torch.allclose(result.mean, expected_mean)

    def test_variance_increases(self, dim):
        """Variance should increase when binding (uncertainty compounds)."""
        a = random_distributional(dim, initial_variance=0.3, seed=1)
        b = random_distributional(dim, initial_variance=0.3, seed=2)

        result = distributional_bind(a, b)

        # Result variance should be larger than either input
        assert result.variance.mean() > a.variance.mean()
        assert result.variance.mean() > b.variance.mean()

    def test_commutative(self, dim):
        """Bind should be commutative: bind(a,b) == bind(b,a)."""
        a = random_distributional(dim, seed=1)
        b = random_distributional(dim, seed=2)

        result_ab = distributional_bind(a, b)
        result_ba = distributional_bind(b, a)

        assert torch.allclose(result_ab.mean, result_ba.mean)
        assert torch.allclose(result_ab.variance, result_ba.variance)

    def test_low_variance_inputs_low_variance_output(self, dim):
        """Low variance inputs should produce relatively low variance output."""
        a = random_distributional(dim, initial_variance=0.05, seed=1)
        b = random_distributional(dim, initial_variance=0.05, seed=2)

        result = distributional_bind(a, b)

        # Should still be relatively low (not exploding)
        assert result.variance.mean() < 0.5

    def test_high_variance_inputs_high_variance_output(self, dim):
        """High variance inputs should produce high variance output."""
        a = random_distributional(dim, initial_variance=0.8, seed=1)
        b = random_distributional(dim, initial_variance=0.8, seed=2)

        result = distributional_bind(a, b)

        # Should be higher than inputs
        assert result.variance.mean() > 0.8


class TestDistributionalUnbind:
    """Test suite for distributional unbind operation."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a DistributionalHDV."""
        a = random_distributional(dim, seed=1)
        b = random_distributional(dim, seed=2)
        bound = distributional_bind(a, b)

        result = distributional_unbind(bound, a)
        assert isinstance(result, DistributionalHDV)

    def test_unbind_same_as_bind(self, dim):
        """For ternary means, unbind is same operation as bind."""
        a = random_distributional(dim, seed=1)
        b = random_distributional(dim, seed=2)
        bound = distributional_bind(a, b)

        unbound = distributional_unbind(bound, a)
        rebound = distributional_bind(bound, a)

        assert torch.allclose(unbound.mean, rebound.mean)

    def test_variance_increases_on_unbind(self, dim):
        """Variance should increase when unbinding (more uncertainty)."""
        a = random_distributional(dim, initial_variance=0.3, seed=1)
        b = random_distributional(dim, initial_variance=0.3, seed=2)
        bound = distributional_bind(a, b)

        unbound = distributional_unbind(bound, a)

        # Unbinding adds uncertainty
        assert unbound.variance.mean() > bound.variance.mean()


class TestBindUnbindRoundtrip:
    """Integration tests for bind/unbind recovery."""

    def test_unbind_recovers_similar_mean(self, dim):
        """unbind(bind(a,b), a) should recover mean similar to b."""
        a = random_distributional(dim, initial_variance=0.2, sparsity=0.8, seed=1)
        b = random_distributional(dim, initial_variance=0.2, sparsity=0.8, seed=2)

        bound = distributional_bind(a, b)
        recovered = distributional_unbind(bound, a)

        # Check similarity of means (allowing for sparsity losses)
        from engram.hdv.uncertainty import distributional_similarity
        sim, _ = distributional_similarity(recovered, b)
        assert sim > 0.3  # Should have meaningful similarity

    def test_wrong_key_low_similarity(self, dim):
        """Unbinding with wrong key should give low similarity."""
        a = random_distributional(dim, initial_variance=0.2, sparsity=0.8, seed=1)
        b = random_distributional(dim, initial_variance=0.2, sparsity=0.8, seed=2)
        wrong_key = random_distributional(dim, initial_variance=0.2, sparsity=0.8, seed=3)

        bound = distributional_bind(a, b)
        recovered = distributional_unbind(bound, wrong_key)

        from engram.hdv.uncertainty import distributional_similarity
        sim, _ = distributional_similarity(recovered, b)
        assert sim < 0.3  # Should be low
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hdv/test_operations_distributional.py -v`
Expected: FAIL with "cannot import name 'distributional_bind'"

**Step 3: Write minimal implementation**

Replace `src/engram/hdv/operations.py` with:

```python
"""HDV operations for distributional vectors.

This module provides the core algebraic operations for distributional HDVs
(diagonal Gaussians over HDV space).
"""

import torch
from .distributional import DistributionalHDV


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
        Bound distributional HDV.
    """
    # Mean: element-wise product
    new_mean = a.mean * b.mean

    # Variance: product of random variables formula
    # Var(XY) ≈ μ_a² * σ_b² + μ_b² * σ_a² + σ_a² * σ_b²
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


def distributional_unbind(
    bound: DistributionalHDV,
    key: DistributionalHDV
) -> DistributionalHDV:
    """Unbind a distributional HDV with a key.

    For ternary-mean vectors, unbinding is the same operation as binding
    (element-wise multiplication). The variance increases because we're
    adding another source of uncertainty.

    Args:
        bound: The bound distributional HDV.
        key: The key to unbind with.

    Returns:
        Unbound distributional HDV.
    """
    # For ternary vectors, unbind is same as bind
    return distributional_bind(bound, key)


# Legacy functions for backward compatibility during transition
# These operate on plain tensors (point estimates)

def random_ternary(
    dim: int, sparsity: float = 0.5, seed: int | None = None
) -> torch.Tensor:
    """Generate a random ternary vector with values in {-1, 0, +1}.

    DEPRECATED: Use random_distributional() for uncertainty-aware operations.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    mask = torch.rand(dim, generator=gen) < sparsity
    signs = 2 * torch.randint(0, 2, (dim,), generator=gen, dtype=torch.float32) - 1
    return signs * mask.float()


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bind two point-estimate vectors.

    DEPRECATED: Use distributional_bind() for uncertainty-aware operations.
    """
    return a * b


def unbind(bound: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Unbind a point-estimate vector.

    DEPRECATED: Use distributional_unbind() for uncertainty-aware operations.
    """
    return bound * a


def bundle(items: list[torch.Tensor]) -> torch.Tensor:
    """Bundle point-estimate vectors.

    DEPRECATED: Use bundle_observations() for uncertainty-aware operations.
    """
    if len(items) == 0:
        raise ValueError("Cannot bundle empty list")
    summed = torch.stack(items).sum(dim=0)
    return normalize(summed)


def similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity for point-estimate vectors.

    DEPRECATED: Use distributional_similarity() for uncertainty-aware operations.
    """
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return (torch.dot(a, b) / (norm_a * norm_b)).item()


def normalize(v: torch.Tensor) -> torch.Tensor:
    """Normalize a vector to unit length."""
    norm = torch.norm(v)
    if norm == 0:
        return torch.zeros_like(v)
    return v / norm
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hdv/test_operations_distributional.py -v`
Expected: PASS (12 tests)

**Step 5: Verify legacy tests still pass**

Run: `pytest tests/hdv/test_bind_unbind.py -v`
Expected: PASS (maintains backward compatibility)

**Step 6: Commit**

```bash
git add src/engram/hdv/operations.py tests/hdv/test_operations_distributional.py
git commit -m "$(cat <<'EOF'
feat(hdv): add distributional bind/unbind operations

Implements bind/unbind for distributional HDVs:
- Mean is element-wise product
- Variance compounds via product formula
- Legacy point-estimate functions preserved with deprecation notes

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Edge with Confidence

**Files:**
- Create: `src/engram/graph/__init__.py`
- Create: `src/engram/graph/edge.py`
- Create: `tests/graph/__init__.py`
- Create: `tests/graph/test_edge.py`

**Step 1: Write the failing test**

```python
"""Tests for Edge with confidence."""

import pytest
from engram.graph.edge import Edge


class TestEdge:
    """Test suite for Edge data class."""

    def test_create_edge(self):
        """Should create edge with required fields."""
        edge = Edge(source="node1", target="node2", edge_type="IS_A")
        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.edge_type == "IS_A"

    def test_default_confidence_is_one(self):
        """Default confidence should be 1.0 (certain)."""
        edge = Edge(source="a", target="b", edge_type="HAS")
        assert edge.confidence == 1.0

    def test_custom_confidence(self):
        """Should accept custom confidence."""
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.7)
        assert edge.confidence == 0.7

    def test_default_weight_is_one(self):
        """Default weight should be 1.0."""
        edge = Edge(source="a", target="b", edge_type="HAS")
        assert edge.weight == 1.0

    def test_custom_weight(self):
        """Should accept custom weight."""
        edge = Edge(source="a", target="b", edge_type="IS_A", weight=0.5)
        assert edge.weight == 0.5

    def test_confidence_bounds_zero_to_one(self):
        """Confidence should be clamped to [0, 1]."""
        edge_low = Edge(source="a", target="b", edge_type="X", confidence=-0.5)
        edge_high = Edge(source="a", target="b", edge_type="X", confidence=1.5)

        assert edge_low.confidence == 0.0
        assert edge_high.confidence == 1.0

    def test_uncertainty_property(self):
        """Uncertainty should be 1 - confidence."""
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.8)
        assert edge.uncertainty == pytest.approx(0.2)

    def test_repr_includes_confidence(self):
        """String representation should include confidence."""
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.7)
        repr_str = repr(edge)
        assert "0.7" in repr_str or "confidence" in repr_str.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/graph/test_edge.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'engram.graph'"

**Step 3: Write minimal implementation**

Create `src/engram/graph/__init__.py`:

```python
"""Graph components for Engram."""

from .edge import Edge

__all__ = ["Edge"]
```

Create `src/engram/graph/edge.py`:

```python
"""Edge with confidence for uncertainty-aware knowledge graphs."""

from dataclasses import dataclass


@dataclass
class Edge:
    """An edge connecting two nodes with confidence.

    Attributes:
        source: ID of the source node.
        target: ID of the target node.
        edge_type: Type of relationship (e.g., "IS_A", "HAS").
        confidence: Certainty of this relationship (0=uncertain, 1=certain).
        weight: Strength/importance of the connection.
    """
    source: str
    target: str
    edge_type: str
    confidence: float = 1.0
    weight: float = 1.0

    def __post_init__(self):
        """Clamp confidence to valid range."""
        self.confidence = max(0.0, min(1.0, self.confidence))

    @property
    def uncertainty(self) -> float:
        """Return uncertainty (1 - confidence)."""
        return 1.0 - self.confidence
```

Create `tests/graph/__init__.py`:

```python
"""Tests for graph components."""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/graph/test_edge.py -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add src/engram/graph/ tests/graph/
git commit -m "$(cat <<'EOF'
feat(graph): add Edge with confidence scalar

Introduces Edge class for uncertainty-aware relationships:
- confidence: how certain the relationship is (0-1)
- uncertainty property: 1 - confidence
- Automatic clamping to valid range

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Propagation with Edge Uncertainty

**Files:**
- Modify: `src/engram/hdv/uncertainty.py`
- Modify: `tests/hdv/test_uncertainty.py`

**Step 1: Write the failing test**

Add to `tests/hdv/test_uncertainty.py`:

```python
from engram.graph.edge import Edge
from engram.hdv.uncertainty import (
    kl_divergence, symmetric_kl, distributional_similarity,
    bayesian_update, temporal_decay, UncertaintyParams,
    human_confirm, handle_contradiction, bundle_observations,
    propagate_through_edge
)


class TestPropagateeThroughEdge:
    """Test suite for propagation with edge uncertainty."""

    def test_returns_distributional_hdv(self, dim):
        """Should return a DistributionalHDV."""
        source = random_distributional(dim, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.8)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams()

        result = propagate_through_edge(source, edge, edge_hdv, params)
        assert isinstance(result, DistributionalHDV)

    def test_mean_transforms_via_bind(self, dim):
        """Mean should be transformed by binding with edge HDV."""
        source = random_distributional(dim, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=1.0)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams()

        result = propagate_through_edge(source, edge, edge_hdv, params)

        expected_mean = source.mean * edge_hdv
        assert torch.allclose(result.mean, expected_mean)

    def test_variance_increases(self, dim):
        """Variance should increase after propagation."""
        source = random_distributional(dim, initial_variance=0.3, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.8)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams()

        result = propagate_through_edge(source, edge, edge_hdv, params)

        assert result.variance.mean() > source.variance.mean()

    def test_low_confidence_edge_more_variance(self, dim):
        """Lower confidence edge should add more variance."""
        source = random_distributional(dim, initial_variance=0.3, seed=1)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams(base_edge_variance=0.2)

        high_conf_edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.9)
        low_conf_edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.3)

        result_high = propagate_through_edge(source, high_conf_edge, edge_hdv, params)
        result_low = propagate_through_edge(source, low_conf_edge, edge_hdv, params)

        assert result_low.variance.mean() > result_high.variance.mean()

    def test_certain_edge_minimal_variance_increase(self, dim):
        """Confidence=1.0 edge should add minimal variance."""
        source = random_distributional(dim, initial_variance=0.3, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=1.0)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams(base_edge_variance=0.2)

        result = propagate_through_edge(source, edge, edge_hdv, params)

        # Variance increase should be minimal (only from the bind operation)
        # Not from edge uncertainty
        var_increase = result.variance.mean() - source.variance.mean()
        # With confidence=1.0, edge_uncertainty contribution is 0
        # So increase is only from the binding operation
        assert var_increase < 0.3  # Reasonable bound

    def test_completely_uncertain_edge_max_variance_increase(self, dim):
        """Confidence=0.0 edge should add maximum edge variance."""
        source = random_distributional(dim, initial_variance=0.3, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.0)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams(base_edge_variance=0.5)

        result = propagate_through_edge(source, edge, edge_hdv, params)

        # Should have added significant variance from edge uncertainty
        assert result.variance.mean() > source.variance.mean() + 0.3

    def test_multi_hop_accumulates_variance(self, dim):
        """Multiple propagation hops should accumulate variance."""
        source = random_distributional(dim, initial_variance=0.2, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.8)
        edge_hdv = torch.randn(dim)
        params = UncertaintyParams(base_edge_variance=0.1)

        # One hop
        hop1 = propagate_through_edge(source, edge, edge_hdv, params)

        # Two hops
        hop2 = propagate_through_edge(hop1, edge, edge_hdv, params)

        # Three hops
        hop3 = propagate_through_edge(hop2, edge, edge_hdv, params)

        # Variance should strictly increase with each hop
        assert hop1.variance.mean() > source.variance.mean()
        assert hop2.variance.mean() > hop1.variance.mean()
        assert hop3.variance.mean() > hop2.variance.mean()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hdv/test_uncertainty.py::TestPropagateeThroughEdge -v`
Expected: FAIL with "cannot import name 'propagate_through_edge'"

**Step 3: Write minimal implementation**

Add to `src/engram/hdv/uncertainty.py` (add import at top):

```python
from ..graph.edge import Edge
```

And add the function:

```python
def propagate_through_edge(
    source: DistributionalHDV,
    edge: Edge,
    edge_hdv: torch.Tensor,
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
        edge: The edge to propagate through.
        edge_hdv: The HDV for this edge type.
        params: Uncertainty parameters.

    Returns:
        Propagated distributional HDV at the target.
    """
    # Mean transforms via bind (element-wise multiply with edge HDV)
    new_mean = source.mean * edge_hdv

    # Edge uncertainty contribution
    # Low confidence = more variance added
    edge_uncertainty = edge.uncertainty * params.base_edge_variance

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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hdv/test_uncertainty.py::TestPropagateeThroughEdge -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/engram/hdv/uncertainty.py tests/hdv/test_uncertainty.py
git commit -m "$(cat <<'EOF'
feat(hdv): add propagate_through_edge with uncertainty compounding

Implements signal propagation through edges:
- Mean transforms via bind with edge HDV
- Variance increases based on edge confidence
- Multi-hop queries naturally accumulate uncertainty

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Update Package Exports

**Files:**
- Modify: `src/engram/hdv/__init__.py`
- Modify: `src/engram/__init__.py`

**Step 1: Update hdv package exports**

Replace `src/engram/hdv/__init__.py`:

```python
"""High-Dimensional Vector (HDV) operations with epistemic uncertainty.

This package provides distributional HDVs (diagonal Gaussians over HDV space)
for principled uncertainty tracking in knowledge graphs.
"""

from .distributional import DistributionalHDV, random_distributional
from .uncertainty import (
    UncertaintyParams,
    kl_divergence,
    symmetric_kl,
    distributional_similarity,
    bayesian_update,
    temporal_decay,
    human_confirm,
    handle_contradiction,
    bundle_observations,
    propagate_through_edge,
)
from .operations import (
    distributional_bind,
    distributional_unbind,
    # Legacy exports (deprecated)
    random_ternary,
    bind,
    unbind,
    bundle,
    similarity,
    normalize,
)

__all__ = [
    # Core types
    "DistributionalHDV",
    "UncertaintyParams",
    # Factory
    "random_distributional",
    # Distributional operations
    "distributional_bind",
    "distributional_unbind",
    "distributional_similarity",
    # Uncertainty management
    "kl_divergence",
    "symmetric_kl",
    "bayesian_update",
    "temporal_decay",
    "human_confirm",
    "handle_contradiction",
    "bundle_observations",
    "propagate_through_edge",
    # Legacy (deprecated)
    "random_ternary",
    "bind",
    "unbind",
    "bundle",
    "similarity",
    "normalize",
]
```

**Step 2: Update main package exports**

Replace `src/engram/__init__.py`:

```python
"""Engram: Self-organizing memory framework with epistemic uncertainty.

A memory system using distributional High-Dimensional Vectors (HDVs)
for principled uncertainty tracking in knowledge graphs.
"""

__version__ = "0.2.0"  # Bump for distributional HDV release

from .hdv import (
    DistributionalHDV,
    UncertaintyParams,
    random_distributional,
    distributional_bind,
    distributional_unbind,
    distributional_similarity,
    bayesian_update,
    temporal_decay,
    human_confirm,
    handle_contradiction,
    bundle_observations,
    propagate_through_edge,
)
from .graph import Edge

__all__ = [
    # Version
    "__version__",
    # Core types
    "DistributionalHDV",
    "UncertaintyParams",
    "Edge",
    # Factory
    "random_distributional",
    # Operations
    "distributional_bind",
    "distributional_unbind",
    "distributional_similarity",
    # Uncertainty management
    "bayesian_update",
    "temporal_decay",
    "human_confirm",
    "handle_contradiction",
    "bundle_observations",
    "propagate_through_edge",
]
```

**Step 3: Run all tests**

Run: `pytest -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/engram/hdv/__init__.py src/engram/__init__.py
git commit -m "$(cat <<'EOF'
feat(engram): update package exports for distributional HDVs

- Export all distributional types and operations at package level
- Bump version to 0.2.0 for distributional HDV release
- Legacy point-estimate functions still available (deprecated)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Integration Tests

**Files:**
- Create: `tests/hdv/test_integration.py`

**Step 1: Write integration tests**

```python
"""Integration tests for the complete distributional HDV system."""

import torch
import pytest
from engram import (
    DistributionalHDV,
    UncertaintyParams,
    Edge,
    random_distributional,
    distributional_bind,
    distributional_unbind,
    distributional_similarity,
    bayesian_update,
    temporal_decay,
    human_confirm,
    handle_contradiction,
    bundle_observations,
    propagate_through_edge,
)


class TestCompleteLifecycle:
    """Test the complete lifecycle from the design doc."""

    def test_lifecycle_example(self, dim):
        """Test the complete lifecycle from creation to confirmation."""
        params = UncertaintyParams()

        # 1. Create node from LLM inference (high initial variance)
        dog = random_distributional(dim, initial_variance=0.6, current_time=0.0, seed=42)
        assert dog.variance.mean() == pytest.approx(0.6)

        # 2. Observe several dogs (variance reduces via updates)
        for i in range(10):
            observation = dog.mean + torch.randn(dim) * 0.1  # Noisy observations
            obs_variance = torch.ones(dim) * 0.3
            dog = bayesian_update(dog, observation, obs_variance, current_time=float(i + 1))

        # Variance should have decreased significantly
        assert dog.variance.mean() < 0.3

        # 3. Time passes without access (variance drifts up)
        dog_decayed = temporal_decay(dog, current_time=500.0, params=params)
        assert dog_decayed.variance.mean() > dog.variance.mean()

        # 4. Contradictory information arrives
        contradicting = -dog.mean  # Opposite values
        dog_conflicted = handle_contradiction(
            dog_decayed, contradicting, current_time=501.0, params=params
        )
        assert dog_conflicted.variance.mean() > dog_decayed.variance.mean()

        # 5. Human confirms correct information
        confirmed_mean = random_distributional(dim, seed=100).mean
        dog_confirmed = human_confirm(
            dog_conflicted, current_time=502.0, params=params, confirmed_mean=confirmed_mean
        )

        # Should be near-certain now
        assert dog_confirmed.variance.mean() < 0.05

    def test_similarity_returns_uncertainty(self, dim):
        """Similarity queries should return uncertainty estimates."""
        dog = random_distributional(dim, initial_variance=0.3, seed=1)
        wolf = random_distributional(dim, initial_variance=0.5, seed=2)

        sim, uncertainty = distributional_similarity(dog, wolf)

        assert 0 < sim < 1
        assert uncertainty > 0
        assert isinstance(sim, float)
        assert isinstance(uncertainty, float)

    def test_bundling_reduces_variance(self, dim):
        """Bundling multiple observations reduces variance."""
        params = UncertaintyParams()

        # Single observation
        single_obs = [(torch.randn(dim), torch.ones(dim) * 0.5)]
        single = bundle_observations(single_obs, current_time=0.0)

        # Ten observations of similar things
        many_obs = [
            (torch.randn(dim) * 0.1, torch.ones(dim) * 0.5)
            for _ in range(10)
        ]
        many = bundle_observations(many_obs, current_time=0.0)

        # Many observations should be more confident
        assert many.variance.mean() < single.variance.mean()

    def test_propagation_compounds_uncertainty(self, dim):
        """Information traveling through graph accumulates variance."""
        params = UncertaintyParams(base_edge_variance=0.1)

        source = random_distributional(dim, initial_variance=0.2, seed=1)
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.8)
        edge_hdv = random_distributional(dim, seed=2).mean

        # Three hops
        hop1 = propagate_through_edge(source, edge, edge_hdv, params)
        hop2 = propagate_through_edge(hop1, edge, edge_hdv, params)
        hop3 = propagate_through_edge(hop2, edge, edge_hdv, params)

        # Variance strictly increases
        assert hop1.variance.mean() > source.variance.mean()
        assert hop2.variance.mean() > hop1.variance.mean()
        assert hop3.variance.mean() > hop2.variance.mean()


class TestInitialVarianceBySource:
    """Test context-dependent initial variance."""

    def test_human_direct_low_variance(self, dim):
        """Human-sourced info should start with low variance."""
        hdv = random_distributional(dim, initial_variance=0.05, seed=1)
        assert hdv.variance.mean() == pytest.approx(0.05)

    def test_llm_inference_high_variance(self, dim):
        """LLM-inferred info should start with high variance."""
        hdv = random_distributional(dim, initial_variance=0.6, seed=1)
        assert hdv.variance.mean() == pytest.approx(0.6)

    def test_propagated_moderate_variance(self, dim):
        """Propagated info should have moderate variance."""
        hdv = random_distributional(dim, initial_variance=0.5, seed=1)
        assert hdv.variance.mean() == pytest.approx(0.5)


class TestEdgeConfidenceEffects:
    """Test that edge confidence affects propagation correctly."""

    def test_high_confidence_edge_preserves_info(self, dim):
        """High confidence edges should preserve information better."""
        params = UncertaintyParams(base_edge_variance=0.3)

        source = random_distributional(dim, initial_variance=0.2, seed=1)
        edge_hdv = random_distributional(dim, seed=2).mean

        high_conf = Edge(source="a", target="b", edge_type="IS_A", confidence=0.95)
        low_conf = Edge(source="a", target="b", edge_type="IS_A", confidence=0.3)

        result_high = propagate_through_edge(source, high_conf, edge_hdv, params)
        result_low = propagate_through_edge(source, low_conf, edge_hdv, params)

        # High confidence should have lower variance increase
        assert result_high.variance.mean() < result_low.variance.mean()

    def test_edge_uncertainty_formula(self, dim):
        """Edge uncertainty contribution should be (1-confidence) * base_edge_variance."""
        params = UncertaintyParams(base_edge_variance=0.5)

        source = random_distributional(dim, initial_variance=0.2, seed=1)
        edge_hdv = torch.ones(dim)  # No binding effect
        edge = Edge(source="a", target="b", edge_type="IS_A", confidence=0.6)

        result = propagate_through_edge(source, edge, edge_hdv, params)

        # Expected edge uncertainty contribution: 0.4 * 0.5 = 0.2
        expected_var = source.variance + 0.2
        assert torch.allclose(result.variance, expected_var, atol=0.01)
```

**Step 2: Run integration tests**

Run: `pytest tests/hdv/test_integration.py -v`
Expected: PASS (all tests)

**Step 3: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/hdv/test_integration.py
git commit -m "$(cat <<'EOF'
test(hdv): add integration tests for distributional HDV system

Comprehensive tests covering:
- Complete lifecycle from creation to confirmation
- Similarity with uncertainty
- Bundling variance reduction
- Propagation uncertainty compounding
- Edge confidence effects

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Final Verification and Documentation

**Step 1: Run complete test suite**

Run: `pytest -v --tb=short`
Expected: All tests PASS (approximately 130+ tests)

**Step 2: Verify imports work correctly**

Run: `python -c "from engram import *; print('All imports successful')"`
Expected: "All imports successful"

**Step 3: Create final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(engram): complete distributional HDV implementation

Epistemic uncertainty tracking for knowledge graphs:

## Core Features
- DistributionalHDV: diagonal Gaussian over HDV space
- Per-dimension variance for feature-level uncertainty
- Hyperbolic KL-based similarity: 1/(1+KL)

## Uncertainty Dynamics
- Bayesian updates: new evidence reduces variance
- Temporal decay: memories become uncertain over time
- Human confirmation: dramatically reduces variance
- Contradiction handling: increases variance instead of overwriting
- Bundle observations: multiple observations reduce variance

## Propagation
- Edge confidence scalar (0-1)
- Uncertainty compounds through graph traversal
- Low confidence edges add more variance

## Breaking Changes
- Point-estimate functions deprecated (still available)
- Version bumped to 0.2.0

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

This plan implements distributional HDVs in 15 tasks with approximately 130 tests:

| Task | Component | Tests |
|------|-----------|-------|
| 1 | DistributionalHDV class | 9 |
| 2 | random_distributional factory | 9 |
| 3 | KL divergence | 9 |
| 4 | Distributional similarity | 10 |
| 5 | Bayesian update | 9 |
| 6 | Temporal decay | 11 |
| 7 | Human confirmation | 7 |
| 8 | Contradiction handling | 7 |
| 9 | Bundle observations | 8 |
| 10 | Distributional bind/unbind | 12 |
| 11 | Edge with confidence | 8 |
| 12 | Propagation through edges | 7 |
| 13 | Package exports | - |
| 14 | Integration tests | 8 |
| 15 | Final verification | - |

**Total: ~104 new tests** (plus existing ~118 tests = ~222 total)
