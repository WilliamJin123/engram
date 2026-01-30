"""Shared fixtures for quantum substrate tests."""

import pytest
import torch

# Use consistent seed for reproducibility
SEED = 42


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return torch.Generator().manual_seed(SEED)


@pytest.fixture
def dim():
    """Default pattern dimensionality."""
    return 1024


@pytest.fixture
def sparsity():
    """Default sparsity (fraction of active dimensions)."""
    return 0.01  # 1% = ~10 active bits in 1024-dim
