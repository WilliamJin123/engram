"""Shared fixtures for quantum substrate tests."""

import pytest
import torch

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
    """Default sparsity (number of active dimensions)."""
    return 50  # k=50 active of dim=1024
