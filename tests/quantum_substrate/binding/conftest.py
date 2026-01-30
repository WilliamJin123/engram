"""Binding-specific fixtures."""

import torch
import pytest


@pytest.fixture
def role_set(rng, dim):
    """Standard set of role vectors for testing."""
    return {
        "agent": torch.randn(dim, dtype=torch.complex64, generator=rng),
        "patient": torch.randn(dim, dtype=torch.complex64, generator=rng),
        "instrument": torch.randn(dim, dtype=torch.complex64, generator=rng),
        "location": torch.randn(dim, dtype=torch.complex64, generator=rng),
        "time": torch.randn(dim, dtype=torch.complex64, generator=rng),
    }
