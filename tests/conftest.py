"""Shared pytest fixtures for engram tests."""

import torch
import pytest


@pytest.fixture
def dim():
    """Standard dimension for HDV tests."""
    return 10000


@pytest.fixture
def hyperbolic_dim():
    """Standard dimension for hyperbolic embedding tests."""
    return 50
