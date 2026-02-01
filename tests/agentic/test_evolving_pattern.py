# tests/agentic/test_evolving_pattern.py
"""Tests for EvolvingPattern with lineage tracking."""

import pytest
from agentic.evolving_pattern import EvolvingPattern
from agentic.text_encoder import TextEncoder


class TestEvolvingPatternCreation:
    """Test basic pattern creation and properties."""

    def test_create_from_encoded_pattern(self):
        """EvolvingPattern can wrap an EncodedPattern."""
        encoder = TextEncoder(dim=1024, k=50)
        encoded = encoder.encode("hello world")

        pattern = EvolvingPattern.from_encoded(encoded)

        assert pattern.dim == 1024
        assert len(pattern.bits) == 50
        assert pattern.original_bits == frozenset(pattern.bits)
        assert len(pattern.acquired_bits) == 0
        assert pattern.text == "hello world"

    def test_create_from_text(self):
        """Pattern can be created directly from text."""
        pattern = EvolvingPattern.from_text("hello world", dim=1024, k=50)

        assert pattern.dim == 1024
        assert len(pattern.bits) == 50
        assert pattern.original_bits == frozenset(pattern.bits)
        assert len(pattern.acquired_bits) == 0
        assert pattern.text == "hello world"

    def test_deterministic_encoding(self):
        """Same text produces same pattern."""
        p1 = EvolvingPattern.from_text("hello world", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("hello world", dim=1024, k=50)

        assert p1.bits == p2.bits
        assert p1.original_bits == p2.original_bits

    def test_phases_preserved(self):
        """Each bit has an associated phase."""
        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)

        assert len(pattern.phases) == len(pattern.bits)
        for bit in pattern.bits:
            assert bit in pattern.phases


class TestEvolvingPatternDenseConversion:
    """Test conversion to dense tensors for HRR operations."""

    def test_to_dense_uses_original_bits(self):
        """to_dense() uses only original bits for binding integrity."""
        import torch

        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)

        pattern.bits.add(999)
        pattern.acquired_bits.add(999)
        pattern.phases[999] = 0.0

        dense = pattern.to_dense()

        assert dense[999].abs().item() < 1e-6

        for bit in pattern.original_bits:
            assert dense[bit].abs().item() > 0.5

    def test_to_dense_evolved_uses_all_bits(self):
        """to_dense_evolved() uses all bits for retrieval."""
        import torch

        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)

        pattern.bits.add(999)
        pattern.acquired_bits.add(999)
        pattern.phases[999] = 0.0

        dense = pattern.to_dense_evolved()

        assert dense[999].abs().item() > 0.5


class TestEvolvingPatternMetrics:
    """Test metrics for tracking pattern evolution."""

    def test_obesity_metric(self):
        """Obesity is ratio of current to original bits."""
        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)

        assert pattern.obesity == 1.0

        for i in range(10):
            pattern.bits.add(900 + i)
            pattern.acquired_bits.add(900 + i)

        assert pattern.obesity == 60 / 50  # 1.2

    def test_acquired_ratio_metric(self):
        """Acquired ratio is fraction of bits that are acquired."""
        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)

        assert pattern.acquired_ratio == 0.0

        for i in range(10):
            pattern.bits.add(900 + i)
            pattern.acquired_bits.add(900 + i)

        assert pattern.acquired_ratio == 10 / 60
