"""Tests for Edge class with confidence."""

import pytest
from engram.graph import Edge


class TestEdge:
    """Test suite for Edge class."""

    def test_create_edge(self):
        """Test basic edge creation with required fields."""
        edge = Edge(source="node_a", target="node_b", edge_type="IS_A")

        assert edge.source == "node_a"
        assert edge.target == "node_b"
        assert edge.edge_type == "IS_A"

    def test_default_confidence_is_one(self):
        """Test that default confidence is 1.0 (fully certain)."""
        edge = Edge(source="a", target="b", edge_type="RELATES_TO")

        assert edge.confidence == 1.0

    def test_custom_confidence(self):
        """Test setting custom confidence value."""
        edge = Edge(source="a", target="b", edge_type="MAYBE_IS", confidence=0.7)

        assert edge.confidence == 0.7

    def test_default_weight_is_one(self):
        """Test that default weight is 1.0."""
        edge = Edge(source="a", target="b", edge_type="CONNECTS")

        assert edge.weight == 1.0

    def test_custom_weight(self):
        """Test setting custom weight value."""
        edge = Edge(source="a", target="b", edge_type="STRONG_LINK", weight=5.0)

        assert edge.weight == 5.0

    def test_confidence_bounds_zero_to_one(self):
        """Test that confidence is clamped to [0, 1] range."""
        # Test upper bound clamping
        edge_high = Edge(source="a", target="b", edge_type="TEST", confidence=1.5)
        assert edge_high.confidence == 1.0

        # Test lower bound clamping
        edge_low = Edge(source="a", target="b", edge_type="TEST", confidence=-0.5)
        assert edge_low.confidence == 0.0

        # Test boundary values are preserved
        edge_zero = Edge(source="a", target="b", edge_type="TEST", confidence=0.0)
        assert edge_zero.confidence == 0.0

        edge_one = Edge(source="a", target="b", edge_type="TEST", confidence=1.0)
        assert edge_one.confidence == 1.0

    def test_uncertainty_property(self):
        """Test uncertainty property returns 1 - confidence."""
        edge_certain = Edge(source="a", target="b", edge_type="SURE", confidence=1.0)
        assert edge_certain.uncertainty == 0.0

        edge_uncertain = Edge(source="a", target="b", edge_type="MAYBE", confidence=0.0)
        assert edge_uncertain.uncertainty == 1.0

        edge_partial = Edge(source="a", target="b", edge_type="LIKELY", confidence=0.75)
        assert edge_partial.uncertainty == 0.25

    def test_repr_includes_confidence(self):
        """Test that repr includes confidence value."""
        edge = Edge(source="node1", target="node2", edge_type="HAS", confidence=0.8)
        repr_str = repr(edge)

        assert "confidence=0.8" in repr_str
        assert "node1" in repr_str
        assert "node2" in repr_str
        assert "HAS" in repr_str
