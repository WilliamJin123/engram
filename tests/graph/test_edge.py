"""Tests for simplified Edge class (unlabeled arrows)."""

import pytest
from engram.graph import Edge


class TestEdge:
    """Test suite for Edge class."""

    def test_create_edge(self):
        """Test basic edge creation with source and target only."""
        edge = Edge(source="node_a", target="node_b")

        assert edge.source == "node_a"
        assert edge.target == "node_b"

    def test_edge_equality(self):
        """Test that edges with same source/target are equal."""
        edge1 = Edge(source="a", target="b")
        edge2 = Edge(source="a", target="b")

        assert edge1 == edge2

    def test_edge_inequality(self):
        """Test that edges with different source/target are not equal."""
        edge1 = Edge(source="a", target="b")
        edge2 = Edge(source="a", target="c")
        edge3 = Edge(source="c", target="b")

        assert edge1 != edge2
        assert edge1 != edge3

    def test_edge_repr(self):
        """Test edge string representation."""
        edge = Edge(source="node1", target="node2")
        repr_str = repr(edge)

        assert "node1" in repr_str
        assert "node2" in repr_str

    def test_edge_hash(self):
        """Test that edges can be used in sets."""
        edge1 = Edge(source="a", target="b")
        edge2 = Edge(source="a", target="b")
        edge3 = Edge(source="b", target="a")

        edge_set = {edge1, edge2, edge3}
        assert len(edge_set) == 2  # edge1 and edge2 are duplicates
