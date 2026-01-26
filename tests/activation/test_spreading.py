"""Tests for spreading activation."""

import pytest
from engram import random_distributional, KnowledgeGraph, Node, Edge
from engram.activation import spread_activation, query, multi_query


def test_spread_activation_basic():
    """Test basic activation spreading."""
    graph = KnowledgeGraph()

    # Create chain: A -> B -> C
    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="A", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="B", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="C", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_edge(Edge(source="A", target="B"))
    graph.add_edge(Edge(source="B", target="C"))

    activations = spread_activation(
        graph, "A", current_time=0.0, max_depth=3, mark_accessed=False
    )

    # A should have highest activation (seed)
    assert activations["A"] == 1.0
    # B should have some activation (1 hop)
    assert activations["B"] > 0
    assert activations["B"] < 1.0
    # C should have lower activation (2 hops)
    assert activations["C"] > 0
    assert activations["C"] < activations["B"]


def test_spread_activation_strength_affects_spread():
    """Test that high-strength nodes receive more activation."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="A", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="B_weak", hdv=hdv, strength=0.1, last_accessed=0.0))
    graph.add_node(Node(id="B_strong", hdv=hdv, strength=10.0, last_accessed=0.0))
    graph.add_edge(Edge(source="A", target="B_weak"))
    graph.add_edge(Edge(source="A", target="B_strong"))

    activations = spread_activation(
        graph, "A", current_time=0.0, mark_accessed=False
    )

    # Strong node should receive more activation
    assert activations["B_strong"] > activations["B_weak"]


def test_spread_activation_recency_affects_spread():
    """Test that recently accessed nodes receive more activation."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="A", hdv=hdv, strength=1.0, last_accessed=10.0))
    graph.add_node(Node(id="B_recent", hdv=hdv, strength=1.0, last_accessed=9.9))
    graph.add_node(Node(id="B_old", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_edge(Edge(source="A", target="B_recent"))
    graph.add_edge(Edge(source="A", target="B_old"))

    activations = spread_activation(
        graph, "A", current_time=10.0, recency_tau=1.0, mark_accessed=False
    )

    # Recent node should receive more activation
    assert activations["B_recent"] > activations["B_old"]


def test_spread_activation_depth_limit():
    """Test that activation stops at max_depth."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    for i in range(5):
        graph.add_node(Node(id=f"N{i}", hdv=hdv, strength=5.0, last_accessed=0.0))
    for i in range(4):
        graph.add_edge(Edge(source=f"N{i}", target=f"N{i+1}"))

    activations = spread_activation(
        graph, "N0", current_time=0.0, max_depth=2, mark_accessed=False
    )

    assert "N0" in activations
    assert "N1" in activations
    assert "N2" in activations
    # N3 would be at depth 3 which exceeds max_depth=2


def test_spread_activation_bidirectional():
    """Test that activation spreads in both edge directions."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="A", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="B", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="C", hdv=hdv, strength=1.0, last_accessed=0.0))
    # B -> A (incoming to A)
    graph.add_edge(Edge(source="B", target="A"))
    # A -> C (outgoing from A)
    graph.add_edge(Edge(source="A", target="C"))

    activations = spread_activation(
        graph, "A", current_time=0.0, mark_accessed=False
    )

    # Both B (incoming) and C (outgoing) should be activated
    assert "B" in activations
    assert "C" in activations


def test_spread_activation_mark_accessed():
    """Test that mark_accessed updates node timestamps."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="A", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="B", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_edge(Edge(source="A", target="B"))

    spread_activation(graph, "A", current_time=5.0, mark_accessed=True)

    # Both nodes should have updated timestamps
    assert graph.get_node("A").last_accessed == 5.0
    assert graph.get_node("B").last_accessed == 5.0


def test_query_returns_sorted_results():
    """Test that query returns nodes sorted by activation."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="center", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="strong", hdv=hdv, strength=10.0, last_accessed=0.0))
    graph.add_node(Node(id="weak", hdv=hdv, strength=0.5, last_accessed=0.0))
    graph.add_edge(Edge(source="center", target="strong"))
    graph.add_edge(Edge(source="center", target="weak"))

    results = query(graph, "center", current_time=0.0, top_k=10, mark_accessed=False)

    # First result should be the seed
    assert results[0][0].id == "center"
    # Strong should come before weak
    ids = [r[0].id for r in results]
    assert ids.index("strong") < ids.index("weak")


def test_query_respects_top_k():
    """Test that query limits results to top_k."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="center", hdv=hdv, strength=1.0, last_accessed=0.0))
    for i in range(10):
        graph.add_node(Node(id=f"N{i}", hdv=hdv, strength=1.0, last_accessed=0.0))
        graph.add_edge(Edge(source="center", target=f"N{i}"))

    results = query(graph, "center", current_time=0.0, top_k=5, mark_accessed=False)

    assert len(results) == 5


def test_multi_query_from_multiple_seeds():
    """Test multi_query spreads from multiple starting points."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="A", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="B", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="C", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="middle", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_edge(Edge(source="A", target="middle"))
    graph.add_edge(Edge(source="B", target="middle"))
    graph.add_edge(Edge(source="middle", target="C"))

    results = multi_query(
        graph, ["A", "B"], current_time=0.0, top_k=10, mark_accessed=False
    )

    # Should include both seeds and middle node
    ids = [r[0].id for r in results]
    assert "A" in ids
    assert "B" in ids
    assert "middle" in ids


def test_spread_activation_isolated_node():
    """Test spreading from isolated node returns only that node."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="isolated", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="other", hdv=hdv, strength=1.0, last_accessed=0.0))

    activations = spread_activation(
        graph, "isolated", current_time=0.0, mark_accessed=False
    )

    assert activations == {"isolated": 1.0}


def test_spread_activation_nonexistent_seed():
    """Test spreading from nonexistent seed returns empty dict."""
    graph = KnowledgeGraph()

    activations = spread_activation(
        graph, "nonexistent", current_time=0.0, mark_accessed=False
    )

    assert activations == {}
