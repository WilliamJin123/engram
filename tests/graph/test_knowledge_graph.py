"""Tests for KnowledgeGraph container."""

import torch
import pytest
from engram.graph import Node, Edge, KnowledgeGraph
from engram.hdv import random_distributional


class TestKnowledgeGraph:
    """Verify knowledge graph operations."""

    def test_add_and_retrieve_nodes(self, dim):
        """Can add nodes and retrieve them by ID."""
        graph = KnowledgeGraph()

        hdv = random_distributional(dim, seed=42)
        node = Node(id="dog", hdv=hdv, content="A dog")

        graph.add_node(node)

        retrieved = graph.get_node("dog")
        assert retrieved is not None
        assert retrieved.id == "dog"
        assert retrieved.content == "A dog"

    def test_add_and_retrieve_edges(self, dim):
        """Can add edges and query connections."""
        graph = KnowledgeGraph()

        # Add nodes
        for name in ["dog", "mammal", "animal"]:
            hdv = random_distributional(dim, seed=hash(name) % 1000)
            graph.add_node(Node(id=name, hdv=hdv))

        # Add edges
        graph.add_edge(Edge(source="dog", target="mammal"))
        graph.add_edge(Edge(source="mammal", target="animal"))

        # Query outgoing edges
        dog_targets = graph.get_targets("dog")
        assert "mammal" in dog_targets

        # Query incoming edges
        mammal_sources = graph.get_sources("mammal")
        assert "dog" in mammal_sources

    def test_similarity_search(self, dim):
        """Can find similar nodes by HDV."""
        graph = KnowledgeGraph()

        # Add some nodes
        dog_hdv = random_distributional(dim, seed=1)
        cat_hdv = random_distributional(dim, seed=2)
        car_hdv = random_distributional(dim, seed=100)

        graph.add_node(Node(id="dog", hdv=dog_hdv))
        graph.add_node(Node(id="cat", hdv=cat_hdv))
        graph.add_node(Node(id="car", hdv=car_hdv))

        # Create query similar to dog
        query_hdv = dog_hdv  # Exact match for test simplicity

        results = graph.find_similar(query_hdv, top_k=2)

        assert len(results) == 2
        assert results[0][0] == "dog"  # Most similar
        assert results[0][1] > 0.99  # High similarity

    def test_update_node(self, dim):
        """Can update existing nodes."""
        graph = KnowledgeGraph()

        hdv = random_distributional(dim, seed=42)
        node = Node(id="concept", hdv=hdv, strength=1.0)
        graph.add_node(node)

        # Update the node
        updated = node.reinforce(boost=0.5)
        graph.update_node(updated)

        retrieved = graph.get_node("concept")
        assert retrieved.strength == 1.5

    def test_remove_node_and_edges(self, dim):
        """Removing a node removes associated edges."""
        graph = KnowledgeGraph()

        for name in ["a", "b", "c"]:
            hdv = random_distributional(dim, seed=hash(name) % 1000)
            graph.add_node(Node(id=name, hdv=hdv))

        graph.add_edge(Edge(source="a", target="b"))
        graph.add_edge(Edge(source="b", target="c"))

        graph.remove_node("b")

        assert graph.get_node("b") is None
        assert "b" not in graph.get_targets("a")
        assert "b" not in graph.get_sources("c")
