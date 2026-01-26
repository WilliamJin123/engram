"""Scenario-based tests for graph module (Node, Edge)."""

import torch
import pytest
from engram.graph import Node, Edge, KnowledgeGraph
from engram.hdv import random_distributional
from engram.activation import spread_activation, query


class TestGraphScenarios:
    """Scenario tests for graph structures."""

    def test_node_lifecycle(self, dim):
        """Create a node, mutate its state, verify behavior.

        Scenario: A node represents a concept that gets accessed and reinforced
        over time, with strength decay between accesses.
        """
        # Create node with HDV and initial properties
        hdv = random_distributional(dim, seed=42)
        node = Node(id="concept_dog", hdv=hdv, strength=1.0, last_accessed=0.0, content="A dog")

        # Verify initial state
        assert node.id == "concept_dog"
        assert node.strength == 1.0
        assert node.last_accessed == 0.0
        assert node.content == "A dog"
        assert node.hdv is hdv

        # Strength decays over time
        node = node.decay(dt=10.0, decay_rate=0.01)
        assert node.strength < 1.0

        # Access updates timestamp
        node = node.access(current_time=5.0)
        assert node.last_accessed == 5.0

        # Effective strength includes recency boost
        eff_strength = node.get_effective_strength(current_time=5.0)
        assert eff_strength > node.strength  # Recency boost at t=access time

        # Reinforcement adds strength
        original_strength = node.strength
        node = node.reinforce(boost=0.5)
        assert node.strength == original_strength + 0.5

        # Strength is clamped to minimum
        node_min = Node(id="test", hdv=hdv, strength=0.001)
        assert node_min.strength == Node.MIN_STRENGTH

        # Node has meaningful string representation
        assert "concept_dog" in repr(node)

    def test_edge_connections(self, dim):
        """Create nodes and connect them with edges.

        Scenario: Build a simple hierarchy of concepts connected by edges,
        verify edge equality and hashing for use in sets.
        """
        # Create nodes
        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=2)
        hdv3 = random_distributional(dim, seed=3)

        animal = Node(id="animal", hdv=hdv1)
        dog = Node(id="dog", hdv=hdv2)
        cat = Node(id="cat", hdv=hdv3)

        # Create edges (child -> parent for "is-a" relationship)
        edge_dog = Edge(source="dog", target="animal")
        edge_cat = Edge(source="cat", target="animal")

        # Verify edge properties
        assert edge_dog.source == "dog"
        assert edge_dog.target == "animal"

        # Edge equality works correctly
        edge_dog_copy = Edge(source="dog", target="animal")
        assert edge_dog == edge_dog_copy

        different_edge = Edge(source="cat", target="animal")
        assert edge_dog != different_edge

        # Edges can be used in sets (hashable, duplicates removed)
        edge_set = {edge_dog, edge_cat, edge_dog_copy}
        assert len(edge_set) == 2  # dog_copy is duplicate

        # Edges have meaningful string representation
        assert "dog" in repr(edge_dog)
        assert "animal" in repr(edge_dog)

    def test_strength_model_integration(self, dim):
        """Integration test for strength model + spreading activation.

        Scenario: Build a small knowledge graph about dogs, query it
        using spreading activation, and verify recency effects work.
        """
        graph = KnowledgeGraph()

        # Create a small knowledge graph about dogs
        dog_hdv = random_distributional(dim, seed=1)
        mammal_hdv = random_distributional(dim, seed=2)
        animal_hdv = random_distributional(dim, seed=3)
        barks_hdv = random_distributional(dim, seed=4)

        graph.add_node(Node(id="dog", hdv=dog_hdv, strength=5.0, last_accessed=0.0))
        graph.add_node(Node(id="mammal", hdv=mammal_hdv, strength=3.0, last_accessed=0.0))
        graph.add_node(Node(id="animal", hdv=animal_hdv, strength=2.0, last_accessed=0.0))
        graph.add_node(Node(id="barks", hdv=barks_hdv, strength=4.0, last_accessed=0.0))

        graph.add_edge(Edge(source="dog", target="mammal"))
        graph.add_edge(Edge(source="mammal", target="animal"))
        graph.add_edge(Edge(source="dog", target="barks"))

        # Query "what is a dog?" using spreading activation
        results = query(graph, "dog", current_time=0.0, top_k=5, mark_accessed=False)

        # Should return dog and connected concepts
        result_ids = [r[0].id for r in results]
        assert "dog" in result_ids
        assert "mammal" in result_ids or "barks" in result_ids

        # Dog should have highest activation (it's the seed)
        assert results[0][0].id == "dog"
        assert results[0][1] == 1.0

        # Access dog again at t=1 (updates timestamp)
        graph.update_node("dog", last_accessed=1.0)

        # Query again at t=1 - dog should still have high activation
        results2 = query(graph, "dog", current_time=1.0, top_k=5, mark_accessed=False)
        assert results2[0][0].id == "dog"

        # Verify recency boost: recently accessed dog has higher effective strength
        dog = graph.get_node("dog")
        eff_at_1 = dog.get_effective_strength(current_time=1.0)
        eff_at_10 = dog.get_effective_strength(current_time=10.0)
        assert eff_at_1 > eff_at_10  # Recency boost decays over time

        # Decay strength over time
        old_dog = graph.get_node("dog")
        decayed_dog = old_dog.decay(dt=100, decay_rate=0.001)
        assert decayed_dog.strength < old_dog.strength

        # Reinforce dog when successfully used
        reinforced_dog = decayed_dog.reinforce(boost=0.5)
        assert reinforced_dog.strength == decayed_dog.strength + 0.5
