"""Scenario-based tests for graph module (Node, Edge)."""

import torch
import pytest
from engram.graph import Node, Edge
from engram.hdv import random_distributional


class TestGraphScenarios:
    """Scenario tests for graph structures."""

    def test_node_lifecycle(self, dim):
        """Create a node, mutate its state, verify behavior.

        Scenario: A node represents a concept that gets accessed and reinforced
        over time, with energy decay between accesses.
        """
        # Create node with HDV and initial properties
        hdv = random_distributional(dim, seed=42)
        node = Node(id="concept_dog", hdv=hdv, energy=1.0, mass=1.0, content="A dog")

        # Verify initial state
        assert node.id == "concept_dog"
        assert node.energy == 1.0
        assert node.mass == 1.0
        assert node.content == "A dog"
        assert node.hdv is hdv

        # Energy decays over time without access
        node = node.decay(dt=10.0, energy_decay_rate=0.1, mass_decay_rate=0.01)
        assert node.energy < 1.0
        assert node.mass < 1.0

        # Access restores energy (up to 1.0)
        node = node.access(energy_boost=0.5)
        assert node.energy <= 1.0

        # Reinforcement adds mass
        original_mass = node.mass
        node = node.reinforce(mass_boost=0.5)
        assert node.mass == original_mass + 0.5

        # Energy is clamped to valid range
        node_clamped = Node(id="test", hdv=hdv, energy=5.0)  # Over 1.0
        assert node_clamped.energy == 1.0

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
