"""Tests for Abstractor sleep agent."""

import torch
import pytest
from engram.graph import Node, Edge, KnowledgeGraph
from engram.hdv import random_distributional, distributional_similarity
from engram.sleep import Abstractor


class TestAbstractor:
    """Verify abstractor creates concept nodes from clusters."""

    def test_creates_concept_from_cluster(self, dim):
        """Abstractor creates a concept node from a cluster.

        Scenario: Given a cluster of similar nodes, abstractor creates
        a parent concept and links instances to it.
        """
        graph = KnowledgeGraph()

        # Create similar nodes (a cluster)
        instances = []
        for i, name in enumerate(["dog1", "dog2", "dog3"]):
            hdv = random_distributional(dim, initial_variance=0.5, seed=1)
            graph.add_node(Node(id=name, hdv=hdv, content=f"A {name}"))
            instances.append(name)

        cluster = set(instances)

        # Run abstractor
        abstractor = Abstractor()
        result = abstractor.abstract_cluster(
            graph,
            cluster,
            concept_id="dog_concept",
            concept_content="The concept of a dog"
        )

        # Concept node should exist
        concept = graph.get_node("dog_concept")
        assert concept is not None
        assert concept.content == "The concept of a dog"

        # Concept mean should be close to instance means (they're identical)
        for instance_id in instances:
            instance = graph.get_node(instance_id)
            # Check mean vectors are close (cosine similarity)
            cos_sim = torch.nn.functional.cosine_similarity(
                concept.hdv.mean.unsqueeze(0),
                instance.hdv.mean.unsqueeze(0)
            ).item()
            assert cos_sim > 0.99  # Nearly identical means

        # Edges should link instances to concept
        for instance_id in instances:
            targets = graph.get_targets(instance_id)
            assert "dog_concept" in targets

    def test_concept_starts_with_low_strength(self, dim):
        """Newly created concepts start with low strength."""
        graph = KnowledgeGraph()

        for i in range(3):
            hdv = random_distributional(dim, seed=1)
            graph.add_node(Node(id=f"node_{i}", hdv=hdv))

        abstractor = Abstractor(initial_concept_strength=0.5)
        abstractor.abstract_cluster(
            graph,
            {"node_0", "node_1", "node_2"},
            concept_id="concept"
        )

        concept = graph.get_node("concept")
        assert concept.strength == 0.5

    def test_returns_created_concept_id(self, dim):
        """Abstractor returns info about what it created."""
        graph = KnowledgeGraph()

        for i in range(2):
            hdv = random_distributional(dim, seed=1)
            graph.add_node(Node(id=f"node_{i}", hdv=hdv))

        abstractor = Abstractor()
        result = abstractor.abstract_cluster(
            graph,
            {"node_0", "node_1"},
            concept_id="my_concept"
        )

        assert result["concept_id"] == "my_concept"
        assert result["instance_count"] == 2
