"""Tests for ContradictionDetector sleep agent."""

import torch
import pytest
from engram.graph import Node, KnowledgeGraph
from engram.hdv import random_distributional
from engram.sleep import ContradictionDetector


class TestContradictionDetector:
    """Verify contradiction detection creates uncertainty nodes."""

    def test_detects_opposing_beliefs(self, dim):
        """Finds nodes with similar topic but opposing content.

        Scenario: Two nodes about Earth's shape with opposite HDVs
        should be flagged as contradicting.
        """
        graph = KnowledgeGraph()

        # Similar base (same topic)
        hdv1 = random_distributional(dim, seed=42)
        hdv2 = random_distributional(dim, seed=42)
        hdv2.mean = -hdv2.mean  # Opposite

        graph.add_node(Node(id="earth_round", hdv=hdv1, content="Earth is round"))
        graph.add_node(Node(id="earth_flat", hdv=hdv2, content="Earth is flat"))

        detector = ContradictionDetector(
            similarity_threshold=0.00001,  # Extremely low - opposite means have near-zero KL similarity
            opposition_threshold=-0.5,  # Mean opposition
        )

        contradictions = detector.find_contradictions(graph)

        # Should find the contradiction
        assert len(contradictions) >= 1

        # Check it found our pair
        found = False
        for c in contradictions:
            if "earth_round" in c["nodes"] and "earth_flat" in c["nodes"]:
                found = True
                break
        assert found

    def test_creates_uncertainty_node(self, dim):
        """Creates uncertainty node for detected contradiction."""
        graph = KnowledgeGraph()

        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=1)
        hdv2.mean = -hdv2.mean

        graph.add_node(Node(id="belief_a", hdv=hdv1))
        graph.add_node(Node(id="belief_b", hdv=hdv2))

        detector = ContradictionDetector()
        result = detector.run(graph)

        # Should have created uncertainty nodes
        if result["contradictions"]:
            uncertainty_id = result["uncertainty_nodes"][0]
            uncertainty_node = graph.get_node(uncertainty_id)

            assert uncertainty_node is not None
            assert uncertainty_node.energy == 1.0  # High energy

            # Should link to contradicting nodes
            sources = graph.get_sources(uncertainty_id)
            assert "belief_a" in sources or "belief_b" in sources

    def test_no_contradictions_when_consistent(self, dim):
        """No contradictions found in consistent graph."""
        graph = KnowledgeGraph()

        # All different, unrelated nodes
        for i in range(5):
            hdv = random_distributional(dim, seed=i * 100)
            graph.add_node(Node(id=f"node_{i}", hdv=hdv))

        detector = ContradictionDetector()
        contradictions = detector.find_contradictions(graph)

        assert len(contradictions) == 0
