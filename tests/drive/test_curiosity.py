"""Tests for Curiosity drive."""

import pytest
from engram.graph import Node, KnowledgeGraph
from engram.hdv import random_distributional
from engram.drive import CuriosityDrive


class TestCuriosityDrive:
    """Verify curiosity drive identifies what needs resolution."""

    def test_surfaces_high_energy_uncertainty(self, dim):
        """High-energy uncertainty nodes should surface.

        Scenario: Uncertainty node with high energy should be
        returned as needing resolution.
        """
        graph = KnowledgeGraph()

        # Normal node
        hdv = random_distributional(dim, seed=1)
        graph.add_node(Node(id="normal", hdv=hdv, energy=0.5))

        # Uncertainty node with high energy
        hdv2 = random_distributional(dim, initial_variance=1.5, seed=2)
        graph.add_node(Node(
            id="uncertainty_123",
            hdv=hdv2,
            energy=1.0,
            content="Contradiction between X and Y"
        ))

        drive = CuriosityDrive(energy_threshold=0.8)
        needs_resolution = drive.get_pending_uncertainties(graph)

        assert "uncertainty_123" in needs_resolution
        assert "normal" not in needs_resolution

    def test_prioritizes_by_importance(self, dim):
        """Higher mass uncertainties should rank higher."""
        graph = KnowledgeGraph()

        # Low importance uncertainty
        hdv1 = random_distributional(dim, initial_variance=1.5, seed=1)
        graph.add_node(Node(
            id="uncertainty_low",
            hdv=hdv1,
            energy=1.0,
            mass=0.1,
        ))

        # High importance uncertainty
        hdv2 = random_distributional(dim, initial_variance=1.5, seed=2)
        graph.add_node(Node(
            id="uncertainty_high",
            hdv=hdv2,
            energy=1.0,
            mass=5.0,
        ))

        drive = CuriosityDrive()
        ranked = drive.get_pending_uncertainties(graph, ranked=True)

        # High importance should come first
        assert ranked[0] == "uncertainty_high"

    def test_should_seek_returns_true_for_high_priority(self, dim):
        """should_seek indicates when active seeking is warranted."""
        graph = KnowledgeGraph()

        hdv = random_distributional(dim, initial_variance=1.5, seed=1)
        graph.add_node(Node(
            id="uncertainty_critical",
            hdv=hdv,
            energy=1.0,
            mass=10.0,  # Very important
        ))

        drive = CuriosityDrive(seek_threshold=5.0)

        assert drive.should_seek(graph, "uncertainty_critical")

    def test_should_seek_false_for_low_priority(self, dim):
        """Low-priority uncertainties don't trigger active seeking."""
        graph = KnowledgeGraph()

        hdv = random_distributional(dim, initial_variance=1.5, seed=1)
        graph.add_node(Node(
            id="uncertainty_minor",
            hdv=hdv,
            energy=0.5,
            mass=0.1,
        ))

        drive = CuriosityDrive(seek_threshold=5.0)

        assert not drive.should_seek(graph, "uncertainty_minor")
