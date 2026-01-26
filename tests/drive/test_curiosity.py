"""Tests for Curiosity drive."""

import pytest
from engram.graph import Node, KnowledgeGraph
from engram.hdv import random_distributional
from engram.drive import CuriosityDrive


class TestCuriosityDrive:
    """Verify curiosity drive identifies what needs resolution."""

    def test_surfaces_recently_accessed_uncertainty(self, dim):
        """Recently accessed uncertainty nodes should surface.

        Scenario: Uncertainty node accessed recently should be
        returned as needing resolution (high effective strength).
        """
        graph = KnowledgeGraph()

        # Normal node (old access)
        hdv = random_distributional(dim, seed=1)
        graph.add_node(Node(id="normal", hdv=hdv, strength=0.5, last_accessed=0.0))

        # Uncertainty node accessed recently (high effective strength)
        hdv2 = random_distributional(dim, initial_variance=1.5, seed=2)
        graph.add_node(Node(
            id="uncertainty_123",
            hdv=hdv2,
            strength=1.0,
            last_accessed=9.9,  # Very recent
            content="Contradiction between X and Y"
        ))

        drive = CuriosityDrive(effective_strength_threshold=1.5, current_time=10.0)
        needs_resolution = drive.get_pending_uncertainties(graph, current_time=10.0)

        assert "uncertainty_123" in needs_resolution
        assert "normal" not in needs_resolution

    def test_prioritizes_by_strength(self, dim):
        """Higher strength uncertainties should rank higher."""
        graph = KnowledgeGraph()

        # Low strength uncertainty
        hdv1 = random_distributional(dim, initial_variance=1.5, seed=1)
        graph.add_node(Node(
            id="uncertainty_low",
            hdv=hdv1,
            strength=0.1,
            last_accessed=0.0,
        ))

        # High strength uncertainty
        hdv2 = random_distributional(dim, initial_variance=1.5, seed=2)
        graph.add_node(Node(
            id="uncertainty_high",
            hdv=hdv2,
            strength=5.0,
            last_accessed=0.0,
        ))

        drive = CuriosityDrive(effective_strength_threshold=0.0, current_time=0.0)
        ranked = drive.get_pending_uncertainties(graph, ranked=True, current_time=0.0)

        # High strength should come first
        assert ranked[0] == "uncertainty_high"

    def test_should_seek_returns_true_for_high_priority(self, dim):
        """should_seek indicates when active seeking is warranted."""
        graph = KnowledgeGraph()

        hdv = random_distributional(dim, initial_variance=1.5, seed=1)
        graph.add_node(Node(
            id="uncertainty_critical",
            hdv=hdv,
            strength=10.0,  # Very important
            last_accessed=0.0,
        ))

        drive = CuriosityDrive(seek_threshold=5.0, current_time=0.0)

        assert drive.should_seek(graph, "uncertainty_critical", current_time=0.0)

    def test_should_seek_false_for_low_priority(self, dim):
        """Low-priority uncertainties don't trigger active seeking."""
        graph = KnowledgeGraph()

        hdv = random_distributional(dim, initial_variance=1.5, seed=1)
        graph.add_node(Node(
            id="uncertainty_minor",
            hdv=hdv,
            strength=0.1,
            last_accessed=0.0,
        ))

        drive = CuriosityDrive(seek_threshold=5.0, current_time=0.0)

        assert not drive.should_seek(graph, "uncertainty_minor", current_time=0.0)
