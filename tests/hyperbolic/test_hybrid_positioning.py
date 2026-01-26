"""Tests for hybrid hyperbolic positioning (radius from graph, angle from HDV)."""

import pytest
import torch
from engram.hyperbolic.poincare import compute_hybrid_position, is_valid_poincare_point
from engram.hdv import random_distributional, distributional_similarity
from engram.graph import Node, Edge


class TestHybridPositioning:
    """Test suite for hybrid positioning."""

    def test_root_node_near_origin(self):
        """Test that a root node (no parents) is positioned near origin."""
        hdv = random_distributional(dim=100, seed=42)
        root = Node(id="root", hdv=hdv, mass=10.0)

        # No edges means root is at depth 0
        edges = []
        nodes = {"root": root}

        pos = compute_hybrid_position(root, nodes, edges)

        assert is_valid_poincare_point(pos)
        assert torch.norm(pos).item() < 0.3  # Near origin

    def test_child_further_from_origin_than_parent(self):
        """Test that children are further from origin than parents."""
        parent_hdv = random_distributional(dim=100, seed=42)
        child_hdv = random_distributional(dim=100, seed=43)

        parent = Node(id="parent", hdv=parent_hdv, mass=5.0)
        child = Node(id="child", hdv=child_hdv, mass=1.0)

        nodes = {"parent": parent, "child": child}
        edges = [Edge(source="child", target="parent")]  # child points to parent

        parent_pos = compute_hybrid_position(parent, nodes, edges)
        child_pos = compute_hybrid_position(child, nodes, edges)

        assert torch.norm(child_pos) > torch.norm(parent_pos)

    def test_similar_hdvs_cluster_angularly(self):
        """Test that nodes with similar HDVs have similar angular positions."""
        base_hdv = random_distributional(dim=100, seed=42)

        # Create two nodes with very similar HDVs
        similar_hdv = random_distributional(dim=100, seed=42)  # Same seed = same HDV
        different_hdv = random_distributional(dim=100, seed=99)

        node_a = Node(id="a", hdv=base_hdv, mass=1.0)
        node_b = Node(id="b", hdv=similar_hdv, mass=1.0)
        node_c = Node(id="c", hdv=different_hdv, mass=1.0)

        nodes = {"a": node_a, "b": node_b, "c": node_c}
        edges = []

        pos_a = compute_hybrid_position(node_a, nodes, edges)
        pos_b = compute_hybrid_position(node_b, nodes, edges)
        pos_c = compute_hybrid_position(node_c, nodes, edges)

        # Normalize to get directions
        dir_a = pos_a / torch.norm(pos_a)
        dir_b = pos_b / torch.norm(pos_b)
        dir_c = pos_c / torch.norm(pos_c)

        # a and b should be more angularly similar than a and c
        cos_ab = torch.dot(dir_a, dir_b).item()
        cos_ac = torch.dot(dir_a, dir_c).item()

        assert cos_ab > cos_ac

    def test_high_mass_nodes_closer_to_origin(self):
        """Test that high-mass nodes tend toward origin (more abstract)."""
        hdv = random_distributional(dim=100, seed=42)

        low_mass = Node(id="low", hdv=hdv, mass=0.1)
        high_mass = Node(id="high", hdv=hdv, mass=100.0)

        nodes = {"low": low_mass, "high": high_mass}
        edges = []

        pos_low = compute_hybrid_position(low_mass, nodes, edges)
        pos_high = compute_hybrid_position(high_mass, nodes, edges)

        # High mass should be closer to origin
        assert torch.norm(pos_high) < torch.norm(pos_low)
