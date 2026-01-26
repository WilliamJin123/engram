"""Scenario-based tests for hyperbolic module."""

import torch
import pytest
from engram.hyperbolic import (
    embed_tree, is_ancestor, cone_query,
    project_to_poincare, poincare_distance, is_valid_poincare_point,
)
from engram.hyperbolic.poincare import compute_hybrid_position
from engram.graph import Node, Edge
from engram.hdv import random_distributional


class TestHyperbolicScenarios:
    """Scenario tests for hyperbolic embeddings."""

    def test_hierarchy_preservation(self, hyperbolic_dim):
        """Embed a tree and verify hierarchy is preserved in Poincare ball.

        Scenario: Create a taxonomy (animal -> mammal -> dog) and verify
        that parent nodes are closer to origin than children.
        """
        # Build a simple tree: root -> level1 -> level2
        tree = {
            "name": "root",
            "children": [
                {
                    "name": "mammal",
                    "children": [
                        {"name": "dog"},
                        {"name": "cat"},
                    ]
                },
                {
                    "name": "bird",
                    "children": [
                        {"name": "eagle"},
                    ]
                },
            ]
        }

        # Embed the tree
        embeddings = embed_tree(tree, dim=hyperbolic_dim, seed=42)

        # Returns dict with embedding for every node
        assert isinstance(embeddings, dict)
        expected_nodes = {"root", "mammal", "bird", "dog", "cat", "eagle"}
        assert set(embeddings.keys()) == expected_nodes

        # All embeddings are tensors of correct dimension
        for name, emb in embeddings.items():
            assert isinstance(emb, torch.Tensor)
            assert emb.shape == (hyperbolic_dim,)
            assert is_valid_poincare_point(emb)

        # Root is closest to origin
        root_dist = torch.norm(embeddings["root"]).item()
        for name, emb in embeddings.items():
            if name != "root":
                assert torch.norm(emb).item() > root_dist

        # Parents closer to origin than children
        assert torch.norm(embeddings["root"]) < torch.norm(embeddings["mammal"])
        assert torch.norm(embeddings["mammal"]) < torch.norm(embeddings["dog"])
        assert torch.norm(embeddings["mammal"]) < torch.norm(embeddings["cat"])
        assert torch.norm(embeddings["root"]) < torch.norm(embeddings["bird"])
        assert torch.norm(embeddings["bird"]) < torch.norm(embeddings["eagle"])

        # Reproducible with same seed
        embeddings2 = embed_tree(tree, dim=hyperbolic_dim, seed=42)
        for name in expected_nodes:
            assert torch.equal(embeddings[name], embeddings2[name])

        # Different seed gives different result
        embeddings3 = embed_tree(tree, dim=hyperbolic_dim, seed=99)
        assert not torch.equal(embeddings["root"], embeddings3["root"])

    def test_cone_containment(self, hyperbolic_dim):
        """Verify ancestor/descendant relationships via cone queries.

        Scenario: In an embedded hierarchy, check that is_ancestor correctly
        identifies ancestry relationships and cone_query returns all ancestors.
        """
        tree = {
            "name": "animal",
            "children": [
                {
                    "name": "mammal",
                    "children": [
                        {"name": "dog"},
                        {"name": "cat"},
                    ]
                },
                {
                    "name": "bird",
                    "children": [
                        {"name": "eagle"},
                    ]
                },
            ]
        }
        embeddings = embed_tree(tree, dim=hyperbolic_dim, seed=42)

        # Root is ancestor of all nodes
        for name in ["mammal", "bird", "dog", "cat", "eagle"]:
            assert is_ancestor(embeddings["animal"], embeddings[name])

        # Parent is ancestor of child
        assert is_ancestor(embeddings["mammal"], embeddings["dog"])
        assert is_ancestor(embeddings["mammal"], embeddings["cat"])
        assert is_ancestor(embeddings["bird"], embeddings["eagle"])

        # Child is NOT ancestor of parent
        assert not is_ancestor(embeddings["dog"], embeddings["mammal"])
        assert not is_ancestor(embeddings["mammal"], embeddings["animal"])

        # Siblings are NOT ancestors of each other
        assert not is_ancestor(embeddings["dog"], embeddings["cat"])
        assert not is_ancestor(embeddings["mammal"], embeddings["bird"])

        # Node is NOT ancestor of itself
        assert not is_ancestor(embeddings["dog"], embeddings["dog"])

        # Cone query returns all ancestors
        dog_ancestors = cone_query(embeddings["dog"], embeddings)
        assert "animal" in dog_ancestors
        assert "mammal" in dog_ancestors
        assert "dog" not in dog_ancestors  # Excludes self
        assert "cat" not in dog_ancestors  # Excludes siblings
        assert "bird" not in dog_ancestors  # Excludes other branches

        # Root has no ancestors
        root_ancestors = cone_query(embeddings["animal"], embeddings)
        assert len(root_ancestors) == 0

    def test_poincare_operations(self, hyperbolic_dim):
        """Verify Poincare ball operations (projection, distance).

        Scenario: Project points into Poincare ball and verify distance
        properties (symmetry, triangle inequality, boundary behavior).
        """
        # Projection keeps points inside ball
        large_vec = torch.randn(hyperbolic_dim) * 10
        projected = project_to_poincare(large_vec)
        assert torch.norm(projected).item() < 1.0
        assert is_valid_poincare_point(projected)

        # Projection preserves direction
        cos_sim = torch.dot(large_vec, projected) / (torch.norm(large_vec) * torch.norm(projected))
        assert cos_sim.item() > 0.99

        # Small vectors stay mostly unchanged
        small_vec = torch.randn(hyperbolic_dim) * 0.1
        proj_small = project_to_poincare(small_vec)
        assert torch.allclose(proj_small, small_vec, atol=0.01)

        # Zero vector maps to origin
        zero = project_to_poincare(torch.zeros(hyperbolic_dim))
        assert torch.norm(zero).item() < 1e-6

        # Distance properties
        a = project_to_poincare(torch.randn(hyperbolic_dim) * 0.5)
        b = project_to_poincare(torch.randn(hyperbolic_dim) * 0.5)
        c = project_to_poincare(torch.randn(hyperbolic_dim) * 0.5)

        # Distance is non-negative
        assert poincare_distance(a, b) >= 0

        # Distance to self is zero
        assert poincare_distance(a, a) < 1e-6

        # Distance is symmetric
        assert abs(poincare_distance(a, b) - poincare_distance(b, a)) < 1e-6

        # Triangle inequality
        d_ac = poincare_distance(a, c)
        d_ab = poincare_distance(a, b)
        d_bc = poincare_distance(b, c)
        assert d_ac <= d_ab + d_bc + 1e-6

        # Distance increases toward boundary
        origin = torch.zeros(hyperbolic_dim)
        direction = torch.randn(hyperbolic_dim)
        direction = direction / torch.norm(direction)
        near = direction * 0.3
        far = direction * 0.8
        assert poincare_distance(origin, far) > poincare_distance(origin, near)

    def test_hybrid_positioning(self):
        """Test hybrid positioning combining graph depth and HDV similarity.

        Scenario: Position nodes using both graph structure (for radius)
        and HDV similarity (for angular position).
        """
        # Create nodes with HDVs
        parent_hdv = random_distributional(dim=100, seed=42)
        child_hdv = random_distributional(dim=100, seed=43)

        parent = Node(id="parent", hdv=parent_hdv, mass=5.0)
        child = Node(id="child", hdv=child_hdv, mass=1.0)

        nodes = {"parent": parent, "child": child}
        edges = [Edge(source="child", target="parent")]

        # Compute positions
        pos_parent = compute_hybrid_position(parent, nodes, edges)
        pos_child = compute_hybrid_position(child, nodes, edges)

        # All positions are valid Poincare points
        assert is_valid_poincare_point(pos_parent)
        assert is_valid_poincare_point(pos_child)

        # Child further from origin than parent (hierarchy)
        assert torch.norm(pos_child) > torch.norm(pos_parent)

        # Root node (no parents) near origin
        root_hdv = random_distributional(dim=100, seed=1)
        root = Node(id="root", hdv=root_hdv, mass=10.0)
        pos_root = compute_hybrid_position(root, {"root": root}, [])
        assert torch.norm(pos_root).item() < 0.3

        # Test angular clustering with similar HDVs (no edges, just HDV similarity)
        base_hdv = random_distributional(dim=100, seed=42)
        similar_hdv = random_distributional(dim=100, seed=42)  # Same seed = same HDV
        different_hdv = random_distributional(dim=100, seed=99)

        node_a = Node(id="a", hdv=base_hdv, mass=1.0)
        node_b = Node(id="b", hdv=similar_hdv, mass=1.0)
        node_c = Node(id="c", hdv=different_hdv, mass=1.0)

        nodes_abc = {"a": node_a, "b": node_b, "c": node_c}
        pos_a = compute_hybrid_position(node_a, nodes_abc, [])
        pos_b = compute_hybrid_position(node_b, nodes_abc, [])
        pos_c = compute_hybrid_position(node_c, nodes_abc, [])

        # Normalize to get directions
        dir_a = pos_a / torch.norm(pos_a)
        dir_b = pos_b / torch.norm(pos_b)
        dir_c = pos_c / torch.norm(pos_c)

        # a and b should be more angularly similar than a and c
        cos_ab = torch.dot(dir_a, dir_b).item()
        cos_ac = torch.dot(dir_a, dir_c).item()
        assert cos_ab > cos_ac

        # High mass nodes closer to origin
        low_mass = Node(id="low", hdv=parent_hdv, mass=0.1)
        high_mass = Node(id="high", hdv=parent_hdv, mass=100.0)
        pos_low = compute_hybrid_position(low_mass, {"low": low_mass}, [])
        pos_high = compute_hybrid_position(high_mass, {"high": high_mass}, [])
        assert torch.norm(pos_high) < torch.norm(pos_low)
