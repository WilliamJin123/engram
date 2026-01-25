"""Tests for hierarchy embedding in hyperbolic space."""

import torch
import pytest
from engram.hyperbolic import (
    embed_tree,
    poincare_distance,
    is_valid_poincare_point,
)


@pytest.fixture
def simple_tree():
    """Simple 3-level tree for testing."""
    return {
        "name": "root",
        "children": [
            {
                "name": "child1",
                "children": [
                    {"name": "grandchild1a"},
                    {"name": "grandchild1b"},
                ]
            },
            {
                "name": "child2",
                "children": [
                    {"name": "grandchild2a"},
                ]
            },
        ]
    }


@pytest.fixture
def deep_tree():
    """Deeper tree (5 levels) for testing."""
    return {
        "name": "level0",
        "children": [{
            "name": "level1",
            "children": [{
                "name": "level2",
                "children": [{
                    "name": "level3",
                    "children": [{"name": "level4"}]
                }]
            }]
        }]
    }


@pytest.fixture
def wide_tree():
    """Wide tree (many siblings) for testing."""
    return {
        "name": "root",
        "children": [
            {"name": f"child{i}"} for i in range(10)
        ]
    }


class TestEmbedTree:
    """Test suite for embed_tree function."""

    def test_returns_dict(self, simple_tree, hyperbolic_dim):
        """Should return a dictionary."""
        result = embed_tree(simple_tree, hyperbolic_dim, seed=42)
        assert isinstance(result, dict)

    def test_contains_all_nodes(self, simple_tree, hyperbolic_dim):
        """Should contain embedding for every node."""
        result = embed_tree(simple_tree, hyperbolic_dim, seed=42)
        expected_nodes = {
            "root", "child1", "child2",
            "grandchild1a", "grandchild1b", "grandchild2a"
        }
        assert set(result.keys()) == expected_nodes

    def test_embeddings_are_tensors(self, simple_tree, hyperbolic_dim):
        """All embeddings should be torch tensors."""
        result = embed_tree(simple_tree, hyperbolic_dim, seed=42)
        for name, embedding in result.items():
            assert isinstance(embedding, torch.Tensor), f"{name} not a tensor"

    def test_correct_dimension(self, simple_tree, hyperbolic_dim):
        """All embeddings should have correct dimension."""
        result = embed_tree(simple_tree, hyperbolic_dim, seed=42)
        for name, embedding in result.items():
            assert embedding.shape == (hyperbolic_dim,), f"{name} wrong shape"

    def test_all_valid_poincare_points(self, simple_tree, hyperbolic_dim):
        """All embeddings should be valid Poincare points."""
        result = embed_tree(simple_tree, hyperbolic_dim, seed=42)
        for name, embedding in result.items():
            assert is_valid_poincare_point(embedding), f"{name} not in Poincare ball"

    def test_reproducible_with_seed(self, simple_tree, hyperbolic_dim):
        """Should produce same embeddings with same seed."""
        result1 = embed_tree(simple_tree, hyperbolic_dim, seed=42)
        result2 = embed_tree(simple_tree, hyperbolic_dim, seed=42)
        for name in result1:
            assert torch.equal(result1[name], result2[name])

    def test_different_with_different_seeds(self, simple_tree, hyperbolic_dim):
        """Should produce different embeddings with different seeds."""
        result1 = embed_tree(simple_tree, hyperbolic_dim, seed=42)
        result2 = embed_tree(simple_tree, hyperbolic_dim, seed=43)
        # At least some embeddings should differ
        any_different = any(
            not torch.equal(result1[name], result2[name])
            for name in result1
        )
        assert any_different


class TestHierarchyStructure:
    """Test that embeddings preserve hierarchical structure."""

    def test_root_closest_to_origin(self, simple_tree, hyperbolic_dim):
        """VALIDATION: Root should be closest to origin."""
        embeddings = embed_tree(simple_tree, hyperbolic_dim, seed=42)
        origin = torch.zeros(hyperbolic_dim)

        root_dist = poincare_distance(origin, embeddings["root"])

        for name, emb in embeddings.items():
            if name != "root":
                dist = poincare_distance(origin, emb)
                assert dist > root_dist, f"{name} closer to origin than root"

    def test_parents_closer_to_origin_than_children(self, simple_tree, hyperbolic_dim):
        """VALIDATION: Parents should be closer to origin than their children."""
        embeddings = embed_tree(simple_tree, hyperbolic_dim, seed=42)
        origin = torch.zeros(hyperbolic_dim)

        # Check parent-child relationships
        parent_child_pairs = [
            ("root", "child1"),
            ("root", "child2"),
            ("child1", "grandchild1a"),
            ("child1", "grandchild1b"),
            ("child2", "grandchild2a"),
        ]

        for parent, child in parent_child_pairs:
            parent_dist = poincare_distance(origin, embeddings[parent])
            child_dist = poincare_distance(origin, embeddings[child])
            assert parent_dist < child_dist, \
                f"Parent {parent} (d={parent_dist:.3f}) not closer than child {child} (d={child_dist:.3f})"

    def test_siblings_closer_than_cousins(self, simple_tree, hyperbolic_dim):
        """VALIDATION: Siblings should be closer to each other than to cousins."""
        embeddings = embed_tree(simple_tree, hyperbolic_dim, seed=42)

        # Siblings
        sibling_dist = poincare_distance(
            embeddings["grandchild1a"],
            embeddings["grandchild1b"]
        )

        # Cousins
        cousin_dist = poincare_distance(
            embeddings["grandchild1a"],
            embeddings["grandchild2a"]
        )

        assert sibling_dist < cousin_dist, \
            f"Siblings (d={sibling_dist:.3f}) not closer than cousins (d={cousin_dist:.3f})"

    def test_deep_tree_hierarchy(self, deep_tree, hyperbolic_dim):
        """Deeper levels should be progressively farther from origin."""
        embeddings = embed_tree(deep_tree, hyperbolic_dim, seed=42)
        origin = torch.zeros(hyperbolic_dim)

        distances = [
            poincare_distance(origin, embeddings[f"level{i}"])
            for i in range(5)
        ]

        # Each level should be farther than previous
        for i in range(1, len(distances)):
            assert distances[i] > distances[i-1], \
                f"Level {i} not farther than level {i-1}"

    def test_wide_tree_sibling_separation(self, wide_tree, hyperbolic_dim):
        """Siblings should be spread out (not all at same point)."""
        embeddings = embed_tree(wide_tree, hyperbolic_dim, seed=42)

        # Check that siblings are separated
        children = [f"child{i}" for i in range(10)]
        min_sibling_dist = float('inf')
        max_sibling_dist = 0

        for i, c1 in enumerate(children):
            for c2 in children[i+1:]:
                d = poincare_distance(embeddings[c1], embeddings[c2])
                min_sibling_dist = min(min_sibling_dist, d)
                max_sibling_dist = max(max_sibling_dist, d)

        # Siblings should have meaningful separation
        assert min_sibling_dist > 0.01, "Siblings too close together"


class TestLeafNode:
    """Test handling of leaf nodes (string format)."""

    def test_leaf_only_tree(self, hyperbolic_dim):
        """Should handle tree that is just a leaf."""
        tree = {"name": "only_node"}
        result = embed_tree(tree, hyperbolic_dim, seed=42)
        assert "only_node" in result
        assert is_valid_poincare_point(result["only_node"])

    def test_string_leaf_format(self, hyperbolic_dim):
        """Should handle string format for leaves."""
        tree = {
            "name": "root",
            "children": ["leaf1", "leaf2"]  # String format
        }
        result = embed_tree(tree, hyperbolic_dim, seed=42)
        assert "leaf1" in result
        assert "leaf2" in result
