"""Tests for cone query operations in hyperbolic space."""

import torch
import pytest
from engram.hyperbolic import (
    embed_tree,
    cone_query,
    is_ancestor,
    poincare_distance,
)


@pytest.fixture
def tree_with_embeddings(hyperbolic_dim):
    """Create a tree and its embeddings for testing."""
    tree = {
        "name": "root",
        "children": [
            {
                "name": "A",
                "children": [
                    {"name": "A1"},
                    {"name": "A2"},
                ]
            },
            {
                "name": "B",
                "children": [
                    {"name": "B1"},
                    {"name": "B2"},
                ]
            },
        ]
    }
    embeddings = embed_tree(tree, hyperbolic_dim, seed=42)
    return tree, embeddings


class TestIsAncestor:
    """Test suite for is_ancestor function."""

    def test_returns_bool(self, tree_with_embeddings):
        """Should return a boolean."""
        _, embeddings = tree_with_embeddings
        result = is_ancestor(embeddings["root"], embeddings["A"])
        assert isinstance(result, bool)

    def test_root_is_ancestor_of_all(self, tree_with_embeddings):
        """Root should be ancestor of all other nodes."""
        _, embeddings = tree_with_embeddings
        for name in ["A", "B", "A1", "A2", "B1", "B2"]:
            assert is_ancestor(embeddings["root"], embeddings[name]), \
                f"Root should be ancestor of {name}"

    def test_parent_is_ancestor_of_child(self, tree_with_embeddings):
        """Parent should be ancestor of its children."""
        _, embeddings = tree_with_embeddings
        assert is_ancestor(embeddings["A"], embeddings["A1"])
        assert is_ancestor(embeddings["A"], embeddings["A2"])
        assert is_ancestor(embeddings["B"], embeddings["B1"])
        assert is_ancestor(embeddings["B"], embeddings["B2"])

    def test_child_not_ancestor_of_parent(self, tree_with_embeddings):
        """Child should not be ancestor of parent."""
        _, embeddings = tree_with_embeddings
        assert not is_ancestor(embeddings["A"], embeddings["root"])
        assert not is_ancestor(embeddings["A1"], embeddings["A"])
        assert not is_ancestor(embeddings["A1"], embeddings["root"])

    def test_sibling_not_ancestor(self, tree_with_embeddings):
        """Siblings should not be ancestors of each other."""
        _, embeddings = tree_with_embeddings
        assert not is_ancestor(embeddings["A"], embeddings["B"])
        assert not is_ancestor(embeddings["B"], embeddings["A"])
        assert not is_ancestor(embeddings["A1"], embeddings["A2"])

    def test_cousin_not_ancestor(self, tree_with_embeddings):
        """Cousins should not be ancestors of each other."""
        _, embeddings = tree_with_embeddings
        assert not is_ancestor(embeddings["A1"], embeddings["B1"])
        assert not is_ancestor(embeddings["B2"], embeddings["A1"])

    def test_node_not_ancestor_of_itself(self, tree_with_embeddings):
        """Node should not be ancestor of itself."""
        _, embeddings = tree_with_embeddings
        assert not is_ancestor(embeddings["A"], embeddings["A"])
        assert not is_ancestor(embeddings["root"], embeddings["root"])


class TestConeQuery:
    """Test suite for cone_query function."""

    def test_returns_list(self, tree_with_embeddings):
        """Should return a list."""
        _, embeddings = tree_with_embeddings
        result = cone_query(embeddings["A1"], embeddings)
        assert isinstance(result, list)

    def test_returns_ancestor_names(self, tree_with_embeddings):
        """VALIDATION: Should return correct ancestors."""
        _, embeddings = tree_with_embeddings

        # Query for A1: ancestors should be A and root
        ancestors_of_A1 = cone_query(embeddings["A1"], embeddings)
        assert "root" in ancestors_of_A1, "root should be ancestor of A1"
        assert "A" in ancestors_of_A1, "A should be ancestor of A1"

    def test_excludes_non_ancestors(self, tree_with_embeddings):
        """VALIDATION: Should exclude nodes from other branches."""
        _, embeddings = tree_with_embeddings

        # Query for A1: should NOT include B, B1, B2
        ancestors_of_A1 = cone_query(embeddings["A1"], embeddings)
        assert "B" not in ancestors_of_A1, "B should not be ancestor of A1"
        assert "B1" not in ancestors_of_A1, "B1 should not be ancestor of A1"
        assert "B2" not in ancestors_of_A1, "B2 should not be ancestor of A1"

    def test_excludes_siblings(self, tree_with_embeddings):
        """Should exclude siblings from ancestor list."""
        _, embeddings = tree_with_embeddings

        ancestors_of_A1 = cone_query(embeddings["A1"], embeddings)
        assert "A2" not in ancestors_of_A1, "A2 (sibling) should not be ancestor of A1"

    def test_excludes_self(self, tree_with_embeddings):
        """Should exclude the query node itself."""
        _, embeddings = tree_with_embeddings

        ancestors_of_A1 = cone_query(embeddings["A1"], embeddings)
        assert "A1" not in ancestors_of_A1, "A1 should not be its own ancestor"

    def test_excludes_descendants(self, tree_with_embeddings):
        """Should exclude descendants from ancestor list."""
        _, embeddings = tree_with_embeddings

        # Query for A: should NOT include A1, A2 (descendants)
        ancestors_of_A = cone_query(embeddings["A"], embeddings)
        assert "A1" not in ancestors_of_A, "A1 should not be ancestor of A"
        assert "A2" not in ancestors_of_A, "A2 should not be ancestor of A"

    def test_root_has_no_ancestors(self, tree_with_embeddings):
        """Root should have no ancestors."""
        _, embeddings = tree_with_embeddings
        ancestors_of_root = cone_query(embeddings["root"], embeddings)
        assert len(ancestors_of_root) == 0, "Root should have no ancestors"


class TestConeQueryRobustness:
    """Test cone query robustness across different trees."""

    def test_deep_hierarchy(self, hyperbolic_dim):
        """Test ancestor finding in deep hierarchy."""
        tree = {
            "name": "L0",
            "children": [{
                "name": "L1",
                "children": [{
                    "name": "L2",
                    "children": [{
                        "name": "L3",
                        "children": [{"name": "L4"}]
                    }]
                }]
            }]
        }
        embeddings = embed_tree(tree, hyperbolic_dim, seed=42)

        # L4 should have all upper levels as ancestors
        ancestors = cone_query(embeddings["L4"], embeddings)
        for level in ["L0", "L1", "L2", "L3"]:
            assert level in ancestors, f"{level} should be ancestor of L4"

        # L4 should not be in ancestors
        assert "L4" not in ancestors

    def test_wide_hierarchy(self, hyperbolic_dim):
        """Test that cone query distinguishes many siblings."""
        tree = {
            "name": "root",
            "children": [
                {"name": f"child{i}", "children": [{"name": f"grandchild{i}"}]}
                for i in range(5)
            ]
        }
        embeddings = embed_tree(tree, hyperbolic_dim, seed=42)

        # grandchild0's ancestors should only be child0 and root
        ancestors = cone_query(embeddings["grandchild0"], embeddings)
        assert "root" in ancestors
        assert "child0" in ancestors

        # Should not include other branches
        for i in range(1, 5):
            assert f"child{i}" not in ancestors, f"child{i} should not be ancestor"
            assert f"grandchild{i}" not in ancestors
