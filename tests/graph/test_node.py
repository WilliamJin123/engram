"""Tests for Node class."""

import pytest
import torch
from engram.graph import Node
from engram.hdv import random_distributional


class TestNode:
    """Test suite for Node class."""

    def test_create_node_minimal(self):
        """Test creating a node with just ID and HDV."""
        hdv = random_distributional(dim=100, seed=42)
        node = Node(id="test_node", hdv=hdv)

        assert node.id == "test_node"
        assert node.hdv is hdv
        assert node.energy == 1.0  # default
        assert node.mass == 1.0  # default
        assert node.content is None  # default

    def test_create_node_full(self):
        """Test creating a node with all fields."""
        hdv = random_distributional(dim=100, seed=42)
        node = Node(
            id="full_node",
            hdv=hdv,
            energy=0.5,
            mass=2.0,
            content="This is a test node",
        )

        assert node.id == "full_node"
        assert node.energy == 0.5
        assert node.mass == 2.0
        assert node.content == "This is a test node"

    def test_energy_clamped_to_valid_range(self):
        """Test that energy is clamped to [0, 1]."""
        hdv = random_distributional(dim=100, seed=42)

        node_high = Node(id="high", hdv=hdv, energy=1.5)
        assert node_high.energy == 1.0

        node_low = Node(id="low", hdv=hdv, energy=-0.5)
        assert node_low.energy == 0.0

    def test_mass_must_be_positive(self):
        """Test that mass must be positive."""
        hdv = random_distributional(dim=100, seed=42)

        node_zero = Node(id="zero", hdv=hdv, mass=0.0)
        assert node_zero.mass > 0  # Should be clamped to minimum

        node_neg = Node(id="neg", hdv=hdv, mass=-1.0)
        assert node_neg.mass > 0  # Should be clamped to minimum

    def test_decay_reduces_energy(self):
        """Test that decay reduces energy over time."""
        hdv = random_distributional(dim=100, seed=42)
        node = Node(id="decaying", hdv=hdv, energy=1.0)

        decayed = node.decay(dt=10.0, energy_decay_rate=0.05, mass_decay_rate=0)

        assert decayed.energy < node.energy
        assert decayed.id == node.id
        assert decayed.mass == node.mass  # mass unchanged when rate=0

    def test_decay_reduces_mass_slowly(self):
        """Test that decay reduces mass very slowly."""
        hdv = random_distributional(dim=100, seed=42)
        node = Node(id="decaying", hdv=hdv, mass=10.0)

        decayed = node.decay(dt=10.0, mass_decay_rate=0.001)

        assert decayed.mass < node.mass
        assert decayed.mass > node.mass * 0.9  # slow decay

    def test_access_restores_energy(self):
        """Test that accessing a node restores energy."""
        hdv = random_distributional(dim=100, seed=42)
        node = Node(id="accessed", hdv=hdv, energy=0.3)

        accessed = node.access(energy_boost=0.5)

        assert accessed.energy > node.energy
        assert accessed.energy <= 1.0  # capped at 1

    def test_reinforce_adds_mass(self):
        """Test that successful use adds mass."""
        hdv = random_distributional(dim=100, seed=42)
        node = Node(id="reinforced", hdv=hdv, mass=1.0)

        reinforced = node.reinforce(mass_boost=0.5)

        assert reinforced.mass == 1.5

    def test_content_can_be_any_type(self):
        """Test that content can hold various types."""
        hdv = random_distributional(dim=100, seed=42)

        # String content
        node_str = Node(id="str", hdv=hdv, content="hello")
        assert node_str.content == "hello"

        # Dict content
        node_dict = Node(id="dict", hdv=hdv, content={"key": "value"})
        assert node_dict.content == {"key": "value"}

        # Numeric content
        node_num = Node(id="num", hdv=hdv, content=42.5)
        assert node_num.content == 42.5

    def test_node_repr(self):
        """Test node string representation."""
        hdv = random_distributional(dim=100, seed=42)
        node = Node(id="repr_test", hdv=hdv, energy=0.8, mass=2.0)
        repr_str = repr(node)

        assert "repr_test" in repr_str
        assert "0.8" in repr_str or "energy" in repr_str
