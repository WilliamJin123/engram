# Node-Centric Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor Engram to a uniform node-centric architecture where everything is a node, edges are unlabeled arrows, and HDV bind/bundle are the compositional primitives.

**Architecture:** Two uniform primitives that mirror each other: (1) Graph: nodes are the single primitive, edges are unlabeled `(source, target)` arrows; (2) HDV: bind creates associations, bundle creates collections. Content is human-readable source, HDV is the compiled computational form. Hyperbolic position is derived (radius from graph structure, angle from HDV similarity) but secondary to HDV as source of truth.

**Tech Stack:** Python, PyTorch, dataclasses

---

## Summary of Design Decisions

| Decision | Choice |
|----------|--------|
| Graph primitive | Node (everything is a node) |
| Edge structure | Unlabeled arrows: `(source, target)` only |
| HDV operations | Bind (associations) + Bundle (collections) |
| Node fields | id, hdv, energy, mass, content |
| Type representation | Types are nodes, connected via edges |
| Confidence/uncertainty | Encoded in distributional HDV variance |
| Content purpose | Human-readable source; LLM encodes to HDV at ingestion |
| Hyperbolic position | Hybrid: radius from graph, angle from HDV |
| HDV as source of truth | Yes; hyperbolic is derived |

---

## Task 1: Simplify Edge to Unlabeled Arrows

**Files:**
- Modify: `src/engram/graph/edge.py`
- Modify: `tests/graph/test_edge.py`

**Step 1: Write the new Edge tests**

Replace the entire test file with tests for the simplified Edge:

```python
"""Tests for simplified Edge class (unlabeled arrows)."""

import pytest
from engram.graph import Edge


class TestEdge:
    """Test suite for Edge class."""

    def test_create_edge(self):
        """Test basic edge creation with source and target only."""
        edge = Edge(source="node_a", target="node_b")

        assert edge.source == "node_a"
        assert edge.target == "node_b"

    def test_edge_equality(self):
        """Test that edges with same source/target are equal."""
        edge1 = Edge(source="a", target="b")
        edge2 = Edge(source="a", target="b")

        assert edge1 == edge2

    def test_edge_inequality(self):
        """Test that edges with different source/target are not equal."""
        edge1 = Edge(source="a", target="b")
        edge2 = Edge(source="a", target="c")
        edge3 = Edge(source="c", target="b")

        assert edge1 != edge2
        assert edge1 != edge3

    def test_edge_repr(self):
        """Test edge string representation."""
        edge = Edge(source="node1", target="node2")
        repr_str = repr(edge)

        assert "node1" in repr_str
        assert "node2" in repr_str

    def test_edge_hash(self):
        """Test that edges can be used in sets."""
        edge1 = Edge(source="a", target="b")
        edge2 = Edge(source="a", target="b")
        edge3 = Edge(source="b", target="a")

        edge_set = {edge1, edge2, edge3}
        assert len(edge_set) == 2  # edge1 and edge2 are duplicates
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/graph/test_edge.py -v`
Expected: FAIL (old Edge has edge_type required)

**Step 3: Implement simplified Edge**

Replace `src/engram/graph/edge.py`:

```python
"""Edge as unlabeled directional arrow.

In the node-centric architecture, edges are pure structural primitives.
All semantic meaning lives in nodes. An edge simply says "source points to target".
Relationships, confidence, metadata are all expressed as nodes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    """An unlabeled directional arrow connecting two nodes.

    Attributes:
        source: ID of the source node.
        target: ID of the target node.
    """
    source: str
    target: str
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/graph/test_edge.py -v`
Expected: PASS

**Step 5: Fix dependent code**

The `propagate_through_edge` function in `src/engram/hdv/uncertainty.py` uses `edge.uncertainty`. This needs to be updated since Edge no longer has that property. For now, we'll remove the edge parameter and just propagate based on a fixed uncertainty or make it a parameter.

Update `src/engram/hdv/uncertainty.py` - change `propagate_through_edge`:

```python
def propagate_through_edge(
    source: DistributionalHDV,
    edge_hdv: torch.Tensor,
    edge_variance: float,
    params: UncertaintyParams,
) -> DistributionalHDV:
    """Propagate a distributional signal through an edge.

    Signal traveling through an edge:
    1. Mean transforms via binding with edge type HDV
    2. Variance increases based on edge uncertainty

    This implements "propagation compounds uncertainty"—information
    traveling through the graph accumulates variance at each hop.

    Args:
        source: Source distributional HDV.
        edge_hdv: The HDV for the relationship node this edge passes through.
        edge_variance: Uncertainty of this edge (0=certain, higher=uncertain).
        params: Uncertainty parameters.

    Returns:
        Propagated distributional HDV at the target.
    """
    # Mean transforms via bind (element-wise multiply with edge HDV)
    new_mean = source.mean * edge_hdv

    # Edge uncertainty contribution
    edge_uncertainty = edge_variance * params.base_edge_variance

    # Variance compounds: source variance + edge uncertainty
    new_variance = source.variance + edge_uncertainty

    # Clamp to bounds
    new_variance = torch.clamp(new_variance, min=params.min_variance, max=params.max_variance)

    return DistributionalHDV(
        mean=new_mean,
        variance=new_variance,
        last_accessed=source.last_accessed,
        last_updated=source.last_updated,
    )
```

Remove the `Edge` import from uncertainty.py (line 6).

**Step 6: Update tests for propagate_through_edge**

Find and update any tests that use the old signature. Check `tests/hdv/test_integration.py` and `tests/hdv/test_uncertainty.py`.

**Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: PASS

**Step 8: Commit**

```bash
git add src/engram/graph/edge.py tests/graph/test_edge.py src/engram/hdv/uncertainty.py
git commit -m "refactor(edge): simplify Edge to unlabeled arrows

Edge now only has source and target. All semantic meaning lives in nodes.
This aligns with the node-centric architecture where relationships are nodes.

Updated propagate_through_edge to take edge_variance parameter instead of Edge object."
```

---

## Task 2: Create Node Class

**Files:**
- Create: `src/engram/graph/node.py`
- Create: `tests/graph/test_node.py`
- Modify: `src/engram/graph/__init__.py`

**Step 1: Write the Node tests**

Create `tests/graph/test_node.py`:

```python
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

        decayed = node.decay(dt=10.0, energy_decay_rate=0.05)

        assert decayed.energy < node.energy
        assert decayed.id == node.id
        assert decayed.mass == node.mass  # mass unchanged

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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/graph/test_node.py -v`
Expected: FAIL (Node doesn't exist)

**Step 3: Implement Node class**

Create `src/engram/graph/node.py`:

```python
"""Node as the universal primitive for knowledge representation.

In the node-centric architecture, everything is a node:
- Facts, concepts, entities
- Relationships (connecting other nodes)
- Types (connected to by instances)
- Agents, workflows, the graph itself
- Meta-relationships, conditionals, provenance

All semantic meaning lives in nodes. Edges are just unlabeled arrows.
"""

from dataclasses import dataclass
from typing import Any

from engram.hdv import DistributionalHDV


@dataclass
class Node:
    """A node in the knowledge graph.

    Attributes:
        id: Unique identifier for this node.
        hdv: Distributional HDV encoding this node's identity/meaning.
             Source of truth for similarity and composition.
        energy: Recency/relevance score (0-1). Decays fast without access.
        mass: Importance/confidence score (>0). Decays slowly, grows with use.
        content: Optional human-readable payload. LLM encodes this to HDV.
    """
    id: str
    hdv: DistributionalHDV
    energy: float = 1.0
    mass: float = 1.0
    content: Any = None

    # Minimum mass to prevent complete decay
    MIN_MASS: float = 0.01

    def __post_init__(self):
        """Clamp energy and mass to valid ranges."""
        self.energy = max(0.0, min(1.0, self.energy))
        self.mass = max(self.MIN_MASS, self.mass)

    def decay(
        self,
        dt: float,
        energy_decay_rate: float = 0.1,
        mass_decay_rate: float = 0.001,
    ) -> "Node":
        """Apply temporal decay to energy and mass.

        Args:
            dt: Time elapsed since last decay.
            energy_decay_rate: Energy loss per time unit.
            mass_decay_rate: Mass loss per time unit (much slower).

        Returns:
            New Node with decayed energy and mass.
        """
        new_energy = self.energy * (1.0 - energy_decay_rate * dt)
        new_energy = max(0.0, new_energy)

        new_mass = self.mass * (1.0 - mass_decay_rate * dt)
        new_mass = max(self.MIN_MASS, new_mass)

        return Node(
            id=self.id,
            hdv=self.hdv,
            energy=new_energy,
            mass=new_mass,
            content=self.content,
        )

    def access(self, energy_boost: float = 0.3) -> "Node":
        """Restore energy when node is accessed.

        Args:
            energy_boost: Amount of energy to restore.

        Returns:
            New Node with boosted energy.
        """
        new_energy = min(1.0, self.energy + energy_boost)

        return Node(
            id=self.id,
            hdv=self.hdv,
            energy=new_energy,
            mass=self.mass,
            content=self.content,
        )

    def reinforce(self, mass_boost: float = 0.1) -> "Node":
        """Add mass when node is successfully used.

        Args:
            mass_boost: Amount of mass to add.

        Returns:
            New Node with increased mass.
        """
        return Node(
            id=self.id,
            hdv=self.hdv,
            energy=self.energy,
            mass=self.mass + mass_boost,
            content=self.content,
        )

    def __repr__(self) -> str:
        content_preview = (
            f"'{self.content[:20]}...'" if isinstance(self.content, str) and len(self.content) > 20
            else repr(self.content)
        )
        return (
            f"Node(id='{self.id}', energy={self.energy:.2f}, "
            f"mass={self.mass:.2f}, content={content_preview})"
        )
```

**Step 4: Update graph __init__.py**

Update `src/engram/graph/__init__.py`:

```python
"""Graph components for Engram."""

from .edge import Edge
from .node import Node

__all__ = ["Edge", "Node"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/graph/test_node.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/engram/graph/node.py tests/graph/test_node.py src/engram/graph/__init__.py
git commit -m "feat(node): add Node class as universal primitive

Node has: id, hdv, energy, mass, content
- HDV is source of truth for similarity/composition
- Energy decays fast, restored on access
- Mass decays slow, grows on successful use
- Content is human-readable payload for LLM encoding"
```

---

## Task 3: Update Package Exports

**Files:**
- Modify: `src/engram/__init__.py`

**Step 1: Update main package exports**

Update `src/engram/__init__.py` to export Node:

```python
"""Engram: Self-organizing memory framework with epistemic uncertainty.

A memory system using distributional High-Dimensional Vectors (HDVs)
for principled uncertainty tracking in knowledge graphs.
"""

__version__ = "0.3.0"  # Bump for node-centric architecture

from .hdv import (
    DistributionalHDV,
    UncertaintyParams,
    random_distributional,
    distributional_bind,
    distributional_unbind,
    distributional_similarity,
    bayesian_update,
    temporal_decay,
    human_confirm,
    handle_contradiction,
    bundle_observations,
    propagate_through_edge,
)
from .graph import Edge, Node

__all__ = [
    # Version
    "__version__",
    # Core types
    "DistributionalHDV",
    "UncertaintyParams",
    "Edge",
    "Node",
    # Factory
    "random_distributional",
    # Operations
    "distributional_bind",
    "distributional_unbind",
    "distributional_similarity",
    # Uncertainty management
    "bayesian_update",
    "temporal_decay",
    "human_confirm",
    "handle_contradiction",
    "bundle_observations",
    "propagate_through_edge",
]
```

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: PASS (may need to fix integration tests)

**Step 3: Commit**

```bash
git add src/engram/__init__.py
git commit -m "feat(exports): export Node from main package

Bump version to 0.3.0 for node-centric architecture."
```

---

## Task 4: Create Hyperbolic Evaluation Escape Hatch Doc

**Files:**
- Create: `docs/HYPERBOLIC_EVALUATION.md`

**Step 1: Write the doc**

Create `docs/HYPERBOLIC_EVALUATION.md`:

```markdown
# Hyperbolic Representation Evaluation

## Status: Secondary / Under Evaluation

The hyperbolic (Poincare ball) representation is currently a **secondary** representation in Engram. The HDV (High-Dimensional Vector) is the source of truth.

## Current Role

Hyperbolic positioning provides:
- **Radius**: Abstraction level (derived from graph structure)
- **Angle**: Semantic clustering (derived from HDV similarity)
- **Cone queries**: Finding ancestors/descendants by geometric containment

## When to Remove

We should remove hyperbolic representation if:

1. **Cone queries aren't used** - If we never query "what are the ancestors of X?" geometrically, and graph traversal suffices, the hyperbolic embedding is overhead.

2. **Hybrid positioning is complex without benefit** - If computing radius from graph structure and angle from HDV similarity proves difficult to maintain or doesn't provide clear value over pure HDV similarity.

3. **Performance cost** - If maintaining consistent hyperbolic positions becomes a bottleneck.

4. **Conceptual overhead** - If the dual representation (HDV + hyperbolic) confuses more than it clarifies.

## Evaluation Criteria

Before removing, evaluate:

- [ ] Are cone queries providing value that HDV similarity + graph traversal can't?
- [ ] Is the hyperbolic hierarchy intuitive and useful for visualization?
- [ ] Does the hybrid positioning (radius from graph, angle from HDV) work in practice?

## If We Remove It

If hyperbolic representation is removed:

1. Delete `src/engram/hyperbolic/` directory
2. Remove cone query functionality
3. Rely on HDV similarity for "what's related" queries
4. Rely on graph traversal for "what's above/below" queries
5. Update any visualization to use force-directed graph layout instead

## Decision Log

- **2026-01-25**: Created as secondary representation. HDV is source of truth. Hybrid positioning planned (radius from graph, angle from HDV). Keeping escape hatch open.
```

**Step 2: Commit**

```bash
git add docs/HYPERBOLIC_EVALUATION.md
git commit -m "docs: add hyperbolic evaluation escape hatch

Documents that hyperbolic representation is secondary to HDV and may be
removed if it doesn't prove useful. Lists evaluation criteria and removal steps."
```

---

## Task 5: Update Hyperbolic Positioning to Hybrid Approach

**Files:**
- Modify: `src/engram/hyperbolic/poincare.py`
- Create: `tests/hyperbolic/test_hybrid_positioning.py`

**Step 1: Write tests for hybrid positioning**

Create `tests/hyperbolic/test_hybrid_positioning.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/hyperbolic/test_hybrid_positioning.py -v`
Expected: FAIL (compute_hybrid_position doesn't exist)

**Step 3: Implement hybrid positioning**

Add to `src/engram/hyperbolic/poincare.py`:

```python
def compute_hybrid_position(
    node: "Node",
    nodes: dict[str, "Node"],
    edges: list["Edge"],
    dim: int = 2,
    seed: int | None = None,
) -> torch.Tensor:
    """Compute hybrid hyperbolic position for a node.

    Hybrid positioning:
    - Radius: From graph structure (depth from roots) and mass (high mass = closer to origin)
    - Angle: From HDV similarity to other nodes

    Args:
        node: The node to position.
        nodes: Dictionary of all nodes by ID.
        edges: List of all edges in the graph.
        dim: Dimension of the Poincare ball (default 2 for visualization).
        seed: Optional random seed for reproducibility.

    Returns:
        Position in the Poincare ball.
    """
    from engram.graph import Node, Edge
    from engram.hdv import distributional_similarity

    gen = torch.Generator().manual_seed(seed) if seed is not None else None

    # Build adjacency: which nodes does this node point to?
    outgoing = {e.target for e in edges if e.source == node.id}
    incoming = {e.source for e in edges if e.target == node.id}

    # Compute depth: how many hops to a root (node with no outgoing edges)?
    # Use BFS from this node following outgoing edges
    depth = _compute_depth(node.id, edges, nodes)

    # Compute radius from depth and mass
    # - Deeper nodes -> larger radius
    # - Higher mass -> smaller radius (more abstract/important)
    base_radius = 0.1 + 0.15 * depth
    mass_factor = 1.0 / (1.0 + 0.1 * node.mass)  # High mass pulls toward origin
    radius = min(0.9, base_radius * mass_factor)

    # Compute angle from HDV similarity to neighbors
    # If no neighbors, use random direction based on HDV
    if outgoing or incoming:
        # Average direction toward similar neighbors
        neighbor_ids = outgoing | incoming
        direction = _compute_direction_from_neighbors(
            node, neighbor_ids, nodes, dim, gen
        )
    else:
        # Use HDV-derived direction for isolated nodes
        direction = _hdv_to_direction(node.hdv, dim, gen)

    # Combine radius and direction
    position = direction * radius

    return project_to_poincare(position)


def _compute_depth(node_id: str, edges: list, nodes: dict) -> int:
    """Compute depth of a node (hops to nearest root via outgoing edges)."""
    visited = set()
    queue = [(node_id, 0)]
    min_depth = float('inf')

    # Build outgoing adjacency
    outgoing = {}
    for e in edges:
        if e.source not in outgoing:
            outgoing[e.source] = []
        outgoing[e.source].append(e.target)

    while queue:
        current, depth = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        # If no outgoing edges, this is a root
        if current not in outgoing or not outgoing[current]:
            min_depth = min(min_depth, depth)
            continue

        for target in outgoing[current]:
            if target not in visited and target in nodes:
                queue.append((target, depth + 1))

    return min_depth if min_depth != float('inf') else 0


def _compute_direction_from_neighbors(
    node: "Node",
    neighbor_ids: set[str],
    nodes: dict[str, "Node"],
    dim: int,
    gen: torch.Generator | None,
) -> torch.Tensor:
    """Compute direction based on HDV similarity to neighbors."""
    from engram.hdv import distributional_similarity

    # Start with random base direction
    direction = torch.randn(dim, generator=gen)

    # Weight by similarity to each neighbor
    total_weight = 0.0
    weighted_direction = torch.zeros(dim)

    for nid in neighbor_ids:
        if nid not in nodes:
            continue
        neighbor = nodes[nid]
        sim, _ = distributional_similarity(node.hdv, neighbor.hdv)

        # Use neighbor's HDV to derive a direction contribution
        neighbor_dir = _hdv_to_direction(neighbor.hdv, dim, gen)
        weighted_direction += sim * neighbor_dir
        total_weight += sim

    if total_weight > 0:
        direction = weighted_direction / total_weight
    else:
        direction = _hdv_to_direction(node.hdv, dim, gen)

    # Normalize to unit vector
    norm = torch.norm(direction)
    if norm > 1e-6:
        direction = direction / norm
    else:
        direction = torch.randn(dim, generator=gen)
        direction = direction / torch.norm(direction)

    return direction


def _hdv_to_direction(hdv: "DistributionalHDV", dim: int, gen: torch.Generator | None) -> torch.Tensor:
    """Derive a direction from an HDV by projecting to lower dimension."""
    # Use first `dim` components of the HDV mean as direction seed
    # This ensures same HDV -> same direction
    if hdv.mean.shape[0] >= dim:
        raw = hdv.mean[:dim].clone()
    else:
        raw = torch.randn(dim, generator=gen)

    norm = torch.norm(raw)
    if norm > 1e-6:
        return raw / norm
    else:
        fallback = torch.randn(dim, generator=gen)
        return fallback / torch.norm(fallback)
```

**Step 4: Add imports at top of poincare.py**

Add at top after existing imports:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.graph import Node, Edge
    from engram.hdv import DistributionalHDV
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/hyperbolic/test_hybrid_positioning.py -v`
Expected: PASS

**Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/engram/hyperbolic/poincare.py tests/hyperbolic/test_hybrid_positioning.py
git commit -m "feat(hyperbolic): add hybrid positioning (radius from graph, angle from HDV)

compute_hybrid_position derives position from:
- Radius: graph depth + mass (high mass = closer to origin)
- Angle: HDV similarity to neighbors

This implements the hybrid approach where hyperbolic position is derived
from HDV (source of truth) and graph structure."
```

---

## Task 6: Write Design Document

**Files:**
- Create: `docs/NODE_CENTRIC_ARCHITECTURE.md`

**Step 1: Write the architecture doc**

Create `docs/NODE_CENTRIC_ARCHITECTURE.md`:

```markdown
# Node-Centric Architecture

## Overview

Engram uses a uniform node-centric architecture where **everything is a node**. This document describes the core design principles and their rationale.

## Core Principles

### 1. Everything is a Node

All semantic meaning lives in nodes:
- Facts and concepts
- Relationships (connecting other nodes)
- Types (connected to by instances)
- Meta-relationships (relationships about relationships)
- Agents, workflows, conditionals
- The graph's own structure

### 2. Edges are Unlabeled Arrows

Edges are pure structural primitives: `(source, target)`. No semantic labels, no confidence scores, no weights on edges.

If you need to express "dog IS_A mammal with 80% confidence":
```
[dog] → [is_a_relationship_123] → [mammal]
[is_a_relationship_123] → [IS_A_type]
[confidence_note_456] → [is_a_relationship_123]
[confidence_note_456].content = 0.8
```

This enables:
- Meta-relationships (relationships about relationships)
- Provenance tracking (who created this relationship)
- Conditional relationships
- History/versioning

### 3. HDV is Source of Truth

Each node has a Distributional HDV (High-Dimensional Vector) that encodes its identity and meaning.

**Two uniform operations:**
- `bind(A, B)` — creates associations (relationship HDV = bind of participants)
- `bundle([A, B, C])` — creates collections/superpositions

**HDV assignment:**
- Atomic nodes: random HDV
- Derived nodes: computed via bind from connected nodes

### 4. Content is Human-Readable Source

The `content` field holds human-readable information. An LLM encodes this into the HDV at ingestion. Core graph operations never parse content—they use HDV.

### 5. Hyperbolic Position is Derived

Position in the Poincare ball is secondary, derived from HDV and graph structure:
- **Radius**: From graph depth and mass (abstract = near origin)
- **Angle**: From HDV similarity (similar nodes cluster)

May be removed if not useful (see `docs/HYPERBOLIC_EVALUATION.md`).

## Node Structure

```python
Node:
    id: str                    # Unique identifier
    hdv: DistributionalHDV     # Source of truth for similarity
    energy: float              # Recency (0-1), decays fast
    mass: float                # Importance (>0), decays slow
    content: Any               # Human-readable payload
```

## Edge Structure

```python
Edge:
    source: str    # Source node ID
    target: str    # Target node ID
```

## Physics

### Energy
- Decays continuously without access
- Restored when node is accessed
- Determines "should I surface this now?"

### Mass
- Decays very slowly
- Grows when node is successfully used
- Determines "should I trust/keep this?"
- High mass = stronger gravitational pull for merging

### Uncertainty
- Encoded in HDV variance (distributional representation)
- High variance = low confidence
- Human confirmation collapses variance
- Contradictions increase variance

## Example: Dog-Mammal Relationship

```
# Nodes
[dog]                    # hdv = random, content = "A dog"
[mammal]                 # hdv = random, content = "A mammal"
[is_a_rel_1]             # hdv = bind(dog.hdv, mammal.hdv)
[IS_A_type]              # hdv = random, content = "IS_A relationship type"

# Edges (unlabeled arrows)
[dog] → [is_a_rel_1]
[is_a_rel_1] → [mammal]
[is_a_rel_1] → [IS_A_type]
```

To query "what is dog?": find nodes that dog points to through relationship nodes.
To query "what things are mammals?": find nodes that point to mammal through relationship nodes.
```

**Step 2: Commit**

```bash
git add docs/NODE_CENTRIC_ARCHITECTURE.md
git commit -m "docs: add node-centric architecture documentation

Describes the uniform primitive design where everything is a node,
edges are unlabeled arrows, and HDV bind/bundle are the compositional primitives."
```

---

## Summary

After completing all tasks:

1. **Edge** simplified to `(source, target)` only
2. **Node** class with id, hdv, energy, mass, content
3. **Hybrid positioning** for hyperbolic (radius from graph, angle from HDV)
4. **Escape hatch doc** for hyperbolic evaluation
5. **Architecture doc** explaining the design

Run final verification:
```bash
pytest tests/ -v
```
