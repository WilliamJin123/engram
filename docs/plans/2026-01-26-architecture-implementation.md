# Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the core architecture decisions from brainstorming: mass-integrated certainty, knowledge graph with proper division of labor, sleep agents, and emergent seeking strategies.

**Architecture:** HDV handles similarity/identity; graph handles structure. Mass modulates Bayesian updates for certainty resistance. Sleep agents are hardcoded processes. Seeking strategies are emergent nodes.

**Tech Stack:** Python 3.11+, PyTorch, pytest

---

## Phase 1: Mass-Integrated Certainty

### Task 1.1: Add Mass-Aware Bayesian Update

**Files:**
- Modify: `src/engram/hdv/uncertainty.py:128-166`
- Test: `tests/hdv/test_mass_integration.py` (create)

**Step 1: Write the failing test**

Create `tests/hdv/test_mass_integration.py`:

```python
"""Tests for mass-integrated Bayesian updates."""

import torch
import pytest
from engram.hdv import random_distributional, DistributionalHDV
from engram.hdv.uncertainty import bayesian_update_with_mass, UncertaintyParams


class TestMassIntegratedUpdates:
    """Verify mass modulates belief updates."""

    def test_high_mass_resists_change(self, dim):
        """High-mass nodes should update less than low-mass nodes.

        Scenario: Same observation applied to high-mass and low-mass priors.
        High-mass prior should move less toward the observation.
        """
        # Create identical priors
        prior_low_mass = random_distributional(dim, initial_variance=0.5, seed=42)
        prior_high_mass = random_distributional(dim, initial_variance=0.5, seed=42)

        # Same observation
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        # Update with different mass values
        posterior_low = bayesian_update_with_mass(
            prior_low_mass, observation, obs_variance,
            mass=1.0, current_time=1.0
        )
        posterior_high = bayesian_update_with_mass(
            prior_high_mass, observation, obs_variance,
            mass=100.0, current_time=1.0
        )

        # Compute how much each moved
        movement_low = torch.norm(posterior_low.mean - prior_low_mass.mean).item()
        movement_high = torch.norm(posterior_high.mean - prior_high_mass.mean).item()

        # High mass should move less
        assert movement_high < movement_low
        # Significant difference (not just numerical noise)
        assert movement_low > movement_high * 2

    def test_zero_mass_behaves_like_original(self, dim):
        """Mass=1.0 should behave similarly to original bayesian_update.

        Ensures backward compatibility.
        """
        from engram.hdv.uncertainty import bayesian_update

        prior = random_distributional(dim, initial_variance=0.5, seed=42)
        observation = torch.randn(dim)
        obs_variance = torch.ones(dim) * 0.3

        # Original update
        posterior_original = bayesian_update(
            prior, observation, obs_variance, current_time=1.0
        )

        # Mass-aware update with mass=1.0
        posterior_mass = bayesian_update_with_mass(
            prior, observation, obs_variance, mass=1.0, current_time=1.0
        )

        # Should be very similar
        assert torch.allclose(posterior_original.mean, posterior_mass.mean, atol=1e-5)

    def test_mass_affects_variance_reduction(self, dim):
        """High mass should also reduce variance less aggressively."""
        prior = random_distributional(dim, initial_variance=0.5, seed=42)
        observation = prior.mean + torch.randn(dim) * 0.1
        obs_variance = torch.ones(dim) * 0.2

        posterior_low = bayesian_update_with_mass(
            prior, observation, obs_variance, mass=1.0, current_time=1.0
        )
        posterior_high = bayesian_update_with_mass(
            prior, observation, obs_variance, mass=50.0, current_time=1.0
        )

        # Both should reduce variance
        assert posterior_low.variance.mean() < prior.variance.mean()
        assert posterior_high.variance.mean() < prior.variance.mean()

        # High mass reduces less (stays more uncertain about new info)
        assert posterior_high.variance.mean() > posterior_low.variance.mean()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hdv/test_mass_integration.py -v`
Expected: FAIL with "cannot import name 'bayesian_update_with_mass'"

**Step 3: Write minimal implementation**

Add to `src/engram/hdv/uncertainty.py` after line 166:

```python
def bayesian_update_with_mass(
    prior: DistributionalHDV,
    observation: torch.Tensor,
    obs_variance: torch.Tensor,
    mass: float,
    current_time: float,
) -> DistributionalHDV:
    """Perform mass-modulated Bayesian update.

    High-mass nodes resist change more than low-mass nodes.
    This implements "certainty resistance" where well-established
    beliefs are harder to shift.

    The mass modulates the effective prior variance:
    - High mass = artificially lower effective variance = trust prior more
    - Low mass = normal behavior = observation has more influence

    Args:
        prior: Current belief distribution.
        observation: New observed HDV.
        obs_variance: Variance/uncertainty of the observation.
        mass: Node mass (importance/confidence, > 0).
        current_time: Current timestamp.

    Returns:
        New DistributionalHDV representing posterior belief.
    """
    # Mass resistance: higher mass = prior is treated as more certain
    # Using log scale to prevent extreme values
    mass_resistance = 1.0 + torch.log1p(torch.tensor(mass)).item()

    # Effective prior variance is reduced by mass (prior appears more certain)
    effective_prior_var = prior.variance / mass_resistance

    # Kalman gain with mass-adjusted variance
    K = effective_prior_var / (effective_prior_var + obs_variance)

    # Update mean: high mass = smaller K = less movement
    posterior_mean = prior.mean + K * (observation - prior.mean)

    # Update variance: high mass = less reduction
    posterior_variance = (1 - K) * prior.variance

    return DistributionalHDV(
        mean=posterior_mean,
        variance=posterior_variance,
        last_accessed=current_time,
        last_updated=current_time,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hdv/test_mass_integration.py -v`
Expected: PASS

**Step 5: Export the new function**

Modify `src/engram/hdv/__init__.py` to add export:

```python
from .uncertainty import (
    # ... existing imports ...
    bayesian_update_with_mass,
)

__all__ = [
    # ... existing exports ...
    "bayesian_update_with_mass",
]
```

**Step 6: Commit**

```bash
git add tests/hdv/test_mass_integration.py src/engram/hdv/uncertainty.py src/engram/hdv/__init__.py
git commit -m "feat(hdv): add mass-integrated Bayesian update

High-mass nodes now resist belief changes more than low-mass nodes.
This implements certainty resistance where established beliefs are
harder to shift, matching INTUITION.md requirements."
```

---

## Phase 2: Knowledge Graph Container

### Task 2.1: Create Graph Container Class

**Files:**
- Create: `src/engram/graph/knowledge_graph.py`
- Test: `tests/graph/test_knowledge_graph.py` (create)

**Step 1: Write the failing test**

Create `tests/graph/test_knowledge_graph.py`:

```python
"""Tests for KnowledgeGraph container."""

import torch
import pytest
from engram.graph import Node, Edge, KnowledgeGraph
from engram.hdv import random_distributional


class TestKnowledgeGraph:
    """Verify knowledge graph operations."""

    def test_add_and_retrieve_nodes(self, dim):
        """Can add nodes and retrieve them by ID."""
        graph = KnowledgeGraph()

        hdv = random_distributional(dim, seed=42)
        node = Node(id="dog", hdv=hdv, content="A dog")

        graph.add_node(node)

        retrieved = graph.get_node("dog")
        assert retrieved is not None
        assert retrieved.id == "dog"
        assert retrieved.content == "A dog"

    def test_add_and_retrieve_edges(self, dim):
        """Can add edges and query connections."""
        graph = KnowledgeGraph()

        # Add nodes
        for name in ["dog", "mammal", "animal"]:
            hdv = random_distributional(dim, seed=hash(name) % 1000)
            graph.add_node(Node(id=name, hdv=hdv))

        # Add edges
        graph.add_edge(Edge(source="dog", target="mammal"))
        graph.add_edge(Edge(source="mammal", target="animal"))

        # Query outgoing edges
        dog_targets = graph.get_targets("dog")
        assert "mammal" in dog_targets

        # Query incoming edges
        mammal_sources = graph.get_sources("mammal")
        assert "dog" in mammal_sources

    def test_similarity_search(self, dim):
        """Can find similar nodes by HDV."""
        graph = KnowledgeGraph()

        # Add some nodes
        dog_hdv = random_distributional(dim, seed=1)
        cat_hdv = random_distributional(dim, seed=2)
        car_hdv = random_distributional(dim, seed=100)

        graph.add_node(Node(id="dog", hdv=dog_hdv))
        graph.add_node(Node(id="cat", hdv=cat_hdv))
        graph.add_node(Node(id="car", hdv=car_hdv))

        # Create query similar to dog
        query_hdv = dog_hdv  # Exact match for test simplicity

        results = graph.find_similar(query_hdv, top_k=2)

        assert len(results) == 2
        assert results[0][0] == "dog"  # Most similar
        assert results[0][1] > 0.99  # High similarity

    def test_update_node(self, dim):
        """Can update existing nodes."""
        graph = KnowledgeGraph()

        hdv = random_distributional(dim, seed=42)
        node = Node(id="concept", hdv=hdv, mass=1.0)
        graph.add_node(node)

        # Update the node
        updated = node.reinforce(mass_boost=0.5)
        graph.update_node(updated)

        retrieved = graph.get_node("concept")
        assert retrieved.mass == 1.5

    def test_remove_node_and_edges(self, dim):
        """Removing a node removes associated edges."""
        graph = KnowledgeGraph()

        for name in ["a", "b", "c"]:
            hdv = random_distributional(dim, seed=hash(name) % 1000)
            graph.add_node(Node(id=name, hdv=hdv))

        graph.add_edge(Edge(source="a", target="b"))
        graph.add_edge(Edge(source="b", target="c"))

        graph.remove_node("b")

        assert graph.get_node("b") is None
        assert "b" not in graph.get_targets("a")
        assert "b" not in graph.get_sources("c")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/graph/test_knowledge_graph.py -v`
Expected: FAIL with "cannot import name 'KnowledgeGraph'"

**Step 3: Write minimal implementation**

Create `src/engram/graph/knowledge_graph.py`:

```python
"""Knowledge graph container for nodes and edges."""

from typing import Optional
from .node import Node
from .edge import Edge
from engram.hdv import distributional_similarity


class KnowledgeGraph:
    """Container for nodes and edges with query operations.

    Provides:
    - Node storage and retrieval by ID
    - Edge storage with source/target queries
    - HDV-based similarity search
    """

    def __init__(self):
        """Initialize empty graph."""
        self._nodes: dict[str, Node] = {}
        self._edges: set[Edge] = set()
        # Indexes for fast edge queries
        self._outgoing: dict[str, set[str]] = {}  # source -> {targets}
        self._incoming: dict[str, set[str]] = {}  # target -> {sources}

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self._nodes[node.id] = node
        if node.id not in self._outgoing:
            self._outgoing[node.id] = set()
        if node.id not in self._incoming:
            self._incoming[node.id] = set()

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID."""
        return self._nodes.get(node_id)

    def update_node(self, node: Node) -> None:
        """Update an existing node."""
        if node.id not in self._nodes:
            raise KeyError(f"Node {node.id} not in graph")
        self._nodes[node.id] = node

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            return

        # Remove edges
        for target in list(self._outgoing.get(node_id, [])):
            self._edges.discard(Edge(source=node_id, target=target))
            self._incoming[target].discard(node_id)

        for source in list(self._incoming.get(node_id, [])):
            self._edges.discard(Edge(source=source, target=node_id))
            self._outgoing[source].discard(node_id)

        # Remove from indexes
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)

        # Remove node
        del self._nodes[node_id]

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise KeyError(f"Both nodes must exist: {edge.source}, {edge.target}")

        self._edges.add(edge)
        self._outgoing[edge.source].add(edge.target)
        self._incoming[edge.target].add(edge.source)

    def get_targets(self, source_id: str) -> set[str]:
        """Get all nodes that source points to."""
        return self._outgoing.get(source_id, set()).copy()

    def get_sources(self, target_id: str) -> set[str]:
        """Get all nodes that point to target."""
        return self._incoming.get(target_id, set()).copy()

    def find_similar(
        self,
        query_hdv,
        top_k: int = 10,
        min_energy: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Find nodes most similar to query HDV.

        Args:
            query_hdv: DistributionalHDV to compare against.
            top_k: Maximum number of results.
            min_energy: Minimum energy threshold for results.

        Returns:
            List of (node_id, similarity) tuples, sorted by similarity descending.
        """
        results = []

        for node_id, node in self._nodes.items():
            if node.energy < min_energy:
                continue

            sim, _ = distributional_similarity(query_hdv, node.hdv)
            results.append((node_id, sim))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return len(self._edges)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/graph/test_knowledge_graph.py -v`
Expected: PASS

**Step 5: Export from graph module**

Modify `src/engram/graph/__init__.py`:

```python
"""Graph components for Engram."""

from .edge import Edge
from .node import Node
from .knowledge_graph import KnowledgeGraph

__all__ = ["Edge", "Node", "KnowledgeGraph"]
```

**Step 6: Commit**

```bash
git add src/engram/graph/knowledge_graph.py tests/graph/test_knowledge_graph.py src/engram/graph/__init__.py
git commit -m "feat(graph): add KnowledgeGraph container

Provides node/edge storage with:
- O(1) node lookup by ID
- O(1) edge queries (sources/targets)
- HDV similarity search"
```

---

## Phase 3: Sleep Agent Framework

### Task 3.1: Create Sleep Agent Base and Clusterer

**Files:**
- Create: `src/engram/sleep/__init__.py`
- Create: `src/engram/sleep/base.py`
- Create: `src/engram/sleep/clusterer.py`
- Test: `tests/sleep/test_clusterer.py` (create)
- Test: `tests/sleep/__init__.py` (create)

**Step 1: Write the failing test**

Create `tests/sleep/__init__.py` (empty file).

Create `tests/sleep/test_clusterer.py`:

```python
"""Tests for Clusterer sleep agent."""

import torch
import pytest
from engram.graph import Node, KnowledgeGraph
from engram.hdv import random_distributional, bundle
from engram.sleep import Clusterer


class TestClusterer:
    """Verify clusterer finds similar node groups."""

    def test_finds_obvious_clusters(self, dim):
        """Clusterer identifies groups of similar nodes.

        Scenario: Create two distinct clusters of nodes, verify
        clusterer finds them.
        """
        graph = KnowledgeGraph()

        # Cluster 1: Animals (similar HDVs)
        base_animal = random_distributional(dim, seed=1)
        for i, name in enumerate(["dog", "cat", "horse"]):
            # Small perturbation from base
            hdv = random_distributional(dim, seed=1)  # Same seed = same HDV
            hdv.mean = hdv.mean + torch.randn(dim) * 0.1
            graph.add_node(Node(id=name, hdv=hdv))

        # Cluster 2: Vehicles (different base, similar to each other)
        for i, name in enumerate(["car", "truck", "bike"]):
            hdv = random_distributional(dim, seed=100)  # Different base
            hdv.mean = hdv.mean + torch.randn(dim) * 0.1
            graph.add_node(Node(id=name, hdv=hdv))

        # Run clusterer
        clusterer = Clusterer(similarity_threshold=0.7)
        clusters = clusterer.find_clusters(graph)

        # Should find at least 2 clusters
        assert len(clusters) >= 2

        # Animals should be in same cluster
        animal_cluster = None
        for cluster in clusters:
            if "dog" in cluster:
                animal_cluster = cluster
                break

        assert animal_cluster is not None
        assert "cat" in animal_cluster or "horse" in animal_cluster

    def test_no_clusters_when_dissimilar(self, dim):
        """No clusters found when all nodes are dissimilar."""
        graph = KnowledgeGraph()

        # All different seeds = random, dissimilar HDVs
        for i in range(5):
            hdv = random_distributional(dim, seed=i * 100)
            graph.add_node(Node(id=f"node_{i}", hdv=hdv))

        clusterer = Clusterer(similarity_threshold=0.9)
        clusters = clusterer.find_clusters(graph)

        # Each node in its own cluster or no clusters
        for cluster in clusters:
            assert len(cluster) == 1

    def test_respects_energy_threshold(self, dim):
        """Low-energy nodes can be excluded from clustering."""
        graph = KnowledgeGraph()

        hdv = random_distributional(dim, seed=1)
        graph.add_node(Node(id="active", hdv=hdv, energy=1.0))
        graph.add_node(Node(id="dormant", hdv=hdv, energy=0.1))

        clusterer = Clusterer(similarity_threshold=0.9, min_energy=0.5)
        clusters = clusterer.find_clusters(graph)

        # Dormant should be excluded
        all_nodes = set()
        for cluster in clusters:
            all_nodes.update(cluster)

        assert "active" in all_nodes
        assert "dormant" not in all_nodes
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sleep/test_clusterer.py -v`
Expected: FAIL with "No module named 'engram.sleep'"

**Step 3: Write minimal implementation**

Create `src/engram/sleep/__init__.py`:

```python
"""Sleep agents for graph reorganization.

Sleep agents are hardcoded processes (DNA-level) that operate on the
knowledge graph during consolidation. They are not nodes themselves
and don't follow node physics.
"""

from .clusterer import Clusterer

__all__ = ["Clusterer"]
```

Create `src/engram/sleep/base.py`:

```python
"""Base class for sleep agents."""

from abc import ABC, abstractmethod
from engram.graph import KnowledgeGraph


class SleepAgent(ABC):
    """Base class for hardcoded sleep processes.

    Sleep agents operate on the graph but are not nodes themselves.
    They are part of the system's "DNA" - fundamental processes that
    exist before the graph has any content.
    """

    @abstractmethod
    def run(self, graph: KnowledgeGraph) -> dict:
        """Execute this sleep agent on the graph.

        Args:
            graph: The knowledge graph to process.

        Returns:
            Dictionary with results/statistics of the operation.
        """
        pass
```

Create `src/engram/sleep/clusterer.py`:

```python
"""Clusterer sleep agent for finding similar node groups."""

from engram.graph import KnowledgeGraph
from engram.hdv import distributional_similarity
from .base import SleepAgent


class Clusterer(SleepAgent):
    """Finds clusters of similar nodes in the graph.

    Uses simple greedy clustering based on HDV similarity.
    This is a hardcoded process - part of the system's DNA.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_energy: float = 0.0,
    ):
        """Initialize clusterer.

        Args:
            similarity_threshold: Minimum similarity to be in same cluster.
            min_energy: Minimum energy for nodes to be considered.
        """
        self.similarity_threshold = similarity_threshold
        self.min_energy = min_energy

    def run(self, graph: KnowledgeGraph) -> dict:
        """Run clustering and return results."""
        clusters = self.find_clusters(graph)
        return {
            "clusters": clusters,
            "cluster_count": len(clusters),
        }

    def find_clusters(self, graph: KnowledgeGraph) -> list[set[str]]:
        """Find clusters of similar nodes.

        Uses greedy clustering: pick a node, find all similar nodes,
        form a cluster, repeat with remaining nodes.

        Args:
            graph: Knowledge graph to cluster.

        Returns:
            List of clusters, each cluster is a set of node IDs.
        """
        # Get eligible nodes
        eligible = []
        for node_id in self._get_all_node_ids(graph):
            node = graph.get_node(node_id)
            if node and node.energy >= self.min_energy:
                eligible.append(node_id)

        if not eligible:
            return []

        clustered = set()
        clusters = []

        for node_id in eligible:
            if node_id in clustered:
                continue

            # Start new cluster
            cluster = {node_id}
            node = graph.get_node(node_id)

            # Find similar nodes
            for other_id in eligible:
                if other_id in clustered or other_id == node_id:
                    continue

                other = graph.get_node(other_id)
                sim, _ = distributional_similarity(node.hdv, other.hdv)

                if sim >= self.similarity_threshold:
                    cluster.add(other_id)

            # Only add clusters with multiple members
            if len(cluster) > 1:
                clusters.append(cluster)
                clustered.update(cluster)
            else:
                # Single-node cluster
                clusters.append(cluster)
                clustered.add(node_id)

        return clusters

    def _get_all_node_ids(self, graph: KnowledgeGraph) -> list[str]:
        """Get all node IDs from graph."""
        # Access internal state - could add proper iteration to KnowledgeGraph
        return list(graph._nodes.keys())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/sleep/test_clusterer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/engram/sleep/ tests/sleep/
git commit -m "feat(sleep): add Clusterer sleep agent

First sleep agent implementation. Finds clusters of similar nodes
based on HDV similarity. Sleep agents are hardcoded DNA-level
processes, not nodes."
```

---

### Task 3.2: Create Abstractor Sleep Agent

**Files:**
- Create: `src/engram/sleep/abstractor.py`
- Modify: `src/engram/sleep/__init__.py`
- Test: `tests/sleep/test_abstractor.py` (create)

**Step 1: Write the failing test**

Create `tests/sleep/test_abstractor.py`:

```python
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
        base_hdv = random_distributional(dim, seed=1)
        instances = []
        for i, name in enumerate(["dog1", "dog2", "dog3"]):
            hdv = random_distributional(dim, seed=1)
            # Small variation
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

        # Concept HDV should be similar to all instances
        for instance_id in instances:
            instance = graph.get_node(instance_id)
            sim, _ = distributional_similarity(concept.hdv, instance.hdv)
            assert sim > 0.5  # Reasonably similar

        # Edges should link instances to concept
        for instance_id in instances:
            targets = graph.get_targets(instance_id)
            assert "dog_concept" in targets

    def test_concept_starts_with_low_mass(self, dim):
        """Newly created concepts start with low mass."""
        graph = KnowledgeGraph()

        for i in range(3):
            hdv = random_distributional(dim, seed=1)
            graph.add_node(Node(id=f"node_{i}", hdv=hdv))

        abstractor = Abstractor(initial_concept_mass=0.5)
        abstractor.abstract_cluster(
            graph,
            {"node_0", "node_1", "node_2"},
            concept_id="concept"
        )

        concept = graph.get_node("concept")
        assert concept.mass == 0.5

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sleep/test_abstractor.py -v`
Expected: FAIL with "cannot import name 'Abstractor'"

**Step 3: Write minimal implementation**

Create `src/engram/sleep/abstractor.py`:

```python
"""Abstractor sleep agent for creating concept nodes from clusters."""

import torch
from engram.graph import Node, Edge, KnowledgeGraph
from engram.hdv import DistributionalHDV
from .base import SleepAgent


class Abstractor(SleepAgent):
    """Creates abstract concept nodes from clusters of similar nodes.

    When given a cluster of similar instances, creates a parent concept
    node with HDV that is the bundle of all instances, and links
    instances to the concept.
    """

    def __init__(
        self,
        initial_concept_mass: float = 0.5,
        initial_concept_energy: float = 1.0,
    ):
        """Initialize abstractor.

        Args:
            initial_concept_mass: Starting mass for new concepts.
            initial_concept_energy: Starting energy for new concepts.
        """
        self.initial_concept_mass = initial_concept_mass
        self.initial_concept_energy = initial_concept_energy

    def run(self, graph: KnowledgeGraph) -> dict:
        """Run abstractor - requires clusters to be provided externally."""
        return {"message": "Use abstract_cluster() with specific clusters"}

    def abstract_cluster(
        self,
        graph: KnowledgeGraph,
        cluster: set[str],
        concept_id: str,
        concept_content: str = None,
    ) -> dict:
        """Create a concept node from a cluster of instances.

        Args:
            graph: Knowledge graph to modify.
            cluster: Set of node IDs that form the cluster.
            concept_id: ID for the new concept node.
            concept_content: Optional human-readable content.

        Returns:
            Dictionary with creation results.
        """
        if not cluster:
            return {"error": "Empty cluster"}

        # Gather instance HDVs
        instance_hdvs = []
        for node_id in cluster:
            node = graph.get_node(node_id)
            if node:
                instance_hdvs.append(node.hdv)

        if not instance_hdvs:
            return {"error": "No valid instances"}

        # Create concept HDV by bundling instance HDVs
        concept_hdv = self._bundle_hdvs(instance_hdvs)

        # Create concept node
        concept_node = Node(
            id=concept_id,
            hdv=concept_hdv,
            energy=self.initial_concept_energy,
            mass=self.initial_concept_mass,
            content=concept_content,
        )

        graph.add_node(concept_node)

        # Create edges from instances to concept
        for node_id in cluster:
            if graph.get_node(node_id):
                graph.add_edge(Edge(source=node_id, target=concept_id))

        return {
            "concept_id": concept_id,
            "instance_count": len(cluster),
        }

    def _bundle_hdvs(self, hdvs: list[DistributionalHDV]) -> DistributionalHDV:
        """Bundle multiple HDVs into one concept HDV.

        Uses precision-weighted averaging for the mean and
        computes combined variance.
        """
        if len(hdvs) == 1:
            return hdvs[0]

        # Stack means and variances
        means = torch.stack([h.mean for h in hdvs])
        variances = torch.stack([h.variance for h in hdvs])

        # Precision-weighted mean
        eps = 1e-10
        precisions = 1.0 / (variances + eps)
        total_precision = precisions.sum(dim=0)

        weighted_mean = (precisions * means).sum(dim=0) / total_precision

        # Combined variance (inverse of total precision)
        combined_variance = 1.0 / total_precision

        # Get latest timestamps
        last_accessed = max(h.last_accessed for h in hdvs)
        last_updated = max(h.last_updated for h in hdvs)

        return DistributionalHDV(
            mean=weighted_mean,
            variance=combined_variance,
            last_accessed=last_accessed,
            last_updated=last_updated,
        )
```

**Step 4: Update exports**

Modify `src/engram/sleep/__init__.py`:

```python
"""Sleep agents for graph reorganization.

Sleep agents are hardcoded processes (DNA-level) that operate on the
knowledge graph during consolidation. They are not nodes themselves
and don't follow node physics.
"""

from .clusterer import Clusterer
from .abstractor import Abstractor

__all__ = ["Clusterer", "Abstractor"]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/sleep/test_abstractor.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/engram/sleep/abstractor.py src/engram/sleep/__init__.py tests/sleep/test_abstractor.py
git commit -m "feat(sleep): add Abstractor sleep agent

Creates concept nodes from clusters by bundling instance HDVs.
Links instances to concepts via edges. Concepts start with low
mass to allow evolution."
```

---

### Task 3.3: Create Contradiction Detector Sleep Agent

**Files:**
- Create: `src/engram/sleep/contradiction.py`
- Modify: `src/engram/sleep/__init__.py`
- Test: `tests/sleep/test_contradiction.py` (create)

**Step 1: Write the failing test**

Create `tests/sleep/test_contradiction.py`:

```python
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
        base_hdv = random_distributional(dim, seed=42)

        # Create two nodes with opposing means
        hdv1 = random_distributional(dim, seed=42)
        hdv2 = random_distributional(dim, seed=42)
        hdv2.mean = -hdv2.mean  # Opposite

        graph.add_node(Node(id="earth_round", hdv=hdv1, content="Earth is round"))
        graph.add_node(Node(id="earth_flat", hdv=hdv2, content="Earth is flat"))

        detector = ContradictionDetector(
            similarity_threshold=0.3,  # Topic similarity
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sleep/test_contradiction.py -v`
Expected: FAIL with "cannot import name 'ContradictionDetector'"

**Step 3: Write minimal implementation**

Create `src/engram/sleep/contradiction.py`:

```python
"""ContradictionDetector sleep agent for finding conflicting beliefs."""

import uuid
import torch
from engram.graph import Node, Edge, KnowledgeGraph
from engram.hdv import distributional_similarity, random_distributional
from .base import SleepAgent


class ContradictionDetector(SleepAgent):
    """Detects contradictory beliefs and creates uncertainty nodes.

    Contradiction is defined as: nodes with similar topic (high structural
    similarity considering variance) but opposing content (negative mean
    correlation).

    Creates uncertainty nodes that:
    - Have high energy (demand attention)
    - Link to contradicting nodes
    - Represent "I need to resolve this"
    """

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        opposition_threshold: float = -0.3,
        min_mass: float = 0.1,
    ):
        """Initialize detector.

        Args:
            similarity_threshold: Min distributional similarity to compare.
            opposition_threshold: Max mean correlation to count as opposing.
            min_mass: Minimum mass for nodes to be considered.
        """
        self.similarity_threshold = similarity_threshold
        self.opposition_threshold = opposition_threshold
        self.min_mass = min_mass

    def run(self, graph: KnowledgeGraph) -> dict:
        """Find contradictions and create uncertainty nodes."""
        contradictions = self.find_contradictions(graph)

        uncertainty_nodes = []
        for contradiction in contradictions:
            node_id = self._create_uncertainty_node(graph, contradiction)
            if node_id:
                uncertainty_nodes.append(node_id)

        return {
            "contradictions": contradictions,
            "uncertainty_nodes": uncertainty_nodes,
        }

    def find_contradictions(self, graph: KnowledgeGraph) -> list[dict]:
        """Find pairs of contradicting nodes.

        Returns:
            List of contradiction dicts with 'nodes' and 'severity'.
        """
        contradictions = []
        node_ids = list(graph._nodes.keys())

        # Compare all pairs
        for i, id_a in enumerate(node_ids):
            node_a = graph.get_node(id_a)
            if not node_a or node_a.mass < self.min_mass:
                continue

            for id_b in node_ids[i + 1:]:
                node_b = graph.get_node(id_b)
                if not node_b or node_b.mass < self.min_mass:
                    continue

                # Check distributional similarity (are they about same topic?)
                sim, _ = distributional_similarity(node_a.hdv, node_b.hdv)

                if sim < self.similarity_threshold:
                    continue  # Different topics

                # Check mean correlation (are they opposing?)
                mean_corr = self._mean_correlation(node_a.hdv.mean, node_b.hdv.mean)

                if mean_corr <= self.opposition_threshold:
                    # Found contradiction
                    severity = abs(mean_corr) * sim  # Worse if similar AND opposing
                    contradictions.append({
                        "nodes": {id_a, id_b},
                        "severity": severity,
                        "similarity": sim,
                        "opposition": mean_corr,
                    })

        return contradictions

    def _mean_correlation(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute correlation between two mean vectors."""
        norm_a = torch.norm(a)
        norm_b = torch.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return (torch.dot(a, b) / (norm_a * norm_b)).item()

    def _create_uncertainty_node(
        self,
        graph: KnowledgeGraph,
        contradiction: dict
    ) -> str:
        """Create an uncertainty node for a contradiction."""
        node_ids = list(contradiction["nodes"])

        # Generate unique ID
        uncertainty_id = f"uncertainty_{uuid.uuid4().hex[:8]}"

        # Get first node's HDV dimension
        first_node = graph.get_node(node_ids[0])
        dim = first_node.hdv.dim

        # Create HDV with high variance (uncertain!)
        hdv = random_distributional(
            dim,
            initial_variance=1.5,  # High uncertainty
            current_time=0.0,
        )

        # Create node with high energy (demands attention)
        uncertainty_node = Node(
            id=uncertainty_id,
            hdv=hdv,
            energy=1.0,  # Maximum energy
            mass=0.1,  # Low mass (not established yet)
            content=f"Contradiction between {node_ids}",
        )

        graph.add_node(uncertainty_node)

        # Link contradicting nodes to uncertainty node
        for node_id in node_ids:
            graph.add_edge(Edge(source=node_id, target=uncertainty_id))

        return uncertainty_id
```

**Step 4: Update exports**

Modify `src/engram/sleep/__init__.py`:

```python
"""Sleep agents for graph reorganization.

Sleep agents are hardcoded processes (DNA-level) that operate on the
knowledge graph during consolidation. They are not nodes themselves
and don't follow node physics.
"""

from .clusterer import Clusterer
from .abstractor import Abstractor
from .contradiction import ContradictionDetector

__all__ = ["Clusterer", "Abstractor", "ContradictionDetector"]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/sleep/test_contradiction.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/engram/sleep/contradiction.py src/engram/sleep/__init__.py tests/sleep/test_contradiction.py
git commit -m "feat(sleep): add ContradictionDetector sleep agent

Finds nodes with similar topics but opposing content.
Creates high-energy uncertainty nodes that demand resolution.
Links uncertainty nodes to contradicting beliefs."
```

---

## Phase 4: Curiosity Drive (Hardcoded)

### Task 4.1: Create Curiosity Drive

**Files:**
- Create: `src/engram/drive/__init__.py`
- Create: `src/engram/drive/curiosity.py`
- Test: `tests/drive/__init__.py` (create)
- Test: `tests/drive/test_curiosity.py` (create)

**Step 1: Write the failing test**

Create `tests/drive/__init__.py` (empty).

Create `tests/drive/test_curiosity.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/drive/test_curiosity.py -v`
Expected: FAIL with "No module named 'engram.drive'"

**Step 3: Write minimal implementation**

Create `src/engram/drive/__init__.py`:

```python
"""Hardcoded drives that motivate system behavior.

Drives are DNA-level processes that create motivation/energy for
certain behaviors. They are not learnable or overridable.
"""

from .curiosity import CuriosityDrive

__all__ = ["CuriosityDrive"]
```

Create `src/engram/drive/curiosity.py`:

```python
"""Curiosity drive - hardcoded motivation to resolve uncertainty."""

from engram.graph import KnowledgeGraph


class CuriosityDrive:
    """Hardcoded drive that identifies what needs resolution.

    The curiosity drive is DNA-level - it's not a node, can't be
    modified, and always operates. It:

    1. Monitors for high-energy uncertainty nodes
    2. Prioritizes by importance (mass of related nodes)
    3. Signals when active seeking should occur

    The STRATEGIES for resolving uncertainty are emergent (learned
    nodes), but the DRIVE itself is hardcoded.
    """

    def __init__(
        self,
        energy_threshold: float = 0.5,
        variance_threshold: float = 1.0,
        seek_threshold: float = 5.0,
    ):
        """Initialize curiosity drive.

        Args:
            energy_threshold: Min energy for uncertainty to surface.
            variance_threshold: Min variance to count as uncertain.
            seek_threshold: Priority score above which active seeking triggers.
        """
        self.energy_threshold = energy_threshold
        self.variance_threshold = variance_threshold
        self.seek_threshold = seek_threshold

    def get_pending_uncertainties(
        self,
        graph: KnowledgeGraph,
        ranked: bool = False,
    ) -> list[str]:
        """Get uncertainty nodes that need resolution.

        Args:
            graph: Knowledge graph to scan.
            ranked: If True, return sorted by priority (highest first).

        Returns:
            List of uncertainty node IDs.
        """
        uncertainties = []

        for node_id in graph._nodes.keys():
            node = graph.get_node(node_id)
            if not node:
                continue

            # Check if this is an uncertainty node
            if not self._is_uncertainty_node(node):
                continue

            # Check energy threshold
            if node.energy < self.energy_threshold:
                continue

            priority = self._compute_priority(node, graph)
            uncertainties.append((node_id, priority))

        if ranked:
            uncertainties.sort(key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in uncertainties]

        return [node_id for node_id, _ in uncertainties]

    def should_seek(self, graph: KnowledgeGraph, uncertainty_id: str) -> bool:
        """Determine if active seeking should be triggered.

        Active seeking means the system should autonomously try to
        resolve this uncertainty (vs passive waiting for info).

        Args:
            graph: Knowledge graph.
            uncertainty_id: ID of uncertainty node to evaluate.

        Returns:
            True if active seeking is warranted.
        """
        node = graph.get_node(uncertainty_id)
        if not node:
            return False

        priority = self._compute_priority(node, graph)
        return priority >= self.seek_threshold

    def _is_uncertainty_node(self, node) -> bool:
        """Check if a node represents uncertainty.

        Heuristics:
        - ID starts with 'uncertainty_'
        - OR has high average variance
        """
        if node.id.startswith("uncertainty_"):
            return True

        avg_variance = node.hdv.variance.mean().item()
        return avg_variance >= self.variance_threshold

    def _compute_priority(self, node, graph: KnowledgeGraph) -> float:
        """Compute priority score for an uncertainty.

        Priority = energy * mass * (1 + connected_mass)

        Higher energy = more urgent
        Higher mass = more important
        Connected to high-mass nodes = more important
        """
        # Base priority
        priority = node.energy * node.mass

        # Add mass of connected nodes
        connected_mass = 0.0
        sources = graph.get_sources(node.id)
        for source_id in sources:
            source = graph.get_node(source_id)
            if source:
                connected_mass += source.mass

        return priority * (1 + connected_mass)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/drive/test_curiosity.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/engram/drive/ tests/drive/
git commit -m "feat(drive): add CuriosityDrive

Hardcoded drive that identifies uncertainties needing resolution.
Prioritizes by energy and mass. Signals when active seeking is
warranted. Strategies for resolution are separate (emergent)."
```

---

## Phase 5: Seeking Strategy Nodes (Emergent)

### Task 5.1: Create Strategy Node Type and Matcher

**Files:**
- Create: `src/engram/strategy/__init__.py`
- Create: `src/engram/strategy/base.py`
- Create: `src/engram/strategy/matcher.py`
- Test: `tests/strategy/__init__.py` (create)
- Test: `tests/strategy/test_matcher.py` (create)

**Step 1: Write the failing test**

Create `tests/strategy/__init__.py` (empty).

Create `tests/strategy/test_matcher.py`:

```python
"""Tests for strategy matching."""

import torch
import pytest
from engram.graph import Node, Edge, KnowledgeGraph
from engram.hdv import random_distributional
from engram.strategy import StrategyMatcher, create_strategy_node


class TestStrategyMatcher:
    """Verify strategy matching finds appropriate strategies."""

    def test_matches_strategy_by_hdv_similarity(self, dim):
        """Finds strategies with similar HDV to uncertainty.

        Scenario: Strategy for 'factual questions' should match
        an uncertainty about a factual matter.
        """
        graph = KnowledgeGraph()

        # Create a strategy for factual questions
        fact_hdv = random_distributional(dim, seed=42)
        strategy = create_strategy_node(
            node_id="strategy_web_search",
            hdv=fact_hdv,
            strategy_type="web_search",
            description="Search web for factual answers",
        )
        graph.add_node(strategy)

        # Create uncertainty with similar HDV (same seed = similar topic)
        uncertainty_hdv = random_distributional(dim, seed=42)
        uncertainty = Node(
            id="uncertainty_fact",
            hdv=uncertainty_hdv,
            content="What is the population of Tokyo?"
        )
        graph.add_node(uncertainty)

        # Match
        matcher = StrategyMatcher()
        matches = matcher.find_strategies(graph, uncertainty)

        assert len(matches) > 0
        assert matches[0]["strategy_id"] == "strategy_web_search"

    def test_prefers_high_mass_strategies(self, dim):
        """Higher mass strategies rank higher (more trusted)."""
        graph = KnowledgeGraph()

        # Two strategies for same topic, different mass
        hdv = random_distributional(dim, seed=1)

        low_mass = create_strategy_node(
            node_id="strategy_low",
            hdv=hdv,
            strategy_type="ask_random",
            mass=0.5,
        )
        high_mass = create_strategy_node(
            node_id="strategy_high",
            hdv=hdv,
            strategy_type="ask_expert",
            mass=5.0,
        )

        graph.add_node(low_mass)
        graph.add_node(high_mass)

        # Uncertainty
        uncertainty = Node(id="uncertainty", hdv=hdv)
        graph.add_node(uncertainty)

        matcher = StrategyMatcher()
        matches = matcher.find_strategies(graph, uncertainty)

        # High mass should rank first
        assert matches[0]["strategy_id"] == "strategy_high"

    def test_no_match_for_dissimilar(self, dim):
        """No strategies returned if none match."""
        graph = KnowledgeGraph()

        # Strategy for one topic
        hdv1 = random_distributional(dim, seed=1)
        strategy = create_strategy_node(
            node_id="strategy_cooking",
            hdv=hdv1,
            strategy_type="check_recipe",
        )
        graph.add_node(strategy)

        # Uncertainty about completely different topic
        hdv2 = random_distributional(dim, seed=999)
        uncertainty = Node(id="uncertainty_math", hdv=hdv2)
        graph.add_node(uncertainty)

        matcher = StrategyMatcher(min_similarity=0.5)
        matches = matcher.find_strategies(graph, uncertainty)

        assert len(matches) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/strategy/test_matcher.py -v`
Expected: FAIL with "No module named 'engram.strategy'"

**Step 3: Write minimal implementation**

Create `src/engram/strategy/__init__.py`:

```python
"""Emergent seeking strategies.

Strategies are NODES in the graph - they follow node physics
(decay, reinforce, merge). Unlike sleep agents, strategies are
learned from experience and can be abstracted.
"""

from .base import create_strategy_node, is_strategy_node
from .matcher import StrategyMatcher

__all__ = ["create_strategy_node", "is_strategy_node", "StrategyMatcher"]
```

Create `src/engram/strategy/base.py`:

```python
"""Base utilities for strategy nodes."""

from engram.graph import Node
from engram.hdv import DistributionalHDV


def create_strategy_node(
    node_id: str,
    hdv: DistributionalHDV,
    strategy_type: str,
    description: str = None,
    mass: float = 1.0,
    energy: float = 1.0,
) -> Node:
    """Create a strategy node.

    Strategy nodes are regular nodes with specific content structure.
    They follow normal node physics (decay, reinforce).

    Args:
        node_id: Unique ID for the strategy.
        hdv: HDV representing what kinds of uncertainties this applies to.
        strategy_type: Type of strategy (e.g., 'web_search', 'ask_human').
        description: Human-readable description.
        mass: Initial mass (trustworthiness).
        energy: Initial energy.

    Returns:
        A Node configured as a strategy.
    """
    content = {
        "type": "strategy",
        "strategy_type": strategy_type,
        "description": description,
    }

    return Node(
        id=node_id,
        hdv=hdv,
        mass=mass,
        energy=energy,
        content=content,
    )


def is_strategy_node(node: Node) -> bool:
    """Check if a node is a strategy node."""
    if not isinstance(node.content, dict):
        return False
    return node.content.get("type") == "strategy"


def get_strategy_type(node: Node) -> str:
    """Get the strategy type from a strategy node."""
    if not is_strategy_node(node):
        return None
    return node.content.get("strategy_type")
```

Create `src/engram/strategy/matcher.py`:

```python
"""Strategy matching for uncertainties."""

from engram.graph import Node, KnowledgeGraph
from engram.hdv import distributional_similarity
from .base import is_strategy_node, get_strategy_type


class StrategyMatcher:
    """Matches uncertainties to appropriate seeking strategies.

    Uses HDV similarity to find strategies that apply to the
    same kind of uncertainty, then ranks by mass (trustworthiness).
    """

    def __init__(
        self,
        min_similarity: float = 0.3,
        min_energy: float = 0.1,
    ):
        """Initialize matcher.

        Args:
            min_similarity: Minimum HDV similarity to consider a match.
            min_energy: Minimum strategy energy to consider.
        """
        self.min_similarity = min_similarity
        self.min_energy = min_energy

    def find_strategies(
        self,
        graph: KnowledgeGraph,
        uncertainty: Node,
        top_k: int = 5,
    ) -> list[dict]:
        """Find strategies that match an uncertainty.

        Args:
            graph: Knowledge graph containing strategies.
            uncertainty: The uncertainty node to find strategies for.
            top_k: Maximum number of strategies to return.

        Returns:
            List of matches, sorted by score (highest first).
            Each match is dict with strategy_id, similarity, score, strategy_type.
        """
        matches = []

        for node_id in graph._nodes.keys():
            node = graph.get_node(node_id)
            if not node or not is_strategy_node(node):
                continue

            if node.energy < self.min_energy:
                continue

            # Compute similarity
            sim, _ = distributional_similarity(uncertainty.hdv, node.hdv)

            if sim < self.min_similarity:
                continue

            # Score combines similarity and mass
            score = sim * node.mass

            matches.append({
                "strategy_id": node_id,
                "similarity": sim,
                "mass": node.mass,
                "score": score,
                "strategy_type": get_strategy_type(node),
            })

        # Sort by score descending
        matches.sort(key=lambda x: x["score"], reverse=True)

        return matches[:top_k]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/strategy/test_matcher.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/engram/strategy/ tests/strategy/
git commit -m "feat(strategy): add emergent seeking strategy nodes

Strategies are regular nodes that follow node physics.
StrategyMatcher finds strategies by HDV similarity and ranks
by mass (trustworthiness). Strategies are learned, not hardcoded."
```

---

## Phase 6: Integration and Exports

### Task 6.1: Update Main Package Exports

**Files:**
- Modify: `src/engram/__init__.py`

**Step 1: Update exports**

Modify `src/engram/__init__.py`:

```python
"""Engram: Self-organizing memory framework with epistemic uncertainty.

A memory system using distributional High-Dimensional Vectors (HDVs)
for principled uncertainty tracking in knowledge graphs.
"""

__version__ = "0.4.0"  # Bump for architecture implementation

from .hdv import (
    DistributionalHDV,
    UncertaintyParams,
    random_distributional,
    distributional_bind,
    distributional_unbind,
    distributional_similarity,
    bayesian_update,
    bayesian_update_with_mass,
    temporal_decay,
    human_confirm,
    handle_contradiction,
    bundle_observations,
    propagate_through_edge,
)
from .graph import Edge, Node, KnowledgeGraph
from .sleep import Clusterer, Abstractor, ContradictionDetector
from .drive import CuriosityDrive
from .strategy import create_strategy_node, is_strategy_node, StrategyMatcher

__all__ = [
    # Version
    "__version__",
    # Core types
    "DistributionalHDV",
    "UncertaintyParams",
    "Edge",
    "Node",
    "KnowledgeGraph",
    # Factory
    "random_distributional",
    # Operations
    "distributional_bind",
    "distributional_unbind",
    "distributional_similarity",
    # Uncertainty management
    "bayesian_update",
    "bayesian_update_with_mass",
    "temporal_decay",
    "human_confirm",
    "handle_contradiction",
    "bundle_observations",
    "propagate_through_edge",
    # Sleep agents (hardcoded)
    "Clusterer",
    "Abstractor",
    "ContradictionDetector",
    # Drives (hardcoded)
    "CuriosityDrive",
    # Strategies (emergent)
    "create_strategy_node",
    "is_strategy_node",
    "StrategyMatcher",
]
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add src/engram/__init__.py
git commit -m "feat: update main exports for v0.4.0

Exports now include:
- KnowledgeGraph container
- Sleep agents (Clusterer, Abstractor, ContradictionDetector)
- CuriosityDrive (hardcoded)
- Strategy utilities (emergent)"
```

---

## Summary

**Phase 1**: Mass-integrated certainty - `bayesian_update_with_mass`
**Phase 2**: Knowledge graph container - `KnowledgeGraph`
**Phase 3**: Sleep agents - `Clusterer`, `Abstractor`, `ContradictionDetector`
**Phase 4**: Curiosity drive - `CuriosityDrive`
**Phase 5**: Seeking strategies - `StrategyMatcher`, `create_strategy_node`
**Phase 6**: Integration and exports

Total: 6 phases, 7 tasks, ~25 commits
