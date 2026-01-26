# Plan A: Strength Model & Spreading Activation

**Date:** 2026-01-26
**Status:** Draft
**Version:** 0.4.0 → 0.5.0
**Depends on:** v0.4.0 architecture
**Enables:** Plan B (Conditionals, History, Exceptions)

## Overview

Unify mass and energy into a single **strength** property, matching biological neural networks where synaptic weights are the primary stored state. Activation becomes transient, computed during spreading activation queries.

### Key Changes

| Before (v0.4.0) | After (v0.5.0) |
|-----------------|----------------|
| `node.mass` (importance) | `node.strength` |
| `node.energy` (recency) | Computed from `node.last_accessed` |
| Energy stored & decayed | Recency computed on-demand |
| No spreading activation | `spread_activation()` function |

### Rationale

In biological neural networks:
- **Synaptic weights** are stored (long-term)
- **Activation** is computed transiently (short-term)
- **Recency effects** come from short-term potentiation, not stored "energy"

This is simpler (one property), more accurate (matches biology), and enables proper spreading activation for queries.

---

## Phase 1: Node Refactoring

### Task 1.1: Update Node Class

**File:** `src/engram/graph/node.py`

**Changes:**
- Remove `energy` property
- Rename `mass` to `strength`
- Add `last_accessed` timestamp
- Add `get_effective_strength()` method
- Update `decay()` to only decay strength
- Update `access()` to update timestamp (not boost energy)
- Keep `reinforce()` (increases strength)

**New Node class:**

```python
@dataclass
class Node:
    """A node in the knowledge graph.

    Attributes:
        id: Unique identifier for this node.
        hdv: Distributional HDV encoding this node's identity/meaning.
        strength: Importance/confidence score (>0). Decays slowly, grows with reinforcement.
        last_accessed: Timestamp of last access (for recency computation).
        content: Optional human-readable payload.
    """
    id: str
    hdv: DistributionalHDV
    strength: float = 1.0
    last_accessed: float = 0.0
    content: Any = None

    MIN_STRENGTH: float = 0.01

    def __post_init__(self):
        self.strength = max(self.MIN_STRENGTH, self.strength)

    def get_effective_strength(
        self,
        current_time: float,
        recency_tau: float = 1.0,
    ) -> float:
        """Get strength with short-term potentiation from recent access.

        Args:
            current_time: Current timestamp.
            recency_tau: Time constant for recency decay. Larger = slower decay.

        Returns:
            Effective strength including recency boost.
        """
        time_since_access = current_time - self.last_accessed
        if time_since_access < 0:
            time_since_access = 0
        recency_boost = math.exp(-time_since_access / recency_tau)
        return self.strength * (1 + recency_boost)

    def decay(self, dt: float, decay_rate: float = 0.001) -> "Node":
        """Apply temporal decay to strength.

        Args:
            dt: Time elapsed since last decay.
            decay_rate: Strength loss per time unit (very slow).

        Returns:
            New Node with decayed strength.
        """
        new_strength = self.strength * (1.0 - decay_rate * dt)
        new_strength = max(self.MIN_STRENGTH, new_strength)

        return Node(
            id=self.id,
            hdv=self.hdv,
            strength=new_strength,
            last_accessed=self.last_accessed,
            content=self.content,
        )

    def access(self, current_time: float) -> "Node":
        """Mark node as accessed (updates timestamp).

        Args:
            current_time: Current timestamp.

        Returns:
            New Node with updated last_accessed.
        """
        return Node(
            id=self.id,
            hdv=self.hdv,
            strength=self.strength,
            last_accessed=current_time,
            content=self.content,
        )

    def reinforce(self, boost: float = 0.1) -> "Node":
        """Increase strength when node is successfully used.

        Args:
            boost: Amount of strength to add.

        Returns:
            New Node with increased strength.
        """
        return Node(
            id=self.id,
            hdv=self.hdv,
            strength=self.strength + boost,
            last_accessed=self.last_accessed,
            content=self.content,
        )
```

**Test (TDD):**

```python
def test_node_strength_model():
    """Test unified strength model."""
    hdv = random_distributional(dim=100, seed=42)
    node = Node(id="test", hdv=hdv, strength=1.0, last_accessed=0.0)

    # Effective strength includes recency boost
    # At t=0, recency_boost = exp(0) = 1.0, so effective = 1.0 * (1 + 1) = 2.0
    assert node.get_effective_strength(current_time=0.0) == pytest.approx(2.0)

    # At t=1 (with tau=1), recency_boost = exp(-1) ≈ 0.368
    assert node.get_effective_strength(current_time=1.0) == pytest.approx(1.368, rel=0.01)

    # At t=10, recency_boost ≈ 0, effective ≈ strength
    assert node.get_effective_strength(current_time=10.0) == pytest.approx(1.0, rel=0.01)


def test_node_access_updates_timestamp():
    """Test that access updates last_accessed."""
    hdv = random_distributional(dim=100, seed=42)
    node = Node(id="test", hdv=hdv, strength=1.0, last_accessed=0.0)

    accessed = node.access(current_time=5.0)
    assert accessed.last_accessed == 5.0

    # Now effective strength at t=5 should be high again
    assert accessed.get_effective_strength(current_time=5.0) == pytest.approx(2.0)


def test_node_decay_only_affects_strength():
    """Test that decay only reduces strength, not last_accessed."""
    hdv = random_distributional(dim=100, seed=42)
    node = Node(id="test", hdv=hdv, strength=1.0, last_accessed=5.0)

    decayed = node.decay(dt=100, decay_rate=0.001)
    assert decayed.strength < 1.0
    assert decayed.last_accessed == 5.0  # Unchanged
```

### Task 1.2: Update Dependent Code

**Files to update:**

1. `src/engram/graph/knowledge_graph.py`
   - `find_similar()`: Change `min_energy` parameter to use effective strength
   - Update any mass/energy references

2. `src/engram/sleep/clusterer.py`
   - Change `min_energy` to `min_strength`

3. `src/engram/sleep/abstractor.py`
   - Change `initial_concept_energy` to just use strength
   - Remove energy from concept node creation

4. `src/engram/sleep/contradiction.py`
   - Update energy references in uncertainty node creation

5. `src/engram/drive/curiosity.py`
   - Change `min_energy` threshold to use effective strength

6. `src/engram/strategy/matcher.py`
   - Change `min_energy` to `min_strength`

7. `src/engram/strategy/base.py`
   - Remove energy parameter from `create_strategy_node()`

8. `src/engram/hdv/uncertainty.py`
   - `bayesian_update_with_mass()` → `bayesian_update_with_strength()`

### Task 1.3: Update Tests

Update all existing tests to use `strength` instead of `mass`/`energy`.

**Test files:**
- `tests/graph/test_node.py`
- `tests/graph/test_knowledge_graph.py`
- `tests/sleep/test_clusterer.py`
- `tests/sleep/test_abstractor.py`
- `tests/sleep/test_contradiction.py`
- `tests/drive/test_curiosity.py`
- `tests/strategy/test_matcher.py`
- `tests/hdv/test_mass_integration.py` → rename to `test_strength_integration.py`

---

## Phase 2: Spreading Activation

### Task 2.1: Create Activation Module

**New module:** `src/engram/activation/`

**Files:**
- `__init__.py`
- `spreading.py`

**`spreading.py`:**

```python
"""Spreading activation for knowledge graph queries.

Activation is TRANSIENT - computed during queries, not stored on nodes.
This matches biological neural networks where activation is computed
each forward pass, while weights (strength) are stored.
"""

import math
from engram.graph import KnowledgeGraph, Node


def spread_activation(
    graph: KnowledgeGraph,
    seed_ids: list[str] | str,
    current_time: float,
    max_depth: int = 3,
    spread_decay: float = 0.5,
    min_activation: float = 0.05,
    recency_tau: float = 1.0,
    mark_accessed: bool = True,
) -> dict[str, float]:
    """Spread activation from seed node(s) through the graph.

    Activation is transient - returned as a dict, not stored on nodes.
    Effective strength (including recency boost) determines how much
    activation each node contributes to its neighbors.

    Args:
        graph: Knowledge graph to spread through.
        seed_ids: Starting node ID(s). Can be single ID or list.
        current_time: Current timestamp for recency computation.
        max_depth: Maximum hops from seed.
        spread_decay: Activation multiplier per hop (0-1).
        min_activation: Stop spreading below this threshold.
        recency_tau: Time constant for recency boost.
        mark_accessed: If True, update last_accessed on visited nodes.

    Returns:
        Dict mapping node_id -> activation strength (transient).
    """
    if isinstance(seed_ids, str):
        seed_ids = [seed_ids]

    activations: dict[str, float] = {}
    frontier: list[tuple[str, float, int]] = []  # (node_id, activation, depth)

    # Initialize seeds
    for seed_id in seed_ids:
        seed = graph.get_node(seed_id)
        if seed:
            activations[seed_id] = 1.0
            frontier.append((seed_id, 1.0, 0))
            if mark_accessed:
                graph.update_node(seed_id, last_accessed=current_time)

    while frontier:
        node_id, activation, depth = frontier.pop(0)

        if depth >= max_depth:
            continue

        # Get all neighbors (both directions)
        outgoing = graph.get_outgoing(node_id)
        incoming = graph.get_incoming(node_id)
        neighbors = outgoing | incoming

        for neighbor_id in neighbors:
            neighbor = graph.get_node(neighbor_id)
            if not neighbor:
                continue

            # Compute spread based on neighbor's effective strength
            eff_strength = neighbor.get_effective_strength(current_time, recency_tau)

            # Strength factor: high strength = more activation received
            strength_factor = eff_strength / (eff_strength + 1)

            # Spread activation
            spread = activation * spread_decay * strength_factor

            if spread < min_activation:
                continue

            # Accumulate (take max if already activated)
            old_activation = activations.get(neighbor_id, 0)
            if spread > old_activation:
                activations[neighbor_id] = spread
                frontier.append((neighbor_id, spread, depth + 1))

                if mark_accessed:
                    graph.update_node(neighbor_id, last_accessed=current_time)

    return activations


def query(
    graph: KnowledgeGraph,
    query_id: str,
    current_time: float,
    top_k: int = 10,
    **spread_kwargs,
) -> list[tuple[Node, float]]:
    """Query the graph by spreading activation from a node.

    Args:
        graph: Knowledge graph.
        query_id: Node to start from.
        current_time: Current timestamp.
        top_k: Number of results to return.
        **spread_kwargs: Additional args for spread_activation.

    Returns:
        List of (node, activation) tuples, sorted by activation descending.
    """
    activations = spread_activation(
        graph, query_id, current_time, **spread_kwargs
    )

    # Sort by activation
    sorted_items = sorted(activations.items(), key=lambda x: -x[1])

    # Return top_k as (Node, activation) tuples
    results = []
    for node_id, activation in sorted_items[:top_k]:
        node = graph.get_node(node_id)
        if node:
            results.append((node, activation))

    return results


def multi_query(
    graph: KnowledgeGraph,
    query_ids: list[str],
    current_time: float,
    top_k: int = 10,
    **spread_kwargs,
) -> list[tuple[Node, float]]:
    """Query from multiple seed nodes simultaneously.

    Useful for queries like "what connects A and B?"

    Args:
        graph: Knowledge graph.
        query_ids: List of starting node IDs.
        current_time: Current timestamp.
        top_k: Number of results to return.
        **spread_kwargs: Additional args for spread_activation.

    Returns:
        List of (node, activation) tuples, sorted by activation descending.
    """
    activations = spread_activation(
        graph, query_ids, current_time, **spread_kwargs
    )

    sorted_items = sorted(activations.items(), key=lambda x: -x[1])

    results = []
    for node_id, activation in sorted_items[:top_k]:
        node = graph.get_node(node_id)
        if node:
            results.append((node, activation))

    return results
```

### Task 2.2: Tests for Spreading Activation

**File:** `tests/activation/test_spreading.py`

```python
"""Tests for spreading activation."""

import pytest
from engram import random_distributional, KnowledgeGraph, Node, Edge
from engram.activation import spread_activation, query


def test_spread_activation_basic():
    """Test basic activation spreading."""
    graph = KnowledgeGraph()

    # Create chain: A -> B -> C
    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="A", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="B", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="C", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_edge(Edge(source="A", target="B"))
    graph.add_edge(Edge(source="B", target="C"))

    activations = spread_activation(graph, "A", current_time=0.0, max_depth=3)

    # A should have highest activation (seed)
    assert activations["A"] == 1.0
    # B should have some activation (1 hop)
    assert activations["B"] > 0
    assert activations["B"] < 1.0
    # C should have lower activation (2 hops)
    assert activations["C"] > 0
    assert activations["C"] < activations["B"]


def test_spread_activation_strength_affects_spread():
    """Test that high-strength nodes receive more activation."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="A", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="B_weak", hdv=hdv, strength=0.1, last_accessed=0.0))
    graph.add_node(Node(id="B_strong", hdv=hdv, strength=10.0, last_accessed=0.0))
    graph.add_edge(Edge(source="A", target="B_weak"))
    graph.add_edge(Edge(source="A", target="B_strong"))

    activations = spread_activation(graph, "A", current_time=0.0)

    # Strong node should receive more activation
    assert activations["B_strong"] > activations["B_weak"]


def test_spread_activation_recency_affects_spread():
    """Test that recently accessed nodes receive more activation."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="A", hdv=hdv, strength=1.0, last_accessed=10.0))
    graph.add_node(Node(id="B_recent", hdv=hdv, strength=1.0, last_accessed=9.9))
    graph.add_node(Node(id="B_old", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_edge(Edge(source="A", target="B_recent"))
    graph.add_edge(Edge(source="A", target="B_old"))

    activations = spread_activation(graph, "A", current_time=10.0, recency_tau=1.0)

    # Recent node should receive more activation
    assert activations["B_recent"] > activations["B_old"]


def test_spread_activation_depth_limit():
    """Test that activation stops at max_depth."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    for i in range(5):
        graph.add_node(Node(id=f"N{i}", hdv=hdv, strength=5.0, last_accessed=0.0))
    for i in range(4):
        graph.add_edge(Edge(source=f"N{i}", target=f"N{i+1}"))

    activations = spread_activation(graph, "N0", current_time=0.0, max_depth=2)

    assert "N0" in activations
    assert "N1" in activations
    assert "N2" in activations
    # N3 and N4 might not be reached depending on min_activation
    # With high strength they should be, but let's test with max_depth=2
    # Actually N2 is at depth 2, so N3 would be depth 3 which exceeds max_depth=2


def test_query_returns_sorted_results():
    """Test that query returns nodes sorted by activation."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="center", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="strong", hdv=hdv, strength=10.0, last_accessed=0.0))
    graph.add_node(Node(id="weak", hdv=hdv, strength=0.5, last_accessed=0.0))
    graph.add_edge(Edge(source="center", target="strong"))
    graph.add_edge(Edge(source="center", target="weak"))

    results = query(graph, "center", current_time=0.0, top_k=10)

    # First result should be the seed
    assert results[0][0].id == "center"
    # Strong should come before weak
    ids = [r[0].id for r in results]
    assert ids.index("strong") < ids.index("weak")
```

### Task 2.3: Update Main Exports

**File:** `src/engram/__init__.py`

Add:
```python
from .activation import spread_activation, query, multi_query
```

Update `__all__` list.

Bump version to `0.5.0`.

---

## Phase 3: Verification

### Task 3.1: Run All Tests

```bash
pytest tests/ -v
```

All tests should pass after updates.

### Task 3.2: Integration Test

Create scenario test that exercises the full flow:

```python
def test_strength_model_integration():
    """Integration test for strength model + spreading activation."""
    graph = KnowledgeGraph()

    # Create a small knowledge graph about dogs
    dog_hdv = random_distributional(dim=100, seed=1)
    mammal_hdv = random_distributional(dim=100, seed=2)
    animal_hdv = random_distributional(dim=100, seed=3)
    barks_hdv = random_distributional(dim=100, seed=4)

    graph.add_node(Node(id="dog", hdv=dog_hdv, strength=5.0, last_accessed=0.0))
    graph.add_node(Node(id="mammal", hdv=mammal_hdv, strength=3.0, last_accessed=0.0))
    graph.add_node(Node(id="animal", hdv=animal_hdv, strength=2.0, last_accessed=0.0))
    graph.add_node(Node(id="barks", hdv=barks_hdv, strength=4.0, last_accessed=0.0))

    graph.add_edge(Edge(source="dog", target="mammal"))
    graph.add_edge(Edge(source="mammal", target="animal"))
    graph.add_edge(Edge(source="dog", target="barks"))

    # Query "what is a dog?"
    results = query(graph, "dog", current_time=0.0, top_k=5)

    # Should return dog and connected concepts
    result_ids = [r[0].id for r in results]
    assert "dog" in result_ids
    assert "mammal" in result_ids or "barks" in result_ids

    # Access dog again at t=1
    dog = graph.get_node("dog")
    graph.update_node("dog", last_accessed=1.0)

    # Query again - dog should still have high activation
    results2 = query(graph, "dog", current_time=1.0, top_k=5)
    assert results2[0][0].id == "dog"
```

---

## Migration Notes

### Breaking Changes

1. `Node.mass` renamed to `Node.strength`
2. `Node.energy` removed (use `get_effective_strength()`)
3. `Node.access()` signature changed: takes `current_time` instead of `energy_boost`
4. `bayesian_update_with_mass()` renamed to `bayesian_update_with_strength()`

### Deprecation Path

For v0.5.0, could add temporary aliases:
```python
@property
def mass(self):
    """Deprecated: use strength instead."""
    warnings.warn("mass is deprecated, use strength", DeprecationWarning)
    return self.strength

@property
def energy(self):
    """Deprecated: use get_effective_strength() instead."""
    warnings.warn("energy is deprecated, use get_effective_strength()", DeprecationWarning)
    return self.get_effective_strength(time.time())
```

Remove in v0.6.0.

---

## Checklist

- [ ] Task 1.1: Update Node class
- [ ] Task 1.2: Update dependent code (7 files)
- [ ] Task 1.3: Update all tests
- [ ] Task 2.1: Create activation module
- [ ] Task 2.2: Tests for spreading activation
- [ ] Task 2.3: Update main exports
- [ ] Task 3.1: Run all tests
- [ ] Task 3.2: Integration test
- [ ] Bump version to 0.5.0
