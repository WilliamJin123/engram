# Plan B: Conditionals, History, and Exceptions

**Date:** 2026-01-26
**Status:** Draft
**Version:** 0.5.0 → 0.6.0
**Depends on:** Plan A (Strength Model & Spreading Activation)

## Overview

Implements three interconnected features from FUTURE_EXPLORATION.md:
1. **Conditionals** - "if sunny then weather is nice" with emergent patterns
2. **History** - belief change tracking without explicit versioning
3. **Exceptions** - "penguins can't fly" through variance-aware inheritance

### Design Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Conditional representation | Relationship nodes | Consistent with node-centric architecture |
| Pattern behavior | Hardcoded behaviors, pattern selects | Balance emergence with practicality |
| Exception handling | Implicit through variance | Simpler, more biological |
| History depth cutoff | Combined (strength + recency) | Most human-like |
| Propagation timing | Batch sweeps | Predictable, easier to reason about |

---

## Phase 1: Variance-Aware Propagation (Exceptions)

Exceptions emerge from variance-aware inheritance. High local variance dampens inherited values.

### Task 1.1: Propagation Module

**New module:** `src/engram/propagation/`

**Files:**
- `__init__.py`
- `inheritance.py`

**`inheritance.py`:**

```python
"""Variance-aware propagation for inheritance with exceptions.

When a child node has high variance on certain dimensions, inherited
values from the parent are dampened on those dimensions. This allows
exceptions like "penguins can't fly" to emerge naturally from observations.
"""

import torch
from engram.hdv import DistributionalHDV


def propagate_with_variance(
    parent_hdv: DistributionalHDV,
    child_hdv: DistributionalHDV,
    inheritance_weight: float = 0.5,
) -> DistributionalHDV:
    """Propagate HDV values while respecting child's variance.

    High child variance on a dimension = uncertain locally = accept less from parent.
    This enables exceptions without explicit marking.

    Args:
        parent_hdv: HDV of the parent/source node.
        child_hdv: HDV of the child/target node.
        inheritance_weight: Base weight for inheritance (0-1).

    Returns:
        New HDV with inherited values dampened by local variance.
    """
    eps = 1e-10

    # Kalman-style gain per dimension
    # Low child variance = trust child, accept little from parent
    # High child variance = uncertain, accept more from parent
    # BUT if parent also has high variance, don't trust it either
    gain = parent_hdv.variance / (parent_hdv.variance + child_hdv.variance + eps)
    gain = gain * inheritance_weight

    # Blend means
    new_mean = child_hdv.mean + gain * (parent_hdv.mean - child_hdv.mean)

    # Variance reduces where we gained information
    new_variance = child_hdv.variance * (1 - gain)

    # Preserve timestamps (take more recent)
    last_accessed = max(parent_hdv.last_accessed, child_hdv.last_accessed)
    last_updated = max(parent_hdv.last_updated, child_hdv.last_updated)

    return DistributionalHDV(
        mean=new_mean,
        variance=new_variance,
        last_accessed=last_accessed,
        last_updated=last_updated,
    )


def compute_inheritance_dampening(
    parent_hdv: DistributionalHDV,
    child_hdv: DistributionalHDV,
) -> torch.Tensor:
    """Compute per-dimension dampening factors.

    Returns tensor where high values = strong dampening (exception).
    Useful for inspecting which dimensions have exceptions.

    Args:
        parent_hdv: HDV of the parent node.
        child_hdv: HDV of the child node.

    Returns:
        Tensor of dampening factors per dimension (0 = full inherit, 1 = no inherit).
    """
    eps = 1e-10

    # Dampening is high when child variance is low (confident local value)
    # and parent variance is high (uncertain inheritance)
    child_confidence = 1 / (child_hdv.variance + eps)
    parent_confidence = 1 / (parent_hdv.variance + eps)

    # Normalize to 0-1 range
    total = child_confidence + parent_confidence
    dampening = child_confidence / total

    return dampening
```

### Task 1.2: Tests for Variance-Aware Propagation

**File:** `tests/propagation/test_inheritance.py`

```python
"""Tests for variance-aware inheritance propagation."""

import pytest
import torch
from engram.hdv import random_distributional, DistributionalHDV
from engram.propagation import propagate_with_variance, compute_inheritance_dampening


def test_high_child_variance_accepts_parent():
    """High child variance = uncertain = accept parent value."""
    dim = 100

    parent = DistributionalHDV(
        mean=torch.ones(dim),
        variance=torch.ones(dim) * 0.1,  # Confident parent
        last_accessed=0.0,
        last_updated=0.0,
    )

    child = DistributionalHDV(
        mean=torch.zeros(dim),
        variance=torch.ones(dim) * 10.0,  # Uncertain child
        last_accessed=0.0,
        last_updated=0.0,
    )

    result = propagate_with_variance(parent, child)

    # Child should move toward parent (high child variance)
    assert result.mean.mean().item() > 0.3


def test_low_child_variance_resists_parent():
    """Low child variance = confident = resist parent value (exception)."""
    dim = 100

    parent = DistributionalHDV(
        mean=torch.ones(dim),
        variance=torch.ones(dim) * 0.1,
        last_accessed=0.0,
        last_updated=0.0,
    )

    child = DistributionalHDV(
        mean=torch.zeros(dim),
        variance=torch.ones(dim) * 0.01,  # Very confident child (exception)
        last_accessed=0.0,
        last_updated=0.0,
    )

    result = propagate_with_variance(parent, child)

    # Child should stay near zero (low child variance = exception)
    assert result.mean.mean().item() < 0.2


def test_per_dimension_exceptions():
    """Test that exceptions work per-dimension."""
    dim = 100

    parent = DistributionalHDV(
        mean=torch.ones(dim),
        variance=torch.ones(dim) * 0.1,
        last_accessed=0.0,
        last_updated=0.0,
    )

    # Child has low variance (exception) on first 50 dims, high on rest
    child_variance = torch.ones(dim) * 10.0
    child_variance[:50] = 0.01  # Exception on first 50 dims

    child = DistributionalHDV(
        mean=torch.zeros(dim),
        variance=child_variance,
        last_accessed=0.0,
        last_updated=0.0,
    )

    result = propagate_with_variance(parent, child)

    # First 50 dims should stay near 0 (exception)
    assert result.mean[:50].mean().item() < 0.2
    # Last 50 dims should move toward 1 (inherit)
    assert result.mean[50:].mean().item() > 0.3


def test_dampening_inspection():
    """Test that we can inspect where exceptions are."""
    dim = 100

    parent = DistributionalHDV(
        mean=torch.ones(dim),
        variance=torch.ones(dim) * 0.1,
        last_accessed=0.0,
        last_updated=0.0,
    )

    child_variance = torch.ones(dim) * 10.0
    child_variance[:50] = 0.01

    child = DistributionalHDV(
        mean=torch.zeros(dim),
        variance=child_variance,
        last_accessed=0.0,
        last_updated=0.0,
    )

    dampening = compute_inheritance_dampening(parent, child)

    # First 50 dims should have high dampening (exception)
    assert dampening[:50].mean().item() > 0.8
    # Last 50 dims should have low dampening (inherit)
    assert dampening[50:].mean().item() < 0.2
```

---

## Phase 2: Conditional Relationships

Conditionals are relationship nodes. Patterns emerge through clustering.

### Task 2.1: Conditional Module

**New module:** `src/engram/conditional/`

**Files:**
- `__init__.py`
- `base.py`
- `behaviors.py`

**`base.py`:**

```python
"""Conditional relationship nodes."""

from engram.graph import Node, Edge, KnowledgeGraph
from engram.hdv import DistributionalHDV, distributional_bind


def create_conditional(
    graph: KnowledgeGraph,
    conditional_id: str,
    antecedent_id: str,
    consequent_id: str,
    confidence: float = 1.0,
    current_time: float = 0.0,
) -> Node:
    """Create a conditional relationship node.

    The conditional's HDV is the binding of antecedent and consequent,
    making similar conditionals have similar HDVs (enabling pattern emergence).

    Args:
        graph: Knowledge graph.
        conditional_id: ID for the conditional node.
        antecedent_id: ID of the "if" node.
        consequent_id: ID of the "then" node.
        confidence: Initial strength of the conditional.
        current_time: Current timestamp.

    Returns:
        The created conditional node.
    """
    antecedent = graph.get_node(antecedent_id)
    consequent = graph.get_node(consequent_id)

    if not antecedent or not consequent:
        raise ValueError("Antecedent and consequent must exist in graph")

    # Bind HDVs to create conditional identity
    hdv = distributional_bind(antecedent.hdv, consequent.hdv)

    node = Node(
        id=conditional_id,
        hdv=hdv,
        strength=confidence,
        last_accessed=current_time,
        content={
            "type": "relationship",
            "relationship_kind": "conditional",
            "antecedent": antecedent_id,
            "consequent": consequent_id,
        },
    )

    graph.add_node(node)

    # Create edges: antecedent -> conditional -> consequent
    graph.add_edge(Edge(source=antecedent_id, target=conditional_id))
    graph.add_edge(Edge(source=conditional_id, target=consequent_id))

    return node


def is_conditional(node: Node) -> bool:
    """Check if a node is a conditional relationship."""
    if not isinstance(node.content, dict):
        return False
    return (
        node.content.get("type") == "relationship"
        and node.content.get("relationship_kind") == "conditional"
    )


def get_conditional_parts(node: Node) -> tuple[str, str] | None:
    """Get antecedent and consequent IDs from a conditional node."""
    if not is_conditional(node):
        return None
    return (
        node.content.get("antecedent"),
        node.content.get("consequent"),
    )
```

**`behaviors.py`:**

```python
"""Hardcoded propagation behaviors that patterns can select."""

from enum import Enum
from engram.graph import KnowledgeGraph


class PropagationBehavior(Enum):
    """Available propagation behaviors."""
    FORWARD = "forward"           # A active -> B activates
    BIDIRECTIONAL = "bidirectional"  # Either active -> other activates
    INHIBITION = "inhibition"     # A active -> B suppressed
    NONE = "none"                 # No automatic propagation


def apply_forward(
    graph: KnowledgeGraph,
    activations: dict[str, float],
    source_id: str,
    target_id: str,
    strength: float,
    spread_factor: float = 0.5,
) -> dict[str, float]:
    """Apply forward propagation: source -> target.

    Args:
        graph: Knowledge graph.
        activations: Current activation dict (will be modified).
        source_id: Source node ID.
        target_id: Target node ID.
        strength: Relationship strength.
        spread_factor: How much activation spreads.

    Returns:
        Updated activations dict.
    """
    source_activation = activations.get(source_id, 0)

    if source_activation > 0.1:  # Source must be sufficiently active
        spread = source_activation * strength * spread_factor
        old = activations.get(target_id, 0)
        activations[target_id] = max(old, spread)

    return activations


def apply_bidirectional(
    graph: KnowledgeGraph,
    activations: dict[str, float],
    node_a_id: str,
    node_b_id: str,
    strength: float,
    spread_factor: float = 0.5,
) -> dict[str, float]:
    """Apply bidirectional propagation: A <-> B."""
    # A -> B
    activations = apply_forward(
        graph, activations, node_a_id, node_b_id, strength, spread_factor
    )
    # B -> A
    activations = apply_forward(
        graph, activations, node_b_id, node_a_id, strength, spread_factor
    )
    return activations


def apply_inhibition(
    graph: KnowledgeGraph,
    activations: dict[str, float],
    source_id: str,
    target_id: str,
    strength: float,
    inhibition_factor: float = 0.5,
) -> dict[str, float]:
    """Apply inhibition: source active -> target suppressed."""
    source_activation = activations.get(source_id, 0)

    if source_activation > 0.1:
        suppression = source_activation * strength * inhibition_factor
        old = activations.get(target_id, 0)
        activations[target_id] = max(0, old - suppression)

    return activations
```

### Task 2.2: Pattern-Aware Spreading

Update spreading activation to use relationship behaviors.

**File:** `src/engram/activation/spreading.py` (update)

Add function:

```python
def spread_with_behaviors(
    graph: KnowledgeGraph,
    seed_ids: list[str] | str,
    current_time: float,
    max_iterations: int = 5,
    convergence_threshold: float = 0.01,
    **kwargs,
) -> dict[str, float]:
    """Spread activation with relationship-aware behavior.

    Unlike basic spreading, this applies specific behaviors for
    relationship nodes (conditionals, etc.).

    Args:
        graph: Knowledge graph.
        seed_ids: Starting node ID(s).
        current_time: Current timestamp.
        max_iterations: Maximum spreading iterations.
        convergence_threshold: Stop when max change < this.
        **kwargs: Additional args for basic spreading.

    Returns:
        Dict mapping node_id -> activation strength.
    """
    from engram.conditional import is_conditional, get_conditional_parts
    from engram.conditional.behaviors import apply_forward, PropagationBehavior

    # Initial spread
    activations = spread_activation(graph, seed_ids, current_time, **kwargs)

    for _ in range(max_iterations):
        old_activations = activations.copy()

        # Apply relationship behaviors
        for node_id, activation in list(activations.items()):
            node = graph.get_node(node_id)
            if not node:
                continue

            # Check if this is a conditional
            if is_conditional(node):
                parts = get_conditional_parts(node)
                if parts:
                    antecedent_id, consequent_id = parts
                    antecedent_activation = activations.get(antecedent_id, 0)

                    # If antecedent is active, propagate to consequent
                    if antecedent_activation > 0.1:
                        eff_strength = node.get_effective_strength(current_time)
                        activations = apply_forward(
                            graph, activations,
                            antecedent_id, consequent_id,
                            eff_strength,
                        )

        # Check convergence
        max_change = max(
            abs(activations.get(k, 0) - old_activations.get(k, 0))
            for k in set(activations) | set(old_activations)
        )
        if max_change < convergence_threshold:
            break

    return activations
```

### Task 2.3: Tests for Conditionals

**File:** `tests/conditional/test_conditional.py`

```python
"""Tests for conditional relationships."""

import pytest
from engram import random_distributional, KnowledgeGraph, Node, Edge
from engram.conditional import create_conditional, is_conditional
from engram.activation import spread_with_behaviors


def test_create_conditional():
    """Test creating a conditional relationship."""
    graph = KnowledgeGraph()

    hdv1 = random_distributional(dim=100, seed=1)
    hdv2 = random_distributional(dim=100, seed=2)

    graph.add_node(Node(id="sunny", hdv=hdv1, strength=1.0))
    graph.add_node(Node(id="nice_weather", hdv=hdv2, strength=1.0))

    cond = create_conditional(
        graph, "if_sunny_nice", "sunny", "nice_weather", confidence=0.8
    )

    assert is_conditional(cond)
    assert cond.content["antecedent"] == "sunny"
    assert cond.content["consequent"] == "nice_weather"


def test_conditional_propagation():
    """Test that conditionals propagate activation correctly."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="sunny", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="nice_weather", hdv=hdv, strength=1.0, last_accessed=0.0))

    create_conditional(graph, "if_sunny_nice", "sunny", "nice_weather", confidence=1.0)

    # Spread from sunny
    activations = spread_with_behaviors(graph, "sunny", current_time=0.0)

    # nice_weather should be activated via the conditional
    assert activations.get("nice_weather", 0) > 0.1


def test_conditional_chain():
    """Test chained conditionals."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="sunny", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="nice_weather", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="go_outside", hdv=hdv, strength=1.0, last_accessed=0.0))

    create_conditional(graph, "cond1", "sunny", "nice_weather", confidence=1.0)
    create_conditional(graph, "cond2", "nice_weather", "go_outside", confidence=1.0)

    activations = spread_with_behaviors(
        graph, "sunny", current_time=0.0, max_iterations=5
    )

    # go_outside should be activated through the chain
    assert activations.get("go_outside", 0) > 0.05


def test_weak_conditional_less_propagation():
    """Test that low-strength conditionals propagate less."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)
    graph.add_node(Node(id="A", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="B_strong", hdv=hdv, strength=1.0, last_accessed=0.0))
    graph.add_node(Node(id="B_weak", hdv=hdv, strength=1.0, last_accessed=0.0))

    create_conditional(graph, "strong_cond", "A", "B_strong", confidence=2.0)
    create_conditional(graph, "weak_cond", "A", "B_weak", confidence=0.2)

    activations = spread_with_behaviors(graph, "A", current_time=0.0)

    # Strong conditional should propagate more
    assert activations.get("B_strong", 0) > activations.get("B_weak", 0)
```

---

## Phase 3: History Tracking

History emerges from succession edges and context nodes.

### Task 3.1: Enhance ContradictionDetector

**File:** `src/engram/sleep/contradiction.py` (update)

Add method:

```python
def resolve_contradiction(
    self,
    graph: KnowledgeGraph,
    contradiction: dict,
    winner_id: str,
    context_content: str = None,
    current_time: float = 0.0,
) -> dict:
    """Resolve a contradiction by declaring a winner.

    Creates succession edge from loser to winner.
    Optionally creates context node explaining the change.

    Args:
        graph: Knowledge graph.
        contradiction: Contradiction dict from find_contradictions().
        winner_id: ID of the winning belief.
        context_content: Optional description of why/how resolved.
        current_time: Current timestamp.

    Returns:
        Resolution results including succession edge.
    """
    node_a_id = contradiction["node_a"]
    node_b_id = contradiction["node_b"]

    loser_id = node_b_id if winner_id == node_a_id else node_a_id

    # Decay loser's strength
    loser = graph.get_node(loser_id)
    graph.update_node(loser_id, strength=loser.strength * 0.1)

    # Boost winner's strength
    winner = graph.get_node(winner_id)
    graph.update_node(winner_id, strength=winner.strength * 1.5)

    # Create succession edge: loser -> winner
    succession_edge = Edge(source=loser_id, target=winner_id)
    graph.add_edge(succession_edge)

    # Mark loser with succession metadata
    loser = graph.get_node(loser_id)
    if isinstance(loser.content, dict):
        loser.content["superseded_by"] = winner_id
    else:
        graph.update_node(loser_id, content={
            "original": loser.content,
            "superseded_by": winner_id,
        })

    # Create context node if provided
    context_id = None
    if context_content:
        context_id = f"context_{loser_id}_{winner_id}"

        # Context HDV binds the two beliefs
        context_hdv = distributional_bind(loser.hdv, winner.hdv)

        context_node = Node(
            id=context_id,
            hdv=context_hdv,
            strength=0.5,
            last_accessed=current_time,
            content={
                "type": "succession_context",
                "from_belief": loser_id,
                "to_belief": winner_id,
                "description": context_content,
                "timestamp": current_time,
            },
        )
        graph.add_node(context_node)
        graph.add_edge(Edge(source=context_id, target=loser_id))
        graph.add_edge(Edge(source=context_id, target=winner_id))

    return {
        "winner": winner_id,
        "loser": loser_id,
        "succession_edge": (loser_id, winner_id),
        "context_id": context_id,
    }
```

### Task 3.2: History Query

**New file:** `src/engram/history/`

**`query.py`:**

```python
"""History query utilities."""

from engram.graph import KnowledgeGraph, Node


def get_belief_history(
    graph: KnowledgeGraph,
    belief_id: str,
    current_time: float,
    recency_tau: float = 1.0,
    min_accessibility: float = 0.05,
) -> list[dict]:
    """Trace the history of a belief through succession edges.

    Uses combined strength + recency cutoff (most human-like).

    Args:
        graph: Knowledge graph.
        belief_id: Current belief to trace history from.
        current_time: Current timestamp.
        recency_tau: Time constant for recency computation.
        min_accessibility: Stop when effective strength < this.

    Returns:
        List of past beliefs ordered from oldest to newest.
        Each entry has: belief_id, content, strength, context.
    """
    history = []
    visited = set()

    def trace_back(node_id: str, depth: int = 0):
        if node_id in visited or depth > 20:  # Prevent cycles
            return
        visited.add(node_id)

        node = graph.get_node(node_id)
        if not node:
            return

        # Check accessibility (combined strength + recency)
        eff_strength = node.get_effective_strength(current_time, recency_tau)
        if eff_strength < min_accessibility:
            return  # Too forgotten

        # Find predecessors (nodes that point to this one)
        predecessors = graph.get_incoming(node_id)

        for pred_id in predecessors:
            pred = graph.get_node(pred_id)
            if not pred:
                continue

            # Check if this is a superseded belief
            if isinstance(pred.content, dict):
                superseded_by = pred.content.get("superseded_by")
                if superseded_by == node_id:
                    # This is a predecessor belief - recurse first
                    trace_back(pred_id, depth + 1)

                    # Find context node
                    context = _find_succession_context(graph, pred_id, node_id)

                    history.append({
                        "belief_id": pred_id,
                        "content": pred.content.get("original", pred.content),
                        "strength": pred.strength,
                        "effective_strength": pred.get_effective_strength(
                            current_time, recency_tau
                        ),
                        "context": context,
                    })

    trace_back(belief_id)
    return history


def _find_succession_context(
    graph: KnowledgeGraph,
    from_id: str,
    to_id: str,
) -> dict | None:
    """Find the context node for a belief succession."""
    # Context nodes connect to both beliefs
    from_incoming = graph.get_incoming(from_id)
    to_incoming = graph.get_incoming(to_id)

    # Find common nodes that are context nodes
    common = from_incoming & to_incoming

    for node_id in common:
        node = graph.get_node(node_id)
        if node and isinstance(node.content, dict):
            if node.content.get("type") == "succession_context":
                return {
                    "context_id": node_id,
                    "description": node.content.get("description"),
                    "timestamp": node.content.get("timestamp"),
                }

    return None
```

### Task 3.3: Tests for History

**File:** `tests/history/test_history.py`

```python
"""Tests for belief history tracking."""

import pytest
from engram import random_distributional, KnowledgeGraph, Node
from engram.sleep import ContradictionDetector
from engram.history import get_belief_history


def test_belief_succession():
    """Test basic belief succession tracking."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)

    # Old belief
    graph.add_node(Node(
        id="pluto_planet",
        hdv=hdv,
        strength=5.0,
        last_accessed=0.0,
        content="Pluto is a planet"
    ))

    # New belief (opposing)
    new_hdv = random_distributional(dim=100, seed=43)
    new_hdv.mean = -hdv.mean  # Opposite
    graph.add_node(Node(
        id="pluto_dwarf",
        hdv=new_hdv,
        strength=1.0,
        last_accessed=0.0,
        content="Pluto is a dwarf planet"
    ))

    # Resolve contradiction
    detector = ContradictionDetector(similarity_threshold=0.00001)

    result = detector.resolve_contradiction(
        graph,
        {"node_a": "pluto_planet", "node_b": "pluto_dwarf"},
        winner_id="pluto_dwarf",
        context_content="IAU 2006 redefinition",
        current_time=1.0,
    )

    # Check succession
    assert result["winner"] == "pluto_dwarf"
    assert result["loser"] == "pluto_planet"

    # Trace history
    history = get_belief_history(graph, "pluto_dwarf", current_time=1.0)

    assert len(history) == 1
    assert history[0]["belief_id"] == "pluto_planet"
    assert history[0]["context"]["description"] == "IAU 2006 redefinition"


def test_history_respects_accessibility_cutoff():
    """Test that history stops at inaccessible (forgotten) beliefs."""
    graph = KnowledgeGraph()

    hdv = random_distributional(dim=100, seed=42)

    # Very old, low-strength belief (should be filtered)
    graph.add_node(Node(
        id="old_belief",
        hdv=hdv,
        strength=0.001,  # Very low
        last_accessed=0.0,  # Long ago
        content={"original": "old thing", "superseded_by": "new_belief"}
    ))

    graph.add_node(Node(
        id="new_belief",
        hdv=hdv,
        strength=5.0,
        last_accessed=100.0,
        content="new thing"
    ))

    from engram.graph import Edge
    graph.add_edge(Edge(source="old_belief", target="new_belief"))

    # Query at t=100 with tau=1
    # old_belief effective strength ≈ 0.001 * (1 + exp(-100)) ≈ 0.001
    history = get_belief_history(
        graph, "new_belief",
        current_time=100.0,
        recency_tau=1.0,
        min_accessibility=0.01
    )

    # Old belief should be filtered out (too forgotten)
    assert len(history) == 0
```

---

## Phase 4: Integration & Exports

### Task 4.1: Update Main Exports

**File:** `src/engram/__init__.py`

Add new exports:

```python
# Propagation
from .propagation import propagate_with_variance, compute_inheritance_dampening

# Conditionals
from .conditional import (
    create_conditional,
    is_conditional,
    get_conditional_parts,
    PropagationBehavior,
)

# History
from .history import get_belief_history

# Update activation exports
from .activation import spread_activation, query, multi_query, spread_with_behaviors
```

Update `__all__`.

Bump version to `0.6.0`.

### Task 4.2: Scenario Tests

**File:** `tests/scenarios/test_future_features.py`

```python
"""Integration scenarios for conditionals, history, exceptions."""

import pytest
import torch
from engram import (
    random_distributional, KnowledgeGraph, Node, Edge,
    create_conditional, get_belief_history,
    spread_with_behaviors,
)
from engram.propagation import propagate_with_variance
from engram.hdv import DistributionalHDV


class TestWeatherConditionalScenario:
    """Scenario: Weather conditionals."""

    def test_sunny_implies_nice_implies_outside(self):
        """If sunny -> nice weather -> can go outside."""
        graph = KnowledgeGraph()

        hdv = random_distributional(dim=100, seed=42)
        graph.add_node(Node(id="sunny", hdv=hdv, strength=2.0, last_accessed=0.0))
        graph.add_node(Node(id="nice_weather", hdv=hdv, strength=1.0, last_accessed=0.0))
        graph.add_node(Node(id="go_outside", hdv=hdv, strength=1.0, last_accessed=0.0))

        create_conditional(graph, "c1", "sunny", "nice_weather", confidence=1.0)
        create_conditional(graph, "c2", "nice_weather", "go_outside", confidence=1.0)

        # Query starting from sunny
        activations = spread_with_behaviors(graph, "sunny", current_time=0.0)

        # Chain should propagate
        assert activations.get("nice_weather", 0) > 0.1
        assert activations.get("go_outside", 0) > 0.01


class TestPenguinExceptionScenario:
    """Scenario: Penguins can't fly (exception to birds flying)."""

    def test_penguin_exception_through_variance(self):
        """Penguin has low variance on flight dims = exception."""
        dim = 100

        # Bird: confident that birds fly (low variance on "flight" dims)
        bird_hdv = DistributionalHDV(
            mean=torch.ones(dim),  # "can fly" encoded as positive
            variance=torch.ones(dim) * 0.1,  # Confident
            last_accessed=0.0,
            last_updated=0.0,
        )

        # Penguin: confident that it can't fly on flight dims (exception)
        penguin_variance = torch.ones(dim) * 10.0  # Uncertain by default
        penguin_variance[:20] = 0.01  # Very confident on "flight" dims (exception)

        penguin_hdv = DistributionalHDV(
            mean=torch.cat([
                torch.zeros(20),  # "cannot fly" on first 20 dims
                torch.zeros(80),  # Uncertain on rest
            ]),
            variance=penguin_variance,
            last_accessed=0.0,
            last_updated=0.0,
        )

        # Propagate bird -> penguin
        result = propagate_with_variance(bird_hdv, penguin_hdv)

        # Flight dims (0-20) should stay near 0 (penguin's exception)
        assert result.mean[:20].mean().item() < 0.3

        # Other dims should inherit from bird (move toward 1)
        assert result.mean[20:].mean().item() > 0.3


class TestBeliefRevisionScenario:
    """Scenario: Pluto reclassification."""

    def test_belief_history_preserved(self):
        """Old belief should be traceable through history."""
        from engram.sleep import ContradictionDetector

        graph = KnowledgeGraph()

        hdv1 = random_distributional(dim=100, seed=1)
        hdv2 = random_distributional(dim=100, seed=2)
        hdv2.mean = -hdv1.mean  # Opposite

        graph.add_node(Node(
            id="pluto_planet",
            hdv=hdv1,
            strength=5.0,
            last_accessed=0.0,
            content="Pluto is a planet",
        ))

        graph.add_node(Node(
            id="pluto_dwarf",
            hdv=hdv2,
            strength=1.0,
            last_accessed=0.0,
            content="Pluto is a dwarf planet",
        ))

        # Resolve contradiction
        detector = ContradictionDetector(similarity_threshold=0.00001)
        detector.resolve_contradiction(
            graph,
            {"node_a": "pluto_planet", "node_b": "pluto_dwarf"},
            winner_id="pluto_dwarf",
            context_content="IAU 2006 reclassification",
            current_time=1.0,
        )

        # Query history
        history = get_belief_history(graph, "pluto_dwarf", current_time=1.0)

        assert len(history) >= 1
        old_beliefs = [h["belief_id"] for h in history]
        assert "pluto_planet" in old_beliefs
```

---

## Checklist

### Phase 1: Exceptions (Variance-Aware Propagation)
- [ ] Task 1.1: Create propagation module
- [ ] Task 1.2: Tests for variance-aware propagation

### Phase 2: Conditionals
- [ ] Task 2.1: Create conditional module
- [ ] Task 2.2: Pattern-aware spreading
- [ ] Task 2.3: Tests for conditionals

### Phase 3: History
- [ ] Task 3.1: Enhance ContradictionDetector
- [ ] Task 3.2: History query module
- [ ] Task 3.3: Tests for history

### Phase 4: Integration
- [ ] Task 4.1: Update main exports
- [ ] Task 4.2: Scenario tests
- [ ] Bump version to 0.6.0

---

## Dependencies

```
Plan A (v0.5.0)
├── Strength model refactoring
└── Spreading activation
         │
         ▼
Plan B (v0.6.0)
├── Phase 1: Exceptions (propagation module)
├── Phase 2: Conditionals (uses spreading activation)
├── Phase 3: History (uses strength model)
└── Phase 4: Integration
```

Execute Plan A first, then Plan B.
