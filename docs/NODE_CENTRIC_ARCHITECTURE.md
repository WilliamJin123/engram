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
[dog] -> [is_a_relationship_123] -> [mammal]
[is_a_relationship_123] -> [IS_A_type]
[confidence_note_456] -> [is_a_relationship_123]
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
- `bind(A, B)` - creates associations (relationship HDV = bind of participants)
- `bundle([A, B, C])` - creates collections/superpositions

**HDV assignment:**
- Atomic nodes: random HDV
- Derived nodes: computed via bind from connected nodes

### 4. Content is Human-Readable Source

The `content` field holds human-readable information. An LLM encodes this into the HDV at ingestion. Core graph operations never parse content - they use HDV.

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
[dog] -> [is_a_rel_1]
[is_a_rel_1] -> [mammal]
[is_a_rel_1] -> [IS_A_type]
```

To query "what is dog?": find nodes that dog points to through relationship nodes.
To query "what things are mammals?": find nodes that point to mammal through relationship nodes.
