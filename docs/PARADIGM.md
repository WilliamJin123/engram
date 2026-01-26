---
name: Engram
version: 0.1.0
description: A self-organizing memory framework for agentic systems

laws:
  - id: 1
    name: Single Source
    law: One graph, all knowledge

  - id: 2
    name: Multimodal Nodes
    law: Nodes are multimodal

  - id: 3
    name: Structural Meaning
    law: Structure is meaning

  - id: 4
    name: Decay
    law: Unused things decay

  - id: 5
    name: Reinforcement
    law: Used things strengthen

  - id: 6
    name: Convergence
    law: Similar things merge

  - id: 7
    name: Emergent Hierarchy
    law: Hierarchy emerges from abstraction

  - id: 8
    name: Downward Constraint
    law: Constraints flow downward

  - id: 9
    name: Contradiction Surfacing
    law: Contradictions must surface

  - id: 10
    name: Sleep/Wake Cycle
    law: Sleep organizes, wake creates

  - id: 11
    name: Agent Equivalence
    law: Agents are nodes

  - id: 12
    name: Tool Physics
    law: Tools obey node physics

  - id: 13
    name: Human Gravity
    law: Human input dominates

  - id: 14
    name: Algebraic Core
    law: LLMs at edges, algebra at core

  - id: 15
    name: Primitive Bootstrap
    law: Bootstrap from primitives
---

# Engram

A self-organizing memory framework for agentic systems.

## Overview

Engram is a memory architecture where structure emerges from simple dynamics rather than explicit engineering. It treats all knowledge—facts, tools, agents, constraints—as nodes in a single graph, subject to universal laws of decay, reinforcement, and consolidation.

## Core Concepts

### The Graph

The graph is the single source of truth. Everything that matters is a node:

- **Semantic**: facts, rules, constraints, decisions
- **Procedural**: tools, workflows, code, capabilities  
- **Episodic**: events, sessions, interactions
- **Entity**: game objects, characters, systems
- **Meta**: agents, the graph's own structure

There are no separate systems. If it exists, it's in the graph.

### Node Representation

Every node carries:

- **Content**: human-readable information
- **HDV**: high-dimensional vector encoding structural relationships
- **Energy**: recency/relevance, decays over time
- **Mass**: importance/confidence, grows with reinforcement
- **Edges**: typed connections to other nodes

For executable nodes (tools, workflows):

- **Inputs**: what it needs
- **Outputs**: what it produces
- **Implementation**: code, API calls, or workflow reference

### Hyperbolic Positioning

Nodes exist in hyperbolic space, derived from their HDV and graph relationships:

- **Near origin**: abstract, general, high-level (vision, principles)
- **Near boundary**: specific, concrete, detailed (code, assets)

This geometry enables natural hierarchy and efficient constraint queries.

## The Laws

### I. Single Source

> One graph, all knowledge

The graph contains everything. Facts, tools, agents, constraints, code, vision. No separate databases, no external state. If the system knows it, it's a node.

### II. Multimodal Nodes

> Nodes are multimodal

A node can represent any type of knowledge. The same structure handles semantic facts, procedural skills, episodic memories, and entity definitions. No special cases.

### III. Structural Meaning

> Structure is meaning

HDVs encode what a node connects to, what role it plays, what created it, what uses it. Similarity is computed from structure, not content. Core operations don't require language understanding.

### IV. Decay

> Unused things decay

Energy drains continuously without access. Mass fades without reinforcement. Nothing persists through inertia alone. The system actively forgets.

### V. Reinforcement

> Used things strengthen

Accessing a node restores energy. Successful use adds mass. The system remembers what matters by noticing what gets used.

### VI. Convergence

> Similar things merge

Nodes with high structural similarity attract. When close enough, they merge—the more massive node absorbs the other, or both combine into something new. Generalization is collision.

### VII. Emergent Hierarchy

> Hierarchy emerges from abstraction

Clusters develop parent nodes. Parents develop parents. Nobody declares the hierarchy. It crystallizes from usage patterns and structural similarity. Position in hyperbolic space reflects abstraction level.

### VIII. Downward Constraint

> Constraints flow downward

A constraint near the origin governs everything in its cone. Specific decisions automatically inherit abstract principles. Checking is geometric—cone queries find all applicable constraints.

### IX. Contradiction Surfacing

> Contradictions must surface

When nodes have high similarity but incompatible assertions, tension exists. The system detects contradictions but doesn't resolve them. Humans resolve. Confidence increases after resolution.

### X. Sleep/Wake Cycle

> Sleep organizes, wake creates

Working agents create nodes, make connections, get things done. They make messes. Sleep agents clean up: cluster, abstract, prune, detect conflicts, propose merges. Creation and organization are separate processes.

### XI. Agent Equivalence

> Agents are nodes

An agent's capabilities, role, and state live in the graph. Agents discover each other through graph queries. No privileged external registry. Agents can inspect, modify, and create other agents.

### XII. Tool Physics

> Tools obey node physics

Tools are nodes with implementation fields populated. They decay without use, merge when similar, form hierarchies through abstraction. A standard library emerges from evolutionary pressure.

### XIII. Human Gravity

> Human input dominates

Human decisions carry massive weight. When a human sets a constraint or makes a decision, it becomes a gravitational attractor. The graph reorganizes around human input. Humans don't micromanage—they set attractors.

### XIV. Algebraic Core

> LLMs at edges, algebra at core

LLMs interpret human input at ingestion. LLMs perform sleep consolidation. Core graph operations—similarity, decay, merging, queries—are algebraic. Fast, deterministic, auditable.

### XV. Primitive Bootstrap

> Bootstrap from primitives

The system starts with seed capabilities: basic I/O, code execution, meta-operations for creating tools and nodes. Everything else is built, discovered, or emerged. The system grows itself.

## Dynamics

### Energy and Mass

```
energy(t+1) = energy(t) × decay_rate      # constant drain
energy(t+1) += access_boost               # when accessed
mass(t+1) = mass(t) × slow_decay          # very slow drain
mass(t+1) += success_boost                # when used successfully
```

Nodes die when energy falls below threshold and nothing depends on them.

### Attraction and Merging

```
force(a, b) = (mass_a × mass_b × abstraction_ratio) / distance²
```

Nodes drift toward similar nodes. More abstract nodes (smaller radius) exert stronger pull on specific nodes. When distance falls below merge threshold, consolidation occurs.

### Sleep Processes

Async agents that operate on the graph:

- **Clusterer**: groups similar nodes, proposes parents
- **Abstractor**: identifies implicit principles, creates them
- **Contradiction Detector**: flags incompatible nodes for human review
- **Pruner**: removes dead nodes, archives low-energy nodes
- **Questioner**: surfaces ambiguity for next human session

### Constraint Checking

When creating or modifying a node at position P:

1. Cone query: find all nodes closer to origin in similar direction
2. Filter to constraint-type nodes
3. Check new node against each constraint
4. Flag violations for review or rejection

## Usage Patterns

### Adding Knowledge

```
human input 
  → LLM interprets 
  → structured node(s) created
  → HDV computed from structure
  → position derived from HDV
  → edges connected
  → energy initialized high
```

### Querying

```
query 
  → encode as HDV
  → find similar nodes (cosine similarity)
  → filter by energy threshold
  → rank by mass × similarity × energy
  → return relevant subgraph
```

### Tool Evolution

```
agent performs manual steps repeatedly
  → sleep agent detects pattern
  → proposes tool crystallization
  → human approves (or auto-approve if low-risk)
  → tool node created with implementation
  → linked to contexts where pattern occurred
  → future agents discover and use
  → usage reinforces, disuse decays
```

## Open Questions

- Exact decay rates and thresholds
- Merge conflict resolution strategies  
- Approval model for auto-generated tools
- Cold start: minimal primitive set
- Visualization and debugging interfaces
- Persistence and versioning

## Influences

- Hyperdimensional computing (Kanerva)
- Hyperbolic embeddings (Poincaré, Nickel et al.)
- Memory consolidation in neuroscience
- Stigmergic coordination in swarm systems
- Evolutionary pressure in genetic algorithms