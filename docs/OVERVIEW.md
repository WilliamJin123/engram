---
name: Engram
---

# Engram

A self-organizing memory framework for agentic systems.

## What It Is

A single graph where all knowledge lives—facts, tools, agents, constraints, code. Nodes follow simple physics: decay without use, strengthen with use, merge when similar. Hierarchy and abstraction emerge from these dynamics, not explicit engineering.

## Core Mechanics

**Nodes** carry content, properties, and an HDV (high-dimensional vector) that encodes structural relationships. HDVs enable fast similarity search without language understanding.

**Hyperedges** connect multiple nodes in n-ary relationships. A node's HDV reflects the company it keeps.

**Propagation** is local. When a node changes, it nudges neighbors. Signal decays with distance. High-mass nodes resist change; low-mass nodes are impressionable.

**Generalization** emerges from collision. Similar nodes cluster, shared properties extract into parents, children inherit. Nobody declares hierarchy—it crystallizes from usage.

## Architecture

```
Human ──▶ Working Agent (LLM) ──▶ Graph + HDV Store
                                        │
                                   (async)
                                        │
                                  Sleep Agents
```

**Wake**: Create, connect, query. Fast and messy.

**Sleep**: Cluster, abstract, prune, detect contradictions. Batched, no latency pressure.

LLMs interpret input and perform consolidation. Core graph operations are algebraic.

## Why This Design

- **Emergence over engineering**: Structure forms from dynamics, not declarations
- **Scalability through locality**: Changes propagate locally, not globally
- **Biological inspiration**: Mass creates stability, energy creates recency, sleep consolidates
- **Human gravity**: Your decisions are massive attractors that reorient the system

## Use Case

Long-term coherent AI collaboration on complex creative projects—game design, world-building, system architecture—where vision, constraints, and details must stay consistent as they evolve.