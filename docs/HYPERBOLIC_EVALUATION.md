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
