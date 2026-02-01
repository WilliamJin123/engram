# Phase 1: Coherence Foundation - Context

**Gathered:** 2026-02-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Every pattern has a coherence value (0-1 scalar) that decays toward 0 when not accessed and refreshes when accessed. This phase implements the core coherence field with decay and refresh mechanics. Surprise re-coherence and interference modulation are Phase 2.

</domain>

<decisions>
## Implementation Decisions

### Decay Dynamics
- System-wide constant decay rate (not per-pattern configurable)
- Time measured in operations, not wall-clock time — each operation is one "tick"
- Small coherence floor (e.g., 0.01) — patterns never fully reach 0
- Decay rate is a function of structural embeddedness — highly connected patterns decay slower
- Decay happens first in each operation, before refresh

### Refresh Behavior
- Refresh amount is proportional to activation strength (emergent, not an explicit parameter)
- Hard cap at 1.0 — no overshoot
- Both store and retrieve operations refresh patterns
- All activated patterns refresh proportionally (not just top-k winners)
- Coactivation refreshes both participating patterns
- Indirect refresh through bindings — weak (~10-20% of direct), multi-hop with decay
- Certainty/entrenchment emerges from structural embeddedness, not tracked explicitly

### Initial Coherence
- All new patterns start at coherence 1.0 (fully quantum)
- Novelty and source confidence emerge from structural overlap with existing patterns
- High overlap → high embeddedness → faster decay to classical
- Low overlap (truly novel) → low embeddedness → stays uncertain longer
- No explicit parameters for novelty or confidence

### Access Definition
- All interactions count as activation: retrieval, storage, coactivation, binding
- Only activated patterns refresh — zero-activation patterns don't
- No activation threshold — continuous (tiny activations cause proportionally tiny refresh)
- Operation order: decay all patterns first, then refresh activated patterns

### Claude's Discretion
- Exact decay rate constant value
- Exact coherence floor value
- Implementation of "structural embeddedness" metric
- Multi-hop propagation depth limit (if needed for performance)

</decisions>

<specifics>
## Specific Ideas

- "New knowledge retains coherence (inherent quantum uncertainty), but over time things settle into decoherence. Some ideas might inherently be uncertain and maintain coherence for a long time, while others might be very convincing/strong/impactful and quickly settle into classical, rooted memories."
- "Well-practiced skills that resurface should not decohere a lot — they're structurally embedded (rooted). It should take lots of proof/cases to change such a memory."
- Certainty should emerge from the substrate itself (like synaptic weights in brains), not from explicit metadata tracking.
- The "unit of time" is operations because we don't have perpetual sensory input — operations are our clock ticks.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-coherence-foundation*
*Context gathered: 2026-02-01*
