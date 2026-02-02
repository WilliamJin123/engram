# Phase 3: Advanced Dynamics - Context

**Gathered:** 2026-02-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix coherence semantics so coherence governs malleability (how hard it is to change a pattern), NOT accessibility. Add tunneling for creative retrieval and criticality for system-wide order/chaos tuning. Retrieval becomes similarity-based; embeddedness emerges naturally from network structure.

</domain>

<decisions>
## Implementation Decisions

### Coherence Semantics (Malleability)
- HIGH coherence = malleable (easy to change) — fresh/uncertain patterns
- LOW coherence = crystallized (hard to change) — stable/confirmed patterns
- Crystallization driven by age + stability (patterns unchanged recently, frequently accessed)
- Contradiction handling: scale response by coherence (low-coherence patterns change proportionally less)
- Exceptions emerge naturally — if contradiction persists, new patterns form through repeated exposure

### Tunneling Behavior
- Always available at baseline, with explicit creative mode for amplification
- Auto-creative triggers: low retrieval confidence AND repeated retrieval failures
- Tunnel targets require BOTH: low bit overlap AND at least one connection path
- Tunneling strength scales linearly with source coherence (configurable exponent for experimentation)

### Criticality Parameter
- "Edge of chaos" model: 0.5 is optimal, extremes (0 or 1) are problematic
- Self-organizing: system adjusts criticality based on feedback signals
- Feedback signals: retrieval quality (poor matches → more chaos) AND surprise frequency (too many → too chaotic, too few → too rigid)
- Global scope only — one criticality value affects entire system

### Retrieval Weighting
- Remove embeddedness from explicit weighting — connectivity naturally affects retrieval through network pathways
- Pure similarity (bit overlap) as primary retrieval mechanism
- Recency boost as optional enhancement — implement and test both approaches
- Zero/low-connection patterns remain retrievable by similarity (no forgetting mechanism)

### Claude's Discretion
- Exact crystallization decay curve
- Specific thresholds for auto-creative mode triggers
- Criticality adjustment rate and dampening
- Linear vs exponential tunneling comparison methodology

</decisions>

<specifics>
## Specific Ideas

- "Engineer the substrate, let everything else emerge" — exceptions and inheritance should emerge from dynamics, not explicit mechanisms
- Embeddedness shouldn't need explicit tracking if connectivity naturally surfaces well-connected patterns
- Test both pure similarity and similarity+recency to find what works

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-advanced-dynamics*
*Context gathered: 2026-02-01*
