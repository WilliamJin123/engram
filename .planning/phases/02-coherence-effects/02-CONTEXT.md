# Phase 2: Coherence Effects - Context

**Gathered:** 2026-02-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Coherence modulates interference retrieval and surprise triggers re-coherence of decayed patterns. This phase makes coherence *functional* — not just a decaying value, but one that affects retrieval outcomes and responds to unexpected events. Tunneling and cascading effects belong to Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Coherence Weighting Curve
- Linear weighting as default: `weight = coherence`
- Optional exponent parameter (default 1.0) for future tuning: `weight = coherence^n`
- No hard cutoff — even very low coherence (< 0.05) patterns contribute proportionally
- Weights are normalized so contributions sum to 1.0 (proper weighted average)
- Coherence weighting applies in superposition building, not as post-hoc score multiplier
  - Build coherence-weighted superposition of patterns, then measure similarity to query
  - This allows interference effects between patterns, not just discounting

### Surprise Detection
- Surprise is emergent within the substrate, not externally injected
- Unified model: surprise = measurement collapse mismatch
  - Before retrieval: build expected superposition from query + current coherence weights
  - After retrieval: measure actual result
  - Surprise magnitude = distance between expected and actual
- Novelty is a subset: low prior expectation, high actual activation
- Surprise metric: **Hamming distance normalized by active bits**
  - Native to SDR representation (bit vectors)
  - Normalizable to 0-1 regardless of pattern size
  - Formula: `(bits_in_A + bits_in_B - 2 * overlap) / total_active_bits`

### Re-coherence Magnitude
- Proportional to surprise: higher surprise = bigger coherence boost
- No threshold: continuous function, even tiny surprises cause tiny boosts
- **All involved patterns re-cohere**, but scaled by proximity to surprise:
  - Surprising pattern (unexpected result): full proportional boost
  - Wrong prediction (expected pattern that was wrong): medium boost
  - Other participants in retrieval: smaller boost based on contribution strength
- Additive when multiple surprises occur simultaneously (capped at 1.0)
- No rate limiting for now — decay provides natural stability
  - If oscillation occurs in testing, add diminishing returns in Phase 3/4

### Edge Cases
- Zero-coherence patterns CAN be resurrected through surprise — nothing permanently forgotten
- Cascading re-coherence (spreading to connected patterns) deferred to Phase 3 tunneling

### Claude's Discretion
- Exact normalization implementation for coherence weights
- Specific hamming distance calculation optimization
- Coefficient for surprise-to-coherence-boost conversion
- How to efficiently track "expected vs actual" for surprise calculation

</decisions>

<specifics>
## Specific Ideas

**Emergent pattern semantics (important insight from discussion):**
- Low coherence ≠ forgotten; meaning depends on embeddedness
- Low coherence + highly connected = "settled knowledge" / foundational principle
  - Pattern's bits embedded in many others through coactivation
  - Stays influential via its connections even when quiet itself
- Low coherence + isolated = "faded artifact"
  - No connections, no embeddedness
  - Effectively irrelevant but not deleted

This spectrum is **emergent from existing mechanics**, not engineered:
- Embeddedness-modulated decay (Phase 1) ✓
- Coherence-weighted retrieval (this phase)
- Coactivation bit transfer (existing)

**Weighted superposition rationale:**
Building coherence-weighted superposition before querying is more "quantum-like" than multiplying scores after. Two 50%-coherent patterns might combine to produce interference effects neither alone would create.

</specifics>

<deferred>
## Deferred Ideas

- **Cascading re-coherence** — Re-coherence spreading to connected patterns belongs in Phase 3 tunneling mechanics
- **Diminishing returns on re-coherence** — Add if oscillation becomes a problem, otherwise keep simple
- **Explicit pruning mechanism** — Not needed in v1; let patterns fade naturally. Optimization for v2 if storage becomes concern
- **External contradiction injection** — Semantic contradiction ("cats are reptiles" vs "cats are mammals") requires agentic layer understanding; substrate handles measurable mismatches only

</deferred>

---

*Phase: 02-coherence-effects*
*Context gathered: 2026-02-01*
