# Phase 5: Architecture Clarity - Context

**Gathered:** 2026-02-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Clear separation between substrate and agentic layers; validate quantum memory hypothesis. This phase establishes architectural boundaries that allow the substrate to be used independently of LLMs, and creates tests that can prove or disprove the quantum memory theory's predictions.

</domain>

<decisions>
## Implementation Decisions

### Module boundaries
- Substrate exposes full dynamics (pattern ops, coherence, decay, refresh, tunneling, surprise, criticality)
- Agentic layer imports substrate classes directly — simple, tightly coupled
- Agentic layer owns semantic interpretation: meaning extraction, intent detection, memory summarization
- Same package, separate modules: `engram/substrate/` and `engram/agentic/`

### LLM-independence scope
- Both import-time AND runtime independence for substrate
- No LLM imports in substrate code; substrate works without LLM at runtime
- TextEncoder split: hash-based encoding in substrate, LLM-embedding encoding in agentic
- Substrate accepts raw bit patterns directly — text encoding is optional convenience
- Prefer torch for math operations (numpy/scipy acceptable) in anticipation of GPU optimizations

### Hypothesis test design
- Core hypothesis: BOTH quantum-like interference AND coherence dynamics must be validated
- Invalidation criteria: Both must outperform baselines — if either matches or loses to alternatives, hypothesis weakened
- Interference baseline: must beat BOTH cosine similarity AND Jaccard similarity on bit patterns
- Success metric: precision (accurate top-K) AND serendipity (discovers non-obvious valid connections)

### INTUITION.md coverage
- All 8 behaviors get tests: generalization, inheritance, exceptions, certainty plasticity, conditionals, meta-relationships, provenance, history
- Failed tests: document as limitation AND identify what substrate capability would enable the behavior
- Test layers: minimal synthetic tests for unit validation, realistic tests for integration
- Pass bar: behavior is observed, reliable across runs, AND outperforms non-quantum alternative

### Claude's Discretion
- Exact module file structure within substrate/ and agentic/
- Specific interface design between layers
- Test data selection for realistic scenarios
- Statistical methodology for "outperforms" comparisons

</decisions>

<specifics>
## Specific Ideas

- Torch preferred over numpy/scipy for GPU optimization path
- Raw bit pattern API allows testing substrate without any text encoding
- Tests should honestly document limitations — not force-pass behaviors that don't emerge

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-architecture-clarity*
*Context gathered: 2026-02-04*
