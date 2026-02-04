# Phase 6: Test Infrastructure - Context

**Gathered:** 2026-02-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a noise generation toolkit for stress-testing retrieval mechanisms. This includes near-miss pattern generators, random clutter generators, configurable noise level presets, and reusable test fixtures. The toolkit enables Phase 7 behavior tests and Phase 8 metrics validation.

</domain>

<decisions>
## Implementation Decisions

### Near-miss generation
- Near-misses should have varied coherence levels (aged patterns) to simulate realistic memory, not fresh baseline coherence
- Claude's discretion: how near-misses relate to targets (bit overlap vs semantic category), count/ratio, and storage approach

### Clutter characteristics
- Clutter must match the sparsity distribution of real patterns (same bit density) — not purely random bits
- Clutter participates in coherence dynamics (has coherence values, not inert)
- Clutter should have varying relatedness to targets — some distant, some mid-range — not completely unrelated
- Support both setup-time clutter and continuous injection during test execution

### Noise level presets
- Named presets (NoiseLevel.LOW, etc.) plus ability to specify custom exact ratios
- Near-miss and clutter ratios defined separately per noise level (not combined)
- All noise generation must be seeded for reproducibility — always seeded, not optional

### Fixture API design
- Noise generation code lives in tests/ directory (conftest.py or test utils)
- Support both fresh memory creation with noise AND injecting noise into existing memory
- Claude's discretion: specific API pattern (builder, parameterized fixture, or factory), exposure of noise patterns for assertions

### Claude's Discretion
- Near-miss implementation approach (bit overlap control vs semantic category)
- Near-miss count/ratio per target
- Near-miss storage strategy (stored in memory vs on-demand)
- Noise level ratio definitions (what constitutes low/medium/high)
- Fixture API pattern choice
- Whether fixtures expose noise patterns for test assertions

</decisions>

<specifics>
## Specific Ideas

- Coherence should be realistic for both near-misses and clutter — aged patterns, not fresh
- "Varying relatedness" for clutter means the noise isn't artificially pure — some patterns will be closer to targets than others, creating a realistic interference landscape
- Both setup-time and continuous injection needed to test different scenarios (stable memory vs memory under growth)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-test-infrastructure*
*Context gathered: 2026-02-04*
