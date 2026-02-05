# Phase 7: Harder Behavior Tests - Context

**Gathered:** 2026-02-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Validate that tunneling, interference retrieval, and coherence dynamics work correctly under realistic memory load (near-misses, clutter, mixed noise). This phase creates stress tests using Phase 6 infrastructure. Metrics and baseline comparisons belong in Phase 8.

</domain>

<decisions>
## Implementation Decisions

### Tunneling test scenarios
- Navigate through mixed noise (both near-misses and random clutter)
- Test both single-hop (cue → target) and multi-hop chains (cue → intermediate → target)
- Verify threshold behavior: tunneling succeeds above coherence threshold, fails below
- Include creative mode tests (3x probability enables tunneling at lower thresholds)

### Interference retrieval criteria
- Test all three success criteria empirically:
  - Target in top-1 result (strict)
  - Target in top-K results (relaxed)
  - Target ranked above near-misses (relative)
- Parametrize near-miss count to observe degradation curves
- Focus on medium and high noise levels (skip trivial cases)
- Validate both precision (near-misses rejected) and recall (targets found)

### Coherence dynamics validation
- Test all three behaviors equally: decay, access refresh, surprise re-coherence
- Use access counts as proxy for time passage (N accesses without touching pattern = decay)
- Parametrize across noise levels to compare behavior under different loads
- Explicitly test stability score (access_count/10): highly-accessed patterns resist decay

### Test structure
- New directory: `tests/stress/` for stress testing suite
- Always output metrics (retrieval ranks, scores) for analysis

### Claude's Discretion
- Whether to use @pytest.mark.stress markers for selective running
- Whether to create specialized stress fixtures or reuse noisy_memory directly
- Exact organization of test files within tests/stress/

</decisions>

<specifics>
## Specific Ideas

- Empirically verify which success criteria (top-1, top-K, relative ranking) hold under different conditions rather than picking one upfront
- Output metrics always, not just on failure, to enable performance analysis across runs

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 07-harder-behavior-tests*
*Context gathered: 2026-02-04*
