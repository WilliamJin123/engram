# Phase 4: Codebase Cleanup - Context

**Gathered:** 2026-02-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix non-deterministic behavior in coactivation (FIX-01) and text encoding collisions at small k values (FIX-02). No new features — pure bug fixes to make existing operations deterministic and collision-free.

</domain>

<decisions>
## Implementation Decisions

### RNG Seeding Strategy
- Session-level seed with optional per-call override (both options)
- If no seed provided at either level: random seed (current behavior preserved, determinism is opt-in)
- RNG state isolated per operation — same seed always produces same result for that specific operation
- Seed used stored in pattern metadata for debugging/reproducibility

### Collision Resolution
- Fix via better hash function (not minimum k enforcement or collision detection)
- Error on k < 10 (refuse to encode at small k values)
- No uniqueness check against existing patterns needed — trust the improved hash

### Backwards Compatibility
- No persisted data to migrate — only in-memory usage
- Breaking API changes are acceptable if they make the fix cleaner
- Existing tests: keep a mix of deterministic (seeded) and stochastic (random) tests

### Test Coverage
- Basic determinism testing: same seed = same result (not property-based)
- Collision tests against medium corpus (100+ words) at k=10
- Regression test structure: Claude's discretion

### Claude's Discretion
- Specific hash function choice for better distribution
- Test file organization (new file vs extend existing)
- Regression test naming/structure
- Exact error message wording for k < 10

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches for determinism and hashing.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-codebase-cleanup*
*Context gathered: 2026-02-01*
