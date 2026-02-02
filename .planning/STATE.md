# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-31)

**Core value:** Engineer the substrate. Let everything else emerge.
**Current focus:** Phase 3 - Advanced Dynamics

## Current Position

Phase: 3 of 5 (Advanced Dynamics)
Plan: 1 of 3 in current phase
Status: In progress
Last activity: 2026-02-02 — Completed 03-01-PLAN.md (Crystallization Dynamics)

Progress: [#######░░░] 54% (7/13 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 8 min
- Total execution time: 51 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 0. Documentation | 1 | - | - |
| 1. Coherence Foundation | 2 | 17 min | 9 min |
| 2. Coherence Effects | 3 | 30 min | 10 min |
| 3. Advanced Dynamics | 1 | 4 min | 4 min |

**Recent Trend:**
- Last 5 plans: [01-02, 02-01, 02-02, 02-03, 03-01]
- Trend: Consistent

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Phase 0]: Coherence per-pattern, not per-bit (simpler; per-bit adds complexity without clear benefit)
- [Phase 0]: HRR over tensor product for binding (O(n) vs O(n^2))
- [01-01]: Decay rate 0.05 (half-life ~14 ticks) as default
- [01-01]: Coherence floor 0.01 (never fully decohere)
- [01-01]: Embeddedness = 1.0 + 0.1 * connection_count
- [01-01]: Refresh uses diminishing returns (50% of headroom per activation)
- [01-02]: Refresh proportional to retrieval score (not fixed 1.0)
- [01-02]: connection_count increments even when no bits transfer
- [02-01]: Coherence weighting uses weight = coherence^exponent (default exponent=1.0)
- [02-01]: No hard cutoffs - floor-coherence patterns still contribute proportionally
- [02-02]: Surprise magnitude = normalized Hamming distance (symmetric_diff / union)
- [02-02]: Re-coherence scaling: surprising=1.0x, expected=0.5x, participants=0.3x*involvement
- [02-02]: Minimum surprise threshold 0.001 to skip trivial updates
- [03-01]: Stability score = min(1.0, access_count / 10.0) - 10 accesses = max stability
- [03-01]: Crystallization multiplier = 1.0 + factor * stability
- [03-01]: Re-coherence scaled by coherence (malleability) with min_delta=0.01 floor
- [03-01]: Pure similarity retrieval uses Jaccard only - no embeddedness weighting

### Pending Todos

None yet.

### Blockers/Concerns

From CONCERNS.md analysis:
- Random bit transfer in coactivation is non-deterministic (FIX-01, Phase 4)
- Text encoding collision at small k values (FIX-02, Phase 4)
- Patterns grow unbounded until max_bits; no backpressure (v2 scope)

## Session Continuity

Last session: 2026-02-02
Stopped at: Completed 03-01-PLAN.md (Crystallization Dynamics)
Resume file: None

---
*State initialized: 2026-01-31*
*Last updated: 2026-02-02*
