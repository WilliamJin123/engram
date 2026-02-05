# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-04)

**Core value:** Engineer the substrate. Let everything else emerge.
**Current focus:** v1.1 Harder Test Cases - Phase 8 in progress

## Current Position

Phase: 8 of 8 (Metrics & Validation)
Plan: 1 of 3 complete in Phase 8
Status: In progress
Last activity: 2026-02-05 - Completed 08-01-PLAN.md

Progress: v1.1 Phases 6-8 [##################--] 90% (6/7 plans)
Progress: v1.1 Overall [##############--] 86% (6/7 plans)

## Milestone Summary

**v1.0 Coherence** shipped 2026-02-04:
- 6 phases, 17 plans
- 92 commits, 11,445 LOC Python
- 287 tests passing
- 4 days execution

**v1.1 Harder Test Cases** started 2026-02-04:
- 3 phases (6-8), 10 requirements
- Addresses ceiling effect from v1.0 hypothesis validation
- Phase 6 (Test Infrastructure) complete
- Phase 7 (Stress Retrieval Tests) COMPLETE
- Phase 8 (Metrics & Validation) in progress - Plan 1 complete

## Phase 7 Test Results Summary

See: .planning/phases/07-harder-behavior-tests/TEST_SUMMARY.md

Key metrics for Phase 8:
- **Degradation curve**: Linear (rank 1->6->9 for 1->3->5 near-misses)
- **Success criteria**: Strict 0%, Relaxed 50%/0%, Relative 0%
- **Coherence decay**: 0.59 (MEDIUM), 0.77 (HIGH) over 30 ops
- **Tunneling**: Success drops from ~20% (MEDIUM) to 0% (HIGH)

## Accumulated Context

### Decisions

See PROJECT.md Key Decisions table for full history.

v1.0 key decisions preserved for context:
- Coherence per-pattern (not per-bit)
- Decay rate 0.05, floor 0.01
- Stability score = access_count / 10
- Tunneling probability 0.1, creative mode 3x
- Criticality 0.5 initial, 0.01 adjustment rate

v1.1 Phase 6 decisions:
- NoiseConfig seed is mandatory (runtime enforced)
- Near-miss overlap = 0.4 (< 0.5 to differentiate from targets)
- Clutter relatedness: 50% distant, 30% mid, 20% close
- Value-based enum lookup in from_preset() for pytest module identity safety
- Post-store coherence override pattern for aged pattern injection

v1.1 Phase 7 decisions:
- Use pytest.skip for probabilistic limitations, not hard assertions
- Document degradation as linear (rank 1->6->9) not cliff-like
- Coherence weighting favors near-misses - documented as Phase 8 improvement area
- Crystallization effect not observable - needs Phase 8 investigation
- Surprise detection targets clutter under noise - documented limitation

v1.1 Phase 8 decisions:
- MRR as primary metric (single correct answer per query)
- Mann-Whitney U test for non-parametric rank comparison
- fill_between for shaded confidence bands in plots
- Recency score = 1/(1+age) for time-based ranking

### Pending Todos

None.

### Blockers/Concerns

Deferred to v2:
- Patterns grow unbounded until max_bits; no backpressure
- N-gram encoding disabled (broken)

Phase 8 improvements identified:
- Coherence weighting in interference retrieval favors aged near-misses
- Consider separating "recency" from "quality" in retrieval scoring
- Investigate why crystallization_factor doesn't produce observable decay differences
- Consider improving surprise detection to target "forgotten" patterns rather than clutter

## Session Continuity

Last session: 2026-02-05T02:30:00Z
Stopped at: Completed 08-01-PLAN.md
Resume file: None

## Next Steps

1. Execute 08-02-PLAN.md (baseline comparison tests)
2. Execute 08-03-PLAN.md (degradation curve validation)
3. Complete v1.1 milestone

---
*State initialized: 2026-01-31*
*Last updated: 2026-02-05 - Completed 08-01-PLAN.md (Phase 8 Plan 1 complete)*
