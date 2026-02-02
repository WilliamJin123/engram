---
phase: 04-codebase-cleanup
plan: 01
subsystem: agentic
tags: [random, determinism, rng, coactivation, testing]

# Dependency graph
requires:
  - phase: 03-advanced-dynamics
    provides: tunneling.py with seeded RNG pattern
provides:
  - Seeded RNG support for coactivation
  - Determinism tests for coactivation
  - Reproducible bit transfer for debugging
affects: [testing, debugging, coactivation-related]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Seeded RNG fallback: _rng = rng if rng is not None else random"
    - "Pass rng through function call chain"

key-files:
  created: []
  modified:
    - src/agentic/coactivation.py
    - tests/agentic/test_coactivation.py

key-decisions:
  - "Follow tunneling.py RNG pattern exactly (rng parameter, _rng fallback)"
  - "Pass rng to _transfer_bits helper to maintain determinism through full call chain"

patterns-established:
  - "Seeded RNG for deterministic testing: all stochastic functions accept optional rng parameter"

# Metrics
duration: 8min
completed: 2026-02-02
---

# Phase 4 Plan 1: Seeded RNG for Coactivation Summary

**Deterministic coactivation via optional seeded RNG parameter, following tunneling.py pattern**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-02
- **Completed:** 2026-02-02
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added `rng: random.Random | None = None` parameter to coactivate() function
- Modified _transfer_bits() to use provided RNG for deterministic bit selection
- Added 3 determinism tests proving seeded RNG works correctly
- All 295 tests pass (no regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add seeded RNG to coactivation** - `fe78cac` (feat)
2. **Task 2: Add determinism tests** - `b338d66` (test)

## Files Created/Modified
- `src/agentic/coactivation.py` - Added rng parameter to coactivate() and _transfer_bits(); uses _rng.sample() for deterministic bit selection
- `tests/agentic/test_coactivation.py` - Added TestCoactivationDeterminism class with 3 tests

## Decisions Made
- Followed tunneling.py pattern exactly: optional rng parameter with None default, local _rng variable that falls back to global random module
- Pass _rng through to _transfer_bits() to maintain determinism through full coactivation process

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - straightforward implementation following established pattern.

## Next Phase Readiness
- FIX-01 (random bit transfer non-determinism) is now resolved
- Coactivation tests can be made deterministic by passing seeded RNG
- Ready for 04-02 (text encoding improvements) or other Phase 4 plans

---
*Phase: 04-codebase-cleanup*
*Completed: 2026-02-02*
