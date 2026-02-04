---
phase: 06-test-infrastructure
plan: 02
subsystem: testing
tags: [pytest, noise-generation, fixtures, stress-testing]

# Dependency graph
requires:
  - phase: 06-01
    provides: noise generator functions (generate_near_misses, generate_clutter_batch)
provides:
  - inject_noise() function for noise injection into existing MemoryStore
  - noisy_memory pytest fixture for test scenarios
  - 40 validation tests for noise generation (test_noise_generators.py)
affects: [07-stress-retrieval, 08-adaptive-coherence]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Factory fixture pattern (noisy_memory returns factory function)
    - Aged coherence injection (override store defaults post-storage)
    - Value-based enum lookup (avoid identity issues across module imports)

key-files:
  created:
    - tests/test_noise_generators.py
  modified:
    - tests/conftest.py

key-decisions:
  - "Value-based enum lookup in from_preset() to avoid pytest module identity issues"
  - "Duck typing for NoiseResult in tests to avoid isinstance cross-module issues"
  - "Override coherence/last_access_tick after store() to inject aged patterns"

patterns-established:
  - "inject_noise pattern: Generate EvolvingPatterns, store, then override coherence values"
  - "noisy_memory factory: Returns function accepting target_texts and noise config"

# Metrics
duration: 8min
completed: 2026-02-04
---

# Phase 6 Plan 02: Noise Injection and Fixture Summary

**inject_noise() function with aged coherence injection, noisy_memory pytest fixture, and 40 validation tests covering INFRA-01 through INFRA-04 requirements**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-04T23:05:00Z
- **Completed:** 2026-02-04T23:13:00Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- inject_noise() adds near-misses and clutter to existing MemoryStore with aged coherence
- noisy_memory fixture creates complete test scenarios with configurable noise levels
- 40 comprehensive validation tests covering all INFRA requirements
- Fixed enum identity issue in from_preset() using value-based lookup

## Task Commits

Each task was committed atomically:

1. **Task 1: inject_noise function** - `e50aeaa` (feat)
2. **Task 2: noisy_memory pytest fixture** - `01a3894` (feat)
3. **Task 3: Noise generation validation tests** - `ba13499` (test)

## Files Created/Modified

- `tests/conftest.py` - Added inject_noise() function and noisy_memory fixture (339 lines total)
- `tests/test_noise_generators.py` - New file with 40 validation tests (490 lines)

## Decisions Made

1. **Value-based enum lookup** - Changed from_preset() to use `level.value` as dictionary key instead of enum identity. This avoids pytest module import identity issues where the same enum value appears as different objects.

2. **Duck typing for NoiseResult** - Tests use `hasattr()` checks instead of `isinstance()` to verify NoiseResult, avoiding cross-module class identity issues.

3. **Post-store coherence override** - inject_noise() calls store.store() (which sets coherence=1.0), then immediately overrides the pattern's coherence and last_access_tick with aged values from the generated pattern. This maintains proper MemoryStore state while injecting realistic aged patterns.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed enum identity issue in from_preset()**
- **Found during:** Task 3 (test execution)
- **Issue:** Tests importing NoiseLevel from tests.conftest got different enum instances than conftest.py internal references, causing KeyError on preset lookup
- **Fix:** Changed dictionary keys from `NoiseLevel.MEDIUM` to `"medium"` and access via `level.value`
- **Files modified:** tests/conftest.py
- **Verification:** All 40 tests pass
- **Committed in:** ba13499 (Task 3 commit)

**2. [Rule 1 - Bug] Fixed isinstance() check for NoiseResult**
- **Found during:** Task 3 (test execution)
- **Issue:** `isinstance(result, NoiseResult)` failed due to different class objects across module imports
- **Fix:** Changed to duck typing with hasattr() checks
- **Files modified:** tests/test_noise_generators.py
- **Verification:** test_fixture_creates_store_with_noise passes
- **Committed in:** ba13499 (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs)
**Impact on plan:** Both fixes were necessary for test correctness. No scope creep.

## Issues Encountered

None beyond the auto-fixed bugs above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Noise generation infrastructure complete and validated
- 40 tests provide regression protection for INFRA-01 through INFRA-04
- Ready for Phase 7 (Stress Retrieval Tests) to use noisy_memory fixture
- inject_noise() available for custom test scenarios

---
*Phase: 06-test-infrastructure*
*Completed: 2026-02-04*
