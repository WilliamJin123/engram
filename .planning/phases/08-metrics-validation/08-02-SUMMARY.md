---
phase: 08-metrics-validation
plan: 02
subsystem: testing
tags: [baseline-comparison, mann-whitney, mrr, recall, statistical-significance, cosine, random, recency]

# Dependency graph
requires:
  - phase: 08-01
    provides: MRR, Recall@K, compare_methods, metrics module
  - phase: 07-harder-behavior-tests
    provides: noisy_memory fixture, stress test infrastructure
provides:
  - Baseline comparison tests for METR-01/METR-02
  - Multi-trial statistical significance testing
  - Clean memory reference baseline
  - Comprehensive all-methods comparison
affects: [08-03, validation-report, hypothesis-validation]

# Tech tracking
tech-stack:
  added: []
  patterns: [multi-trial comparison, numpy-to-python bool conversion]

key-files:
  created:
    - tests/stress/test_baseline_comparison.py
  modified: []

key-decisions:
  - "Interference vs cosine: no significant difference (both achieve MRR 1.0)"
  - "Interference vs random: significantly better (p < 0.001)"
  - "Interference vs recency: no significant difference (recency favors fresh targets)"
  - "Convert numpy bool to Python bool for JSON serialization"

patterns-established:
  - "_get_rank_from_tuples: Convert baseline (tuple) results to rank format"
  - "run_comparison_trials: Multi-trial comparison with seeded RNG per trial"
  - "METRICS JSON: Standardized output format for report parsing"

# Metrics
duration: 6min
completed: 2026-02-05
---

# Phase 8 Plan 2: Baseline Comparison Tests Summary

**Multi-trial baseline comparison tests measuring interference retrieval against cosine, random, and recency baselines with Mann-Whitney U statistical significance**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-05T02:29:00Z
- **Completed:** 2026-02-05T02:35:00Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments
- METR-01: Interference vs cosine comparison with MRR and p-value
- METR-02: Interference vs random comparison (significantly better, p < 0.001)
- Bonus: Interference vs recency comparison
- Comprehensive all-methods summary with Recall@3, Recall@5
- Clean memory baseline for reference performance

## Task Commits

Each task was committed atomically:

1. **Task 1: Create baseline comparison test file with helper functions** - `5ea0839` (feat)
2. **Task 2: Implement baseline comparison tests (METR-01, METR-02)** - `435e951` (feat)
3. **Task 3: Add all-methods comparison summary test** - `f153e0b` (feat)

## Files Created/Modified
- `tests/stress/test_baseline_comparison.py` - 396 lines with 9 tests covering all baseline comparisons

## Test Summary

| Test | MEDIUM Noise | HIGH Noise |
|------|-------------|------------|
| interference_vs_cosine | No sig. diff (MRR 1.0 vs 1.0) | No sig. diff (MRR 1.0 vs 1.0) |
| interference_vs_random | Significantly better (p < 0.001) | Significantly better (p < 0.001) |
| interference_vs_recency | No sig. diff | No sig. diff |
| all_methods_comparison | any_baseline_beaten: true | any_baseline_beaten: true |
| clean_memory_baseline | Reference (all methods MRR 1.0) | N/A |

Key findings:
- Interference significantly outperforms random (floor baseline) - validates basic retrieval
- Cosine achieves same MRR as interference (both find exact match at rank 1)
- Recency baseline performs similarly because targets are freshly stored

## Decisions Made

1. **Numpy bool conversion**: Compare_methods returns numpy bool from scipy, must convert with `bool()` for JSON serialization
2. **Trial count flexibility**: 30 trials with `--update-report` flag, 10 trials for faster CI
3. **Seeded RNG per trial**: `seed=trial * 100` ensures reproducible but varied noise per trial

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed JSON serialization of numpy booleans**
- **Found during:** Task 2 (running tests)
- **Issue:** `comparison["significant_at_0.05"]` returns numpy bool, not JSON serializable
- **Fix:** Added explicit `bool()` conversion before passing to StressMetrics
- **Files modified:** tests/stress/test_baseline_comparison.py
- **Verification:** All tests pass, METRICS JSON output valid
- **Committed in:** 435e951 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary for test output functionality. No scope creep.

## Issues Encountered
None - plan executed smoothly after JSON serialization fix.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Baseline comparison infrastructure complete
- Ready for 08-03: Degradation curve validation tests
- All METR-01/METR-02 requirements validated
- METRICS JSON format established for report parsing

---
*Phase: 08-metrics-validation*
*Completed: 2026-02-05*
