---
phase: 05-architecture-clarity
plan: 02
subsystem: testing
tags: [scipy, pingouin, hypothesis-testing, statistical-tests, baselines]

# Dependency graph
requires:
  - phase: 05-01
    provides: SubstratePattern protocol for type hints
provides:
  - Statistical test infrastructure (scipy, pingouin)
  - Shared fixtures (seeded_rng, statistical_comparator)
  - Baseline retrieval methods (jaccard, cosine, random)
affects: [05-03, future hypothesis validation tests]

# Tech tracking
tech-stack:
  added: [scipy>=1.10.0, pingouin>=0.5.0, numpy>=1.20.0]
  patterns: [Welch's t-test for method comparison, Cohen's d effect size]

key-files:
  created:
    - tests/hypothesis_validation/__init__.py
    - tests/hypothesis_validation/conftest.py
    - tests/hypothesis_validation/baselines.py
  modified:
    - pyproject.toml

key-decisions:
  - "N_TRIALS = 50 default (balance between statistical power and CI speed)"
  - "Welch's t-test (not Student's) - no equal variance assumption"
  - "Cohen's d via pingouin - handles edge cases properly"

patterns-established:
  - "seeded_rng fixture factory for reproducible tests"
  - "statistical_comparator returns dict with significance + effect size"
  - "Baselines return List[Tuple[str, float]] matching interference_retrieval"

# Metrics
duration: 8min
completed: 2026-02-04
---

# Phase 5 Plan 2: Hypothesis Test Infrastructure Summary

**Statistical testing infrastructure with scipy/pingouin fixtures and classical baselines for quantum method comparison**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-04T12:00:00Z
- **Completed:** 2026-02-04T12:08:00Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added scipy, pingouin, numpy as optional dev dependencies
- Created hypothesis_validation test directory with fixtures
- Implemented Jaccard, cosine, and random baselines for comparison
- All baselines return same format as interference_retrieval

## Task Commits

Each task was committed atomically:

1. **Task 1: Add hypothesis test dependencies** - `3b6ee6d` (chore)
2. **Task 2: Create hypothesis validation fixtures** - `a140222` (feat)
3. **Task 3: Implement baseline retrieval methods** - `040e12e` (feat)

## Files Created/Modified

- `pyproject.toml` - Added hypothesis and dev optional dependencies
- `tests/hypothesis_validation/__init__.py` - Package docstring explaining test philosophy
- `tests/hypothesis_validation/conftest.py` - seeded_rng, statistical_comparator, memory_store_factory fixtures
- `tests/hypothesis_validation/baselines.py` - jaccard_similarity, cosine_similarity_sparse, retrieval functions

## Decisions Made

1. **N_TRIALS = 50** - Balance between statistical power and CI speed. 100 trials provides more power but doubles test time. 50 is sufficient to detect medium effects.

2. **Welch's t-test over Student's** - Does not assume equal variance between groups, more robust for comparing methods with potentially different score distributions.

3. **Cohen's d via pingouin** - Library handles edge cases (zero variance, small samples) that manual calculation misses.

4. **Baselines use Set[int] bits** - Works with SubstratePattern protocol, different from interference.py which uses ComplexSparsePattern indices. This enables testing EvolvingPattern-based retrieval.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed pip in venv**
- **Found during:** Task 1 (dependency installation)
- **Issue:** venv missing pip module, `python -m pip install` failed
- **Fix:** Ran `python -m ensurepip --upgrade`
- **Files modified:** None (venv internal)
- **Verification:** Subsequent pip install succeeded
- **Impact:** None - standard venv bootstrap

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minor infrastructure fix. No scope change.

## Issues Encountered

- Python 3.14 venv created without pip - resolved with ensurepip

## User Setup Required

None - dependencies install with `pip install -e ".[dev]"`.

## Next Phase Readiness

- Infrastructure ready for TEST-06 (hypothesis validation tests)
- Fixtures available: seeded_rng, statistical_comparator, memory_store_factory
- Baselines available: jaccard_retrieval, cosine_retrieval, random_retrieval
- Ready to implement actual comparison tests in 05-03

---
*Phase: 05-architecture-clarity*
*Completed: 2026-02-04*
