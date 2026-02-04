---
phase: 06-test-infrastructure
plan: 01
subsystem: testing
tags: [noise-generation, near-miss, clutter, pytest, fixtures]

# Dependency graph
requires:
  - phase: 05-architecture
    provides: EvolvingPattern, coherence dynamics infrastructure
provides:
  - NoiseLevel enum with NONE/LOW/MEDIUM/HIGH presets
  - NoiseConfig dataclass for noise generation parameters
  - NoiseResult dataclass for tracking generated patterns
  - create_near_miss() and generate_near_misses() functions
  - create_clutter() and generate_clutter_batch() functions
affects: [06-02, 07-harder-behavior-tests, 08-metrics-validation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Seeded torch.Generator for reproducible noise generation"
    - "Factory functions returning EvolvingPattern instances"
    - "Fractional ratio handling with probabilistic rounding"

key-files:
  created:
    - tests/conftest.py
    - tests/noise_generators.py
  modified: []

key-decisions:
  - "Seed is REQUIRED in NoiseConfig - enforced by runtime check"
  - "Near-miss overlap fixed at 0.4 (must be < 0.5)"
  - "Clutter relatedness distribution: 50% distant, 30% mid, 20% close"
  - "Coherence ranges: near-miss (0.2-0.7), clutter (0.1-0.9)"

patterns-established:
  - "All noise functions accept torch.Generator for reproducibility"
  - "Noise patterns are full EvolvingPattern instances with coherence/aging"

# Metrics
duration: 9 min
completed: 2026-02-04
---

# Phase 6 Plan 01: Core Noise Types and Generators Summary

**NoiseLevel enum, NoiseConfig/NoiseResult dataclasses, and near-miss/clutter generator functions for stress-testing retrieval**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-04T22:45:30Z
- **Completed:** 2026-02-04T22:54:49Z
- **Tasks:** 3
- **Files created:** 2

## Accomplishments

- NoiseLevel enum providing NONE/LOW/MEDIUM/HIGH presets with configurable ratios
- NoiseConfig dataclass with validation (seed required, overlap < 0.5)
- Near-miss generator creating patterns with controlled bit overlap and phase perturbation
- Clutter generator with varying relatedness distribution (50% distant, 30% mid, 20% close)
- Full reproducibility via seeded torch.Generator throughout

## Task Commits

Each task was committed atomically:

1. **Task 1: Core noise configuration types** - `607d124` (feat)
2. **Task 2: Near-miss generator function** - `e0b10f4` (feat)
3. **Task 3: Clutter generator function** - included in `e0b10f4` (same file)

## Files Created/Modified

- `tests/conftest.py` - NoiseLevel enum, NoiseConfig dataclass, NoiseResult dataclass
- `tests/noise_generators.py` - create_near_miss(), generate_near_misses(), create_clutter(), generate_clutter_batch()

## Decisions Made

1. **Seed is mandatory**: NoiseConfig requires explicit seed with runtime enforcement - prevents accidental non-reproducible tests
2. **Near-miss overlap = 0.4**: Fixed default that's low enough to differentiate from targets but high enough to create challenge
3. **Clutter relatedness varies**: 50/30/20 distribution creates realistic interference landscape per CONTEXT.md
4. **Separate coherence ranges**: Near-misses (0.2-0.7) narrower than clutter (0.1-0.9) for realistic aging simulation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Core types and generators ready for Phase 6 Plan 02
- Plan 02 will implement inject_noise() function and noisy_memory pytest fixture
- Types are importable from tests.conftest, generators from tests.noise_generators

---
*Phase: 06-test-infrastructure*
*Completed: 2026-02-04*
