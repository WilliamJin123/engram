---
phase: 03-advanced-dynamics
plan: 03
subsystem: memory
tags: [criticality, self-organization, tunneling, edge-of-chaos, feedback]

# Dependency graph
requires:
  - phase: 03-advanced-dynamics-01
    provides: Pure similarity retrieval, crystallization dynamics, stability tracking
  - phase: 03-advanced-dynamics-02
    provides: Tunneling mechanics, connection tracking, creative mode
provides:
  - Global criticality parameter with self-adjustment
  - Feedback-based system regulation
  - Criticality-modulated tunneling probability
  - Edge-of-chaos dynamics
affects: [phase-4-integration, system-tuning, emergent-behavior]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Criticality as global system state (0-1 scale)"
    - "Self-adjustment via feedback signals (quality + surprise)"
    - "Dampening to prevent oscillation"
    - "tunneling_amplification = 0.5 + criticality"

key-files:
  created:
    - src/quantum_substrate/criticality.py
  modified:
    - src/quantum_substrate/__init__.py
    - src/quantum_substrate/tunneling.py
    - src/agentic/memory_store.py

key-decisions:
  - "Initial criticality 0.5 (edge of chaos)"
  - "Adjustment rate 0.01, dampening 0.9"
  - "Rolling window size 20, min samples 5"
  - "Quality threshold 0.5, surprise targets 0.1-0.5"
  - "tunneling_amplification linear: 0.5 + value"
  - "Auto-adjust every 10 operations"

patterns-established:
  - "Criticality affects tunneling: higher criticality = more exploration"
  - "Surprise indicator: successful tunneling = discovery"
  - "System self-regulates toward 0.5 over time"
  - "is_healthy checks for non-extreme values (0.1-0.9)"

# Metrics
duration: 4min
completed: 2026-02-02
---

# Phase 3 Plan 3: Criticality Dynamics Summary

**Global criticality parameter (0-1) that self-adjusts based on retrieval quality and surprise frequency, controlling tunneling probability for edge-of-chaos dynamics**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-02T02:40:45Z
- **Completed:** 2026-02-02T02:44:26Z
- **Tasks:** 3
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments
- Created criticality.py module with CriticalityConfig and CriticalityState
- Implemented feedback tracking (retrieval quality + surprise) in rolling window
- Implemented self_adjust() with dampening to prevent oscillation
- Added tunneling_amplification property (0.5 + value)
- Integrated CriticalityState into MemoryStore with auto-adjustment
- Connected criticality to tunneling via criticality_amplification parameter

## Task Commits

Each task was committed atomically:

1. **Task 1: Create criticality module** - `b52696d` (feat)
2. **Task 2: Integrate criticality with MemoryStore** - `ed81a51` (feat)
3. **Task 3: Connect criticality to tunneling probability** - `62799da` (feat)

## Files Created/Modified
- `src/quantum_substrate/criticality.py` - New module: CriticalityConfig, CriticalityState
- `src/quantum_substrate/__init__.py` - Added exports for criticality components
- `src/quantum_substrate/tunneling.py` - Added criticality_amplification parameter to attempt_tunneling()
- `src/agentic/memory_store.py` - Added CriticalityState, feedback recording, auto-adjustment

## Decisions Made
- **Initial value:** 0.5 (optimal edge-of-chaos)
- **Adjustment mechanics:** rate=0.01, dampening=0.9 (slow, stable changes)
- **Feedback window:** 20 samples, minimum 5 before adjusting
- **Thresholds:** quality<0.5 triggers increase, surprise 0.1-0.5 is healthy range
- **Tunneling formula:** probability *= (0.5 + criticality_value)
- **Surprise indicator:** Successful tunneling counts as surprise (discovery)
- **Auto-adjust interval:** Every 10 retrieve_with_tunneling operations

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed successfully, all 153 existing tests pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 3 (Advanced Dynamics) complete
- Full self-organizing memory system with:
  - Coherence decay/refresh (Phase 1)
  - Surprise-based re-coherence (Phase 2)
  - Crystallization and stability (Plan 03-01)
  - Tunneling mechanics (Plan 03-02)
  - Criticality self-regulation (Plan 03-03)
- Ready for Phase 4: Integration and Testing

---
*Phase: 03-advanced-dynamics*
*Completed: 2026-02-02*
