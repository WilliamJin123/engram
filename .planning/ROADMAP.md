# Roadmap: Engram Coherence Milestone

## Overview

This milestone implements coherence dynamics for the quantum-inspired memory substrate. Starting from a working foundation (HRR binding, phase encoding, interference retrieval), we add the coherence field that governs quantum vs classical behavior. Coherence decay, refresh, surprise-triggered re-coherence, coherence-modulated interference, and tunneling complete the substrate dynamics described in quantum_proposal.md. Parallel tracks fix known determinism bugs and establish clear architectural boundaries between primitive substrate and agentic layer.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 0: Documentation** - Project planning artifacts (DOC-01, DOC-02, DOC-03, DOC-04)
- [x] **Phase 1: Coherence Foundation** - Core coherence field with decay and refresh
- [x] **Phase 2: Coherence Effects** - Surprise re-coherence and interference modulation
- [x] **Phase 3: Advanced Dynamics** - Tunneling and criticality tuning
- [x] **Phase 4: Codebase Cleanup** - Determinism fixes (parallel track)
- [ ] **Phase 5: Architecture Clarity** - Substrate/agentic separation and hypothesis validation

## Phase Details

### Phase 0: Documentation
**Goal**: Establish project planning artifacts for milestone tracking
**Depends on**: Nothing (first phase)
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04
**Success Criteria** (what must be TRUE):
  1. PROJECT.md captures core value, requirements, constraints, and decisions
  2. REQUIREMENTS.md has traceable requirement IDs with v1/v2 separation
  3. ROADMAP.md maps all requirements to phases with success criteria
  4. STATE.md provides session continuity and project memory
**Plans**: 1 plan (completed via project initialization)

Plans:
- [x] 00-01: Project initialization and planning

### Phase 1: Coherence Foundation
**Goal**: Patterns have a coherence field that decays over time and refreshes on access
**Depends on**: Phase 0
**Requirements**: COHR-01, COHR-02, COHR-03, TEST-01, TEST-02
**Success Criteria** (what must be TRUE):
  1. Every pattern has a coherence value (0-1 scalar) accessible as a property
  2. Pattern coherence decays toward 0 following exponential decay when not accessed
  3. Accessing a pattern (via retrieve or explicit access) increases its coherence
  4. Tests mathematically validate decay follows spec: coherence *= exp(-decay_rate * dt)
  5. Tests validate refresh mechanism restores coherence without exceeding 1.0
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Add coherence field to EvolvingPattern and create CoherenceManager
- [x] 01-02-PLAN.md — Wire CoherenceManager into operations and validate with tests

### Phase 2: Coherence Effects
**Goal**: Coherence modulates interference and surprise re-coheres decayed patterns
**Depends on**: Phase 1
**Requirements**: COHR-04, COHR-05, TEST-03, TEST-04
**Success Criteria** (what must be TRUE):
  1. Low-coherence patterns contribute weakly to interference retrieval (near-zero weight when coherence approaches 0)
  2. High-coherence patterns dominate interference results
  3. When a pattern experiences contradiction/surprise, its coherence increases significantly
  4. Tests demonstrate coherence-weighted interference produces better retrieval than uniform weighting
  5. Tests validate that surprise detection triggers re-coherence on specific patterns
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Coherence-weighted interference retrieval (COHR-05, TEST-04)
- [x] 02-02-PLAN.md — Surprise detection module and re-coherence method (COHR-04 foundation)
- [x] 02-03-PLAN.md — Surprise integration and re-coherence tests (COHR-04, TEST-03)

### Phase 3: Advanced Dynamics
**Goal**: Fix coherence semantics (malleability not accessibility), add tunneling and criticality
**Depends on**: Phase 2
**Requirements**: COHR-08, COHR-06, COHR-07, TEST-05
**Success Criteria** (what must be TRUE):
  1. Retrieval is weighted by embeddedness/connections, NOT coherence
  2. Coherence governs only malleability (how much surprise needed to change a pattern)
  3. Low-coherence (crystallized) patterns are still easily retrievable if well-connected
  4. High-coherence patterns can tunnel to weakly-related patterns (creative retrieval)
  5. Tunneling strength is proportional to source pattern coherence
  6. System-wide criticality parameter controls the order/chaos balance
  7. Tests validate the new retrieval model and tunneling behavior
**Plans**: 4 plans in 3 waves

Plans:
- [x] 03-01-PLAN.md — Refactor retrieval to pure similarity + crystallization dynamics (COHR-08)
- [x] 03-02-PLAN.md — Implement tunneling mechanism (COHR-06)
- [x] 03-03-PLAN.md — Implement criticality parameter and self-adjustment (COHR-07)
- [x] 03-04-PLAN.md — Tests for advanced dynamics (TEST-05)

### Phase 4: Codebase Cleanup
**Goal**: Fix determinism bugs in existing code
**Depends on**: Phase 0 (can run parallel to Phases 1-3)
**Requirements**: FIX-01, FIX-02
**Success Criteria** (what must be TRUE):
  1. Coactivation produces identical bit transfers given same seed
  2. Running coactivation twice with same inputs and seed produces identical patterns
  3. Text encoding at k=10 produces unique bit patterns for different single-word inputs (k < 10 rejected)
  4. All existing tests continue to pass
**Plans**: 2 plans in 1 wave (parallel)

Plans:
- [x] 04-01-PLAN.md — Add seeded RNG to coactivation (FIX-01)
- [x] 04-02-PLAN.md — Fix text encoding collision at small k (FIX-02)

### Phase 5: Architecture Clarity
**Goal**: Clear separation between substrate and agentic layers; validate quantum memory hypothesis
**Depends on**: Phases 3 and 4
**Requirements**: ARCH-01, ARCH-02, ARCH-03, TEST-06
**Success Criteria** (what must be TRUE):
  1. Primitive substrate can be imported and used without any LLM dependency
  2. Agentic layer builds on substrate through documented public interfaces
  3. Architectural boundaries are documented and enforced by module structure
  4. Tests exist that can validate OR invalidate the quantum memory hypothesis
  5. Hypothesis validation tests have clear pass/fail criteria based on theory predictions
**Plans**: 3 plans in 3 waves

**Intuition Tests (from INTUITION.md)** — tests must address these intelligent behaviors:
  - Generalization: repeated exposure -> shared concept with attributes
  - Inheritance: shared attributes stored efficiently, not duplicated
  - Exceptions: specific instances can override general patterns
  - Certainty plasticity: LOW-coherence (crystallized) nodes hard to change, HIGH-coherence (malleable) easy
  - Conditionals: "if X then Y" represented as connected nodes
  - Meta-relationships: analogies and reasoning about relationships
  - Provenance: tracking where knowledge was learned (when relevant)
  - History: how understandings change over time and why

Plans:
- [ ] 05-01-PLAN.md — Create SubstratePattern protocol and refactor substrate for LLM-independence
- [ ] 05-02-PLAN.md — Set up hypothesis test infrastructure (deps, fixtures, baselines)
- [ ] 05-03-PLAN.md — Create hypothesis validation test suite (including INTUITION.md tests)

## Progress

**Execution Order:**
Phases execute in numeric order. Phase 4 can run parallel to Phases 1-3.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 0. Documentation | 1/1 | Complete | 2026-01-31 |
| 1. Coherence Foundation | 2/2 | Complete | 2026-02-01 |
| 2. Coherence Effects | 3/3 | Complete | 2026-02-01 |
| 3. Advanced Dynamics | 4/4 | Complete | 2026-02-02 |
| 4. Codebase Cleanup | 2/2 | Complete | 2026-02-01 |
| 5. Architecture Clarity | 0/3 | Planned | - |

---
*Roadmap created: 2026-01-31*
*Phase 1 planned: 2026-02-01*
*Phase 1 completed: 2026-02-01*
*Phase 2 planned: 2026-02-01*
*Phase 2 completed: 2026-02-01*
*Phase 3 planned: 2026-02-01*
*Phase 3 completed: 2026-02-02*
*Phase 4 planned: 2026-02-02*
*Phase 4 completed: 2026-02-01*
*Phase 5 planned: 2026-02-04*
