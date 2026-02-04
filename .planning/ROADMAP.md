# Roadmap: Engram

## Milestones

- [x] **v1.0 Coherence** - Phases 0-5 (shipped 2026-02-04)
- [ ] **v1.1 Harder Test Cases** - Phases 6-8 (in progress)

## Phases

<details>
<summary>v1.0 Coherence (Phases 0-5) - SHIPPED 2026-02-04</summary>

### Phase 0: Foundation
**Goal**: Establish project structure and core substrate patterns
**Plans**: 3 plans

Plans:
- [x] 00-01: Project setup and dependencies
- [x] 00-02: Core sparse pattern implementation
- [x] 00-03: Basic interference retrieval

### Phase 1: HRR Binding
**Goal**: Implement holographic reduced representation for role-filler binding
**Plans**: 3 plans

Plans:
- [x] 01-01: HRR circular convolution binding
- [x] 01-02: Role-filler recovery with correlation
- [x] 01-03: Binding accuracy validation

### Phase 2: Phase Sequences
**Goal**: Enable temporal ordering through phase encoding
**Plans**: 2 plans

Plans:
- [x] 02-01: Phase sequence encoding
- [x] 02-02: Ordering validation to n=100

### Phase 3: Coactivation Learning
**Goal**: Implement Hebbian-style pattern strengthening
**Plans**: 3 plans

Plans:
- [x] 03-01: Bidirectional bit transfer
- [x] 03-02: Obesity tracking and limits
- [x] 03-03: Determinism fixes (seeded RNG)

### Phase 4: Coherence Dynamics
**Goal**: Full coherence field with decay, refresh, and re-coherence
**Plans**: 3 plans

Plans:
- [x] 04-01: Coherence field and decay
- [x] 04-02: Access refresh and surprise re-coherence
- [x] 04-03: Coherence-modulated interference

### Phase 5: Architecture Clarity
**Goal**: Clean substrate layer with tunneling and criticality
**Plans**: 3 plans

Plans:
- [x] 05-01: SubstratePattern protocol
- [x] 05-02: Tunneling and criticality mechanisms
- [x] 05-03: Hypothesis validation test suite

</details>

### v1.1 Harder Test Cases (In Progress)

**Milestone Goal:** Stress-test retrieval mechanisms with realistic memory noise to resolve the v1.0 ceiling effect and validate that interference-based retrieval outperforms baselines under challenging conditions.

- [ ] **Phase 6: Test Infrastructure** - Noise generators and reusable fixtures
- [ ] **Phase 7: Harder Behavior Tests** - Tunneling, interference, and coherence with noise
- [ ] **Phase 8: Metrics & Validation** - Baseline comparisons and degradation curves

## Phase Details

### Phase 6: Test Infrastructure
**Goal**: Noise generation toolkit for stress-testing retrieval
**Depends on**: Phase 5 (v1.0 complete)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04
**Success Criteria** (what must be TRUE):
  1. Noise generator produces semantic near-miss patterns that are similar to targets but semantically unrelated
  2. Noise generator produces random clutter patterns that fill memory with unrelated content
  3. Test fixture accepts noise composition parameters (near-miss ratio, clutter ratio)
  4. Noise levels (none/low/medium/high) are parameterized and produce consistent, reproducible results
**Plans**: 2 plans

Plans:
- [x] 06-01-PLAN.md — Core noise types and generator functions (NoiseLevel, NoiseConfig, near-miss/clutter generators)
- [ ] 06-02-PLAN.md — inject_noise function, noisy_memory fixture, and validation tests

### Phase 7: Harder Behavior Tests
**Goal**: Validate core behaviors under realistic memory load
**Depends on**: Phase 6
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. Tunneling tests demonstrate high-coherence retrieval navigating through clutter to find related patterns
  2. Interference retrieval tests show target patterns found among semantic near-misses and random noise
  3. Coherence dynamics tests validate decay, refresh, and surprise re-coherence behavior under load
  4. All existing INTUITION.md behavior tests continue to pass
**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD

### Phase 8: Metrics & Validation
**Goal**: Quantify differentiation from baselines and degradation behavior
**Depends on**: Phase 7
**Requirements**: METR-01, METR-02, METR-03
**Success Criteria** (what must be TRUE):
  1. Differentiation test shows measurable gap between interference retrieval and cosine similarity baseline
  2. Differentiation test shows measurable gap between interference retrieval and random retrieval baseline
  3. Graceful degradation curves demonstrate performance at none/low/medium/high noise levels
  4. Results are documented with specific metrics (precision, recall, or rank-based measures)
**Plans**: TBD

Plans:
- [ ] 08-01: TBD
- [ ] 08-02: TBD

## Progress

**Execution Order:** 6 -> 7 -> 8

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 0. Foundation | v1.0 | 3/3 | Complete | 2026-02-01 |
| 1. HRR Binding | v1.0 | 3/3 | Complete | 2026-02-01 |
| 2. Phase Sequences | v1.0 | 2/2 | Complete | 2026-02-02 |
| 3. Coactivation | v1.0 | 3/3 | Complete | 2026-02-03 |
| 4. Coherence | v1.0 | 3/3 | Complete | 2026-02-03 |
| 5. Architecture | v1.0 | 3/3 | Complete | 2026-02-04 |
| 6. Test Infrastructure | v1.1 | 1/2 | In progress | - |
| 7. Harder Behavior Tests | v1.1 | 0/? | Not started | - |
| 8. Metrics & Validation | v1.1 | 0/? | Not started | - |

---
*Roadmap created: 2026-02-04*
*Last updated: 2026-02-04*
