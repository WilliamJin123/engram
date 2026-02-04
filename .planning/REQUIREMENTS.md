# Requirements: Engram v1.1 Harder Test Cases

**Defined:** 2026-02-04
**Core Value:** Engineer the substrate. Let everything else emerge.

## v1.1 Requirements

Requirements for this milestone. Address the v1.0 ceiling effect by stress-testing retrieval mechanisms with realistic memory noise.

### Test Infrastructure

- [x] **INFRA-01**: Noise generator creates semantic near-miss patterns (similar but not related)
- [x] **INFRA-02**: Noise generator creates random clutter patterns (unrelated filler)
- [x] **INFRA-03**: Reusable noisy memory fixture with configurable noise composition
- [x] **INFRA-04**: Parameterized noise levels (none/low/medium/high) for test scaling

### Harder Behavior Tests

- [ ] **TEST-01**: Tunneling tests with noisy memory (high-coherence retrieval through clutter)
- [ ] **TEST-02**: Interference retrieval tests with noisy memory (find relevant among noise)
- [ ] **TEST-03**: Coherence dynamics tests with noisy memory (decay/refresh/surprise under load)

### Metrics & Validation

- [ ] **METR-01**: Differentiation test: interference retrieval vs cosine similarity baseline
- [ ] **METR-02**: Differentiation test: interference retrieval vs random retrieval baseline
- [ ] **METR-03**: Graceful degradation curves: measure performance at each noise level

## Future Requirements

Deferred to later milestones.

### Production Readiness

- **PROD-01**: Persistence layer for memory storage
- **PROD-02**: LSH indexing for O(1) retrieval
- **PROD-03**: Real LLM integration testing

## Out of Scope

Explicitly excluded from v1.1.

| Feature | Reason |
|---------|--------|
| New substrate mechanics | v1.1 is validation-focused, not feature-focused |
| Performance optimizations | Focus on correctness validation first |
| Fix n-gram encoding | Defer; not needed for noise testing |
| Pattern pruning/forgetting | Needed eventually, not for test validation |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 6 | Complete |
| INFRA-02 | Phase 6 | Complete |
| INFRA-03 | Phase 6 | Complete |
| INFRA-04 | Phase 6 | Complete |
| TEST-01 | Phase 7 | Pending |
| TEST-02 | Phase 7 | Pending |
| TEST-03 | Phase 7 | Pending |
| METR-01 | Phase 8 | Pending |
| METR-02 | Phase 8 | Pending |
| METR-03 | Phase 8 | Pending |

**Coverage:**
- v1.1 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0

---
*Requirements defined: 2026-02-04*
*Last updated: 2026-02-04 after roadmap creation*
