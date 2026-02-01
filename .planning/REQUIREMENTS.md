# Requirements: Engram Coherence Milestone

**Defined:** 2026-01-31
**Core Value:** Engineer the substrate. Let everything else emerge.

## v1 Requirements

Requirements for completing coherence dynamics and establishing clean architecture.

### Coherence Dynamics

- [ ] **COHR-01**: Pattern has coherence field (0-1 scalar)
- [ ] **COHR-02**: Coherence decays toward 0 over time without interaction
- [ ] **COHR-03**: Accessing a pattern refreshes its coherence
- [ ] **COHR-04**: Surprise/contradiction re-coheres decayed patterns
- [ ] **COHR-05**: Interference strength is modulated by pattern coherence
- [ ] **COHR-06**: High-coherence patterns can tunnel to weakly-related patterns
- [ ] **COHR-07**: System-wide criticality parameter tunes order vs chaos balance

### Validation Tests

- [ ] **TEST-01**: Tests validate coherence decay behavior mathematically
- [ ] **TEST-02**: Tests validate coherence refresh on access
- [ ] **TEST-03**: Tests validate surprise-triggered re-coherence
- [ ] **TEST-04**: Tests validate coherence-modulated interference produces better retrieval
- [ ] **TEST-05**: Tests validate tunneling enables creative/exploratory activation
- [ ] **TEST-06**: Tests are designed to validate OR invalidate quantum memory hypothesis

### Codebase Cleanup

- [ ] **FIX-01**: Coactivation uses seeded RNG for deterministic bit transfer
- [ ] **FIX-02**: Text encoding prevents token collisions at small k values

### Architecture Clarity

- [ ] **ARCH-01**: Clear separation between primitive substrate and agentic layer
- [ ] **ARCH-02**: Primitive substrate is usable without LLM dependency
- [ ] **ARCH-03**: Agentic layer builds on substrate without violating its invariants

### Documentation

- [x] **DOC-01**: PROJECT.md captures full project context
- [x] **DOC-02**: REQUIREMENTS.md with traceable requirement IDs
- [x] **DOC-03**: ROADMAP.md with phases mapped to requirements
- [x] **DOC-04**: STATE.md for project memory across sessions

## v2 Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Persistence

- **PERS-01**: Patterns can be serialized to disk (crystallization)
- **PERS-02**: Substrate state can be restored across sessions
- **PERS-03**: Coherence values decay or preserve based on reheating strategy

### Scale Optimization

- **SCALE-01**: LSH indexing for O(1) approximate nearest neighbor
- **SCALE-02**: Partitioned storage for >100k patterns
- **SCALE-03**: Pattern pruning/forgetting mechanism

### LLM Integration

- **LLM-01**: Real LLM reranker integration with error handling
- **LLM-02**: Async retrieval for non-blocking operations
- **LLM-03**: Token budget management for reflection operations

## Out of Scope

| Feature | Reason |
|---------|--------|
| N-gram encoding fix | Currently disabled; not blocking coherence work |
| Pattern merging | Complex; defer to v2 after pruning |
| Multi-agent shared substrate | Architectural complexity; nail single-agent first |
| GPU kernel optimization | Performance nice-to-have; CPU sufficient for validation |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DOC-01 | Phase 0 | Complete |
| DOC-02 | Phase 0 | Complete |
| DOC-03 | Phase 0 | Complete |
| DOC-04 | Phase 0 | Complete |
| COHR-01 | Phase 1 | Pending |
| COHR-02 | Phase 1 | Pending |
| COHR-03 | Phase 1 | Pending |
| TEST-01 | Phase 1 | Pending |
| TEST-02 | Phase 1 | Pending |
| COHR-04 | Phase 2 | Pending |
| COHR-05 | Phase 2 | Pending |
| TEST-03 | Phase 2 | Pending |
| TEST-04 | Phase 2 | Pending |
| COHR-06 | Phase 3 | Pending |
| COHR-07 | Phase 3 | Pending |
| TEST-05 | Phase 3 | Pending |
| FIX-01 | Phase 4 | Pending |
| FIX-02 | Phase 4 | Pending |
| ARCH-01 | Phase 5 | Pending |
| ARCH-02 | Phase 5 | Pending |
| ARCH-03 | Phase 5 | Pending |
| TEST-06 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0

---
*Requirements defined: 2026-01-31*
*Last updated: 2026-01-31 after roadmap creation*
