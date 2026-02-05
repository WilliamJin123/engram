# Project Milestones: Engram

## v1.1 Harder Test Cases (Shipped: 2026-02-05)

**Delivered:** Stress-tested retrieval mechanisms with realistic memory noise, resolving the v1.0 ceiling effect and validating that interference-based retrieval significantly outperforms random baseline.

**Phases completed:** 6-8 (8 plans total)

**Key accomplishments:**

- Noise generation toolkit with semantic near-misses (40% overlap) and clutter patterns
- Reusable noisy_memory pytest fixture with configurable noise levels (NONE/LOW/MEDIUM/HIGH)
- Tunneling, interference, and coherence dynamics validated under realistic memory load
- Interference retrieval significantly beats random (p < 0.001, Mann-Whitney U)
- Linear degradation curves documented (target rank 1→6→9 as near-misses increase)
- MRR, Recall@K metrics with statistical testing infrastructure

**Stats:**

- 39 commits
- 3,703 lines added (15,148 total Python)
- 3 phases, 8 plans, 10 requirements
- 1 day (2026-02-04 to 2026-02-05)
- 105+ new tests (45 stress tests, 40 noise generation, 20+ metrics)

**Git range:** `feat(06-01)` → `docs(08)`

**Key findings:**

- Interference vs Random: Significantly better (p < 0.001)
- Interference vs Cosine: No significant difference at clean conditions (both achieve MRR 1.0)
- Degradation: Linear, not cliff-like (graceful performance reduction under noise)
- Tunneling: Success ~20% at MEDIUM noise, 0% at HIGH noise

**What's next:** v2.0 persistence layer, LSH indexing, or production readiness

---

## v1.0 Coherence (Shipped: 2026-02-04)

**Delivered:** Complete quantum-inspired coherence dynamics for memory substrate with decay, refresh, surprise re-coherence, tunneling, and criticality self-adjustment.

**Phases completed:** 0-5 (15 plans total)

**Key accomplishments:**

- Coherence field on patterns with exponential decay and access-based refresh
- Surprise-triggered re-coherence - decayed patterns resurrect through unexpected results
- Tunneling mechanism enabling creative retrieval from high-coherence patterns
- Criticality self-adjustment for order/chaos balance at system level
- Determinism fixes: seeded RNG in coactivation, collision-free text encoding
- SubstratePattern protocol establishing clean LLM-independent substrate layer

**Stats:**

- 92 commits
- 11,445 lines of Python
- 6 phases, 15 plans
- 4 days (2026-01-31 to 2026-02-04)
- 287 tests passing

**Git range:** `feat(01-01)` -> `feat(05-02)`

**Hypothesis validation:**
- Coherence dynamics: VALIDATED (10/10 tests pass)
- INTUITION.md behaviors: VALIDATED (9/9 tests pass)
- Interference vs baselines: INCONCLUSIVE (3 skipped due to ceiling effect)

**What's next:** v1.1 persistence layer, LSH indexing, or real LLM integration

---
