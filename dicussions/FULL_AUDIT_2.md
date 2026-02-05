# Engram v1 (quantum branch) - Full Audit

**Audit Date:** 2026-02-04
**Auditor:** Claude Opus 4.5
**Branch:** quantum
**Commit:** 53da50c

---

## Executive Summary

Engram v1 on the quantum branch represents a **successful proof-of-concept** for quantum-inspired memory substrate for agentic AI. The core philosophy "Engineer the substrate. Let everything else emerge" was largely realized at the substrate level: the primitives (sparse patterns with complex amplitudes, HRR binding, phase encoding, interference retrieval, coherence dynamics) all work and exceed original expectations. The test suite shows **327 passed, 3 skipped** across 330 collected tests.

**What Works Excellently:**
- HRR binding: 100% accuracy up to 30+ roles (far exceeding expected 4-7 limit)
- Phase sequence encoding: 100% accuracy to n=100, >95% to n=500
- Text encoding: Deterministic, lexical overlap creates bit overlap, word order affects phase
- Coherence dynamics: Decay/refresh/crystallization all match spec formulas
- Coactivation learning: Original bits transfer, prevents pollution, deterministic with seed

**What Remains Unvalidated:**
- Emergent properties (prediction, hierarchy, decision-making) - mechanisms exist but no tests prove emergence
- Interference vs baselines: Statistical tests inconclusive (3 skipped due to near-ceiling precision)
- Real LLM integration: Only mock reranker tested
- Multi-agent unified substrate: Described in spec, not implemented

**Verdict for v2:** The core substrate (`patterns.py`, `binding.py`, `interference.py`, `coherence.py`) should be **kept with minimal changes**. The agentic layer works but needs architectural cleanup to separate LLM-independent operations from LLM-dependent ones. Most importantly, v2 needs harder test scenarios to properly validate or invalidate the quantum memory hypothesis - the current tests are often too easy (ceiling effects).

---

## Part 1: Intuition Validation

### 1.1 Core Philosophy Intuitions

The foundational belief was: **"Engineer the substrate. Let everything else emerge."**

| Intuition | Source | Status | Evidence |
|-----------|--------|--------|----------|
| Sparse distributed patterns can represent arbitrary concepts | quantum_proposal.md | **VALIDATED** | `test_sparse_binding.py` - 100% accuracy at k=20,50,100; 6.8x compression ratio |
| Phase relationships can encode temporal/sequential information | quantum_proposal.md | **VALIDATED** | `test_phase_encoding.py`, `test_long_sequences.py` - 100% to n=100, >95% to n=500 |
| Interference-based retrieval provides signal that Jaccard cannot | quantum_proposal.md | **LIMBO** | `test_interference_beats_baselines.py` - 3/4 tests skipped (ceiling effects), 1 passed. Signal exists but statistical significance not demonstrated |
| HRR binding solves the binding problem (who did what to whom) | quantum_proposal.md | **VALIDATED** | `test_hrr_basics.py` - 100% accuracy up to 20 entities, 3+ roles; dog-bit-mailman disambiguation works |
| Coherence can model attention/salience with quantum-like dynamics | quantum_proposal.md | **VALIDATED** | `test_coherence_dynamics.py`, `test_decay.py` - Decay formula matches spec, refresh works, 6-item working memory |
| Entanglement can bind roles to fillers in composable structures | quantum_proposal.md | **LIMBO** | HRR binding works as entanglement proxy, but true entanglement semantics (joint states that cannot be described independently) not implemented |

### 1.2 Behavioral Intuitions (from INTUITION.md)

| Intuition | Description | Status | Evidence |
|-----------|-------------|--------|----------|
| Abstraction | Repeated exposure to similar things should generalize to shared concept | **VALIDATED** | `test_intuition_behaviors.py::TestGeneralization::test_generalization_via_shared_bits` - coactivation creates shared bits, query retrieves all related patterns |
| Inheritance | Shared attributes stored efficiently, not duplicated per instance | **VALIDATED** | `test_intuition_behaviors.py::TestInheritance::test_inheritance_via_coactivation` - coactivation increases bit overlap between similar patterns |
| Exceptions | Specific instances can override inherited attributes | **VALIDATED** | `test_intuition_behaviors.py::TestExceptions::test_exception_retrieval` - "penguins cannot fly" retrieved when querying about penguins flying |
| Certainty plasticity | Crystallized (low coherence) nodes hard to change; malleable (high coherence) easy | **VALIDATED** | `test_intuition_behaviors.py::TestCertaintyPlasticity::test_plasticity_based_on_coherence` - malleable delta > crystallized delta; `test_advanced_dynamics.py::TestMalleabilitySemantics` |
| Conditionals | "If X then Y" represented as connected patterns | **VALIDATED** | `test_intuition_behaviors.py::TestConditionals::test_conditional_chain_retrieval` - connection map enables chain retrieval |
| Meta-relationships | Analogies and reasoning about relationships themselves | **VALIDATED** | `test_intuition_behaviors.py::TestMetaRelationships::test_analogy_pattern_storage` - "cat is to kitten as dog is to puppy" pattern retrievable |
| Provenance | Tracking where knowledge was learned (when relevant) | **VALIDATED** | `test_intuition_behaviors.py::TestProvenance::test_provenance_metadata_preserved` - metadata dict fully preserved |
| History | How understandings change over time and why | **VALIDATED** | `test_intuition_behaviors.py::TestHistory::test_history_tracking_via_modification_tick` - last_modified_tick, access_count_since_modification work |

### 1.3 Emergent Properties (predicted but not yet observed)

| Property | Mechanism | Status | Notes |
|----------|-----------|--------|-------|
| Prediction | Temporal binding; A pre-activates B | **NOT OBSERVED** | Phase encoding exists but predictive pre-activation not tested |
| Hierarchy | Convergence creates levels (DOG->MAMMAL->ANIMAL) | **NOT OBSERVED** | Coactivation creates similarity; no hierarchy emergence tests |
| Decision | Superposition + interference + collapse | **NOT OBSERVED** | Measurement/collapse semantics described in spec but not implemented |
| Working memory limit (~4-7 items) | Coherence competition | **PARTIALLY** | `test_decay.py` shows 6-item coherence budget; implemented via decay not competition |
| Compositionality | Entanglement enables role binding | **VALIDATED** | HRR binding enables compositional structures; giver/receiver/item recovered |

---

## Part 2: What Worked

### 2.1 Validated Mechanisms

#### HRR Binding (Holographic Reduced Representations)
- **What it does:** Binds entities to roles via circular convolution in frequency domain, enabling recovery of "who did what to whom"
- **Evidence:** `test_hrr_basics.py` (8 tests), `test_sparse_binding.py` (7 tests), `test_superposition_limits.py` (4 tests) - All pass
- **Confidence level:** HIGH
- **Carry forward to v2?** YES - Keep as-is
- **Notes:** Works identically with sparse and dense patterns. 30+ role superposition exceeds expectations by 5x. Implementation is clean: `binding.py` is only 74 lines.

#### Phase Sequence Encoding
- **What it does:** Encodes temporal order in phase values (0 to 2π), enabling sequence recovery
- **Evidence:** `test_phase_encoding.py` (8 tests), `test_long_sequences.py` (13 tests) - All pass
- **Confidence level:** HIGH
- **Carry forward to v2?** YES - Keep as-is
- **Notes:** Linear mapping: phase = 2πi/n for position i in sequence of length n. Resolution degrades at n>1000 (0.006 rad) but still usable.

#### Interference-Based Retrieval
- **What it does:** Retrieves patterns considering phase alignment, not just bit overlap
- **Evidence:** `test_basics.py` (8 tests), `test_phase_interference.py` (6 tests), `test_retrieval.py` (4 tests) - All pass
- **Confidence level:** HIGH
- **Carry forward to v2?** YES - Keep as-is
- **Notes:** Same-phase constructive interference (score 1.0), opposite-phase destructive (score 0.0). Statistical advantage over Jaccard inconclusive but signal exists.

#### Coherence Dynamics
- **What it does:** Models pattern "quantumness" - high coherence participates in interference, low becomes classical/stable
- **Evidence:** `test_coherence_primitives.py` (16 tests), `test_decay.py` (15 tests), `test_coherence_effects.py` (20 tests), `test_advanced_dynamics.py` (10 tests) - All pass
- **Confidence level:** HIGH
- **Carry forward to v2?** YES with clarification
- **Notes:** Formula `coherence *= exp(-rate * dt / embeddedness)` validated. Malleability semantics finalized: coherence = how easy to change, not accessibility.

#### Text Encoding
- **What it does:** Deterministic hash-based encoding of text to sparse patterns with phase from word position
- **Evidence:** `test_text_encoder.py` (15 tests), `test_realistic_encoding.py` (9 tests) - All pass
- **Confidence level:** HIGH
- **Carry forward to v2?** YES - Keep as-is
- **Notes:** SHA256 hashing, k>=10 enforced (collision fix FIX-02 completed), word order affects phase but not bit identity.

#### Coactivation Learning
- **What it does:** Patterns retrieved together share bits over time, creating emergent similarity
- **Evidence:** `test_coactivation.py` (8 tests) - All pass
- **Confidence level:** HIGH
- **Carry forward to v2?** YES - Keep as-is
- **Notes:** Only original bits transfer (prevents pollution). Seeded RNG for determinism (FIX-01 completed). Obesity decay works.

### 2.2 Successful Architectural Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| HRR over tensor product | O(n) vs O(n²) binding | GOOD - Scales to 30+ roles without degradation |
| Text stored, not embeddings | Avoids LLM bias, keeps substrate LLM-agnostic | GOOD - Clean separation achieved |
| Original bits for binding, all bits for retrieval | Prevents transitive pollution while enabling learning | GOOD - Both work correctly, tested in `test_coactivation.py` |
| Coherence per-pattern, not per-bit | Simpler implementation | ADEQUATE - Works but limits attention granularity |
| Tick-based time vs wall-clock | Reproducible dynamics | GOOD - All coherence tests are deterministic |
| Sparse representation | Memory efficiency | GOOD - 6.8x compression at dim=1024, k=50 |

### 2.3 Test Suite Health

| Test File | Tests | Pass Rate | Coverage Quality |
|-----------|-------|-----------|------------------|
| test_agent_memory.py | 20 | 100% | GOOD |
| test_coactivation.py | 8 | 100% | GOOD |
| test_evaluation.py | 16 | 100% | GOOD |
| test_evolving_pattern.py | 8 | 100% | GOOD |
| test_llm_interface.py | 4 | 100% | ADEQUATE (mock only) |
| test_long_sequences.py | 13 | 100% | GOOD |
| test_memory_store.py | 6 | 100% | GOOD |
| test_realistic_encoding.py | 9 | 100% | GOOD |
| test_text_encoder.py | 15 | 100% | GOOD |
| test_hrr_basics.py | 8 | 100% | GOOD |
| test_sparse_binding.py | 7 | 100% | GOOD |
| test_superposition_limits.py | 4 | 100% | GOOD |
| test_coherence_primitives.py | 16 | 100% | GOOD |
| test_decay.py | 15 | 100% | GOOD |
| test_coherence_effects.py | 20 | 100% | GOOD |
| test_advanced_dynamics.py | 10 | 100% | GOOD |
| test_criticality.py | 18 | 100% | GOOD |
| test_tunneling.py | 15 | 100% | GOOD |
| test_intuition_behaviors.py | 9 | 100% | GOOD |
| test_coherence_dynamics.py | 10 | 100% | GOOD |
| test_interference_beats_baselines.py | 4 | 25% (3 skipped) | NEEDS WORK |
| test_protocol_compliance.py | 12 | 100% | GOOD |
| **TOTAL** | **330** | **99%** | **GOOD** |

---

## Part 3: What Kind of Worked

### 3.1 Partial Successes

#### Interference vs Baselines Statistical Validation
- **What works:** Phase information does distinguish same-phase from opposite-phase patterns; interference beats random significantly
- **What doesn't:** 3 of 4 statistical tests skipped due to ceiling effects (both methods achieve >90% precision)
- **Limiting factors:** Test scenarios too easy - relevant patterns have both high overlap AND similar phases, so Jaccard alone finds them
- **What would make it fully work:** Harder scenarios where overlap is equal but phase differs; more confounding distractors

#### Tunneling for Creative Retrieval
- **What works:** High-coherence patterns can activate weakly-related connected patterns via `attempt_tunneling()`
- **What doesn't:** Connection map must be manually populated; no automatic connection discovery
- **Limiting factors:** Requires pre-existing connections to tunnel through; `create_connection()` must be called explicitly
- **What would make it fully work:** Coactivation should automatically add to `connection_map`, not just increment `connection_count`

#### Criticality Self-Adjustment
- **What works:** System tracks retrieval quality and surprise frequency; adjusts order/chaos parameter
- **What doesn't:** No validation that adjustment actually improves system performance
- **Limiting factors:** Feedback signals are proxies; no end-to-end evaluation of self-adjustment benefit
- **What would make it fully work:** A/B test comparing self-adjusting vs fixed criticality over extended use

#### Surprise Detection and Re-coherence
- **What works:** Detects mismatch between expected (coherence-weighted) and actual top result; triggers re-coherence proportional to surprise magnitude
- **What doesn't:** Surprise is purely bit-based (Hamming distance), not semantic
- **Limiting factors:** What's "surprising" semantically requires LLM judgment
- **What would make it fully work:** Integration where LLM signals semantic surprise back to substrate

### 3.2 Theoretical Soundness vs Practical Implementation

| Concept | Theory (from proposal) | Implementation | Gap Analysis |
|---------|------------------------|----------------|--------------|
| Coherence | "High = interference/tunneling, Low = classical" | Phase 3 finalized: coherence = malleability, not accessibility | Semantics evolved correctly; documented in CONTEXT.md |
| Entanglement | "Joint quantum states that cannot be described independently" | HRR binding creates bound patterns that can be unbound | HRR is structural proxy, not true entanglement - acceptable |
| Collapse/Measurement | "Commitment to interpretation with external consequences" | Not implemented | Spec philosophical; unclear how to operationalize |
| Self-Referential Field | "Pattern represents itself observing itself" | Not implemented | Philosophical concept; not needed for memory substrate |
| Embeddedness | "Retrieval weighted by connections" | Phase 3: connectivity affects decay rate, not retrieval ranking | Design changed - embeddedness slows decay, pure similarity retrieves |

### 3.3 Performance Characteristics

| Operation | Target | Achieved | Acceptable? |
|-----------|--------|----------|-------------|
| Binding (single, dim=1024) | <1ms | 0.049ms | YES |
| Binding (single, dim=16384) | <1ms | 0.267ms | YES |
| Retrieval (5k patterns, Jaccard) | <100ms | 32.1ms | YES |
| Retrieval (5k patterns, Interference) | <100ms | 51.5ms | YES |
| Memory (10k patterns) | <50MB | 11.4MB | YES |
| Phase encoding (100 items) | 100% | 100% | YES |
| Phase encoding (1000 items) | >90% | >95% | YES |
| Phase encoding (10000 items) | Degraded | Works but tight | MARGINAL |

---

## Part 4: What Didn't Work

### 4.1 Failed or Abandoned Approaches

#### N-gram Encoding
- **What we tried:** Character n-gram encoding for sub-word features via `EncodingConfig(use_ngrams=True)`
- **Why it failed:** Assigns phase 0.0 to all n-gram bits (line 127 in text_encoder.py), breaking phase uniqueness
- **Lessons learned:** Phase must be meaningful for all bits, not just word-derived ones
- **Should v2 revisit?** MAYBE - If proper phase assignment (based on position in text) implemented

#### Statistical Interference Advantage
- **What we tried:** Prove interference retrieval significantly outperforms Jaccard via Welch's t-test
- **Why it failed:** Test scenarios too easy; both methods achieve >90% precision; ceiling effects
- **Lessons learned:** Need harder test cases with confounding distractors
- **Should v2 revisit?** YES - Critical for validating quantum hypothesis

### 4.2 Known Bugs (at time of audit)

| Bug | Severity | Location | Root Cause | Workaround |
|-----|----------|----------|------------|------------|
| Metadata mutation on store | LOW | `memory_store.py:104` | Pattern.metadata["id"] added in-place | Pass new dict each time |
| N-gram phase = 0.0 | MEDIUM | `text_encoder.py:127` | N-gram bits hardcoded to phase 0.0 | Leave use_ngrams=False |
| Obesity can exceed 2.0 | LOW | `evolving_pattern.py:119` | max_bits > 2*original_bits allowed | Document as expected |
| TextEncoder instantiated per pattern | LOW | `evolving_pattern.py:82-84` | `from_text()` creates new encoder | Works but inefficient |

### 4.3 Tech Debt Accumulated

| Debt Item | Impact | Effort to Fix | Priority for v2 |
|-----------|--------|---------------|-----------------|
| No persistence layer | Can't save/restore memory | HIGH | MUST |
| No pattern pruning/forgetting | Memory grows unbounded | MEDIUM | SHOULD |
| O(n) retrieval scan | Slow at scale beyond 10k | MEDIUM | SHOULD |
| Dense tensor conversion per retrieval | Memory waste | LOW | COULD |
| No async/IO layer | Blocks on LLM calls | MEDIUM | SHOULD |
| Connection map not auto-populated | Manual connection creation required | LOW | SHOULD |

---

## Part 5: Architecture Review

### 5.1 Layer Analysis

#### Quantum Substrate Layer (`src/quantum_substrate/`)
- **Responsibility clarity:** CLEAR - Primitives for patterns, binding, interference, coherence, surprise, tunneling, criticality
- **Cohesion:** HIGH - Each module has single responsibility
- **Coupling:** LOW - Only depends on torch; modules are independent
- **Issues:** `protocols.py` exists but `SubstratePattern` protocol could be more explicitly used
- **v2 recommendation:** Keep as-is; consider adding `__all__` exports

#### Agentic Layer (`src/agentic/`)
- **Responsibility clarity:** SOMEWHAT CLEAR - Memory operations, text encoding, coactivation, LLM interface
- **Cohesion:** MEDIUM - `memory_store.py` has grown large (715 lines) with mixed concerns
- **Coupling:** Coupled to substrate (good); optional coupling to LLM via BaseLLMReranker (good)
- **Issues:** `memory_store.py` contains coherence manager, surprise detector, tunneling, criticality - should be composed not embedded
- **v2 recommendation:** Split `memory_store.py` into `pattern_store.py` (storage) + `retrieval_engine.py` (retrieval + dynamics)

### 5.2 Module-by-Module Review

| Module | Purpose | Quality | Issues | v2 Action |
|--------|---------|---------|--------|-----------|
| `patterns.py` | Sparse complex patterns | GOOD | None | KEEP |
| `binding.py` | HRR bind/unbind | GOOD | None | KEEP |
| `interference.py` | Phase-aware retrieval | GOOD | None | KEEP |
| `coherence.py` | Decay/refresh dynamics | GOOD | None | KEEP |
| `tunneling.py` | Creative retrieval | GOOD | Needs auto-connection | PORT |
| `criticality.py` | Order/chaos balance | GOOD | Unvalidated benefit | KEEP |
| `surprise.py` | Mismatch detection | GOOD | Bit-based only | KEEP |
| `protocols.py` | SubstratePattern protocol | GOOD | None | KEEP |
| `text_encoder.py` | Text to pattern | GOOD | N-gram phase broken | KEEP, fix ngram |
| `evolving_pattern.py` | Learning-capable pattern | GOOD | None | KEEP |
| `coactivation.py` | Bit sharing learning | GOOD | None | KEEP |
| `memory_store.py` | Storage + retrieval | ADEQUATE | Too large (715 lines) | REFACTOR |
| `agent_memory.py` | High-level API | GOOD | Thin wrapper | KEEP |
| `llm_interface.py` | Reranker abstraction | ADEQUATE | Mock only tested | KEEP |
| `evaluation.py` | Metrics | GOOD | None | KEEP |

### 5.3 Data Flow Issues

1. **MemoryStore is too monolithic**: Contains CoherenceManager, SurpriseDetector, TunnelingConfig, CriticalityState, CreativeModeTracker - should be composed externally
2. **TextEncoder instantiation per pattern**: `EvolvingPattern.from_text()` creates new encoder each time (line 82-84)
3. **Connection map separate from patterns**: Patterns don't know their own connections; external dict required (`connection_map`)
4. **Coherence manager embedded in store**: Should be passed in, not owned, for better testability

### 5.4 API Surface Review

| API | Clarity | Usability | Breaking Changes Needed? |
|-----|---------|-----------|--------------------------|
| `AgentMemory.store()` | GOOD | Easy | No |
| `AgentMemory.recall()` | GOOD | Easy | No |
| `AgentMemory.learn()` | GOOD | Easy | No |
| `MemoryStore.retrieve()` | ADEQUATE | method param confusing | Simplify in v2 |
| `MemoryStore.retrieve_pure_similarity()` | ADEQUATE | Why separate from retrieve()? | Merge |
| `MemoryStore.retrieve_with_surprise()` | ADEQUATE | Complex params | Simplify |
| `MemoryStore.retrieve_with_tunneling()` | ADEQUATE | Complex params | Simplify |

---

## Part 6: Theoretical Alignment

### 6.1 Quantum Proposal Compliance

| Concept | Spec Summary | Implementation | Fidelity |
|---------|--------------|----------------|----------|
| Pillar 1: Sparse Distributed Representations | Few units active vs all | `ComplexSparsePattern`, `EvolvingPattern` | **FAITHFUL** |
| Pillar 2: Temporal/Phase Structure | Info in WHEN things activate | Phase in `text_encoder.py`, `patterns.py` | **FAITHFUL** |
| Pillar 3: Quantum Dynamics | Magnitude, phase, coherence | `coherence.py`, patterns have all three | **FAITHFUL** |
| Pillar 4: Entanglement | Joint quantum states | HRR binding (structural proxy) | **PARTIAL** |
| Self-Referential Field (Faggin) | Agent/observed/observer triad | Not implemented | **DIVERGENT** (acceptable) |
| Binding: Synchrony vs Entanglement | Association vs structure | Coactivation (synchrony) + HRR (structure) | **FAITHFUL** |
| Core Dynamics | Activation, interference, decoherence, tunneling | All implemented | **FAITHFUL** |
| Collapse/Measurement | Commitment to interpretation | Not implemented | **DIVERGENT** |
| Criticality | Edge of order/chaos | `criticality.py` self-adjustment | **FAITHFUL** |

### 6.2 Agentic Memory Proposal Compliance

| Concept | Spec Summary | Implementation | Fidelity |
|---------|--------------|----------------|----------|
| Store operation | Text + context -> entangled pattern | `AgentMemory.store()` with metadata | **FAITHFUL** |
| Recall operation | Query -> activation -> interference -> rerank | `AgentMemory.recall()` with optional LLM | **FAITHFUL** |
| Reflect operation | Summarize, reorganize, abstract | Not implemented | **DIVERGENT** |
| Division of labor | Substrate = structure, LLM = semantics | Clean interface, LLM optional | **FAITHFUL** |
| Text as primary representation | No embeddings, store actual text | `EvolvingPattern.text` stored | **FAITHFUL** |
| Encoding is simple; dynamics are rich | Hash to sparse, magic in interference | SHA256 encoding, rich dynamics | **FAITHFUL** |
| Multi-agent unified substrate | One mind, multiple foci | Not implemented | **DIVERGENT** |
| Measurement and collapse | Acting on memory = collapse | Not implemented | **DIVERGENT** |
| Crystallization for persistence | Controlled decoherence for storage | Coherence decay exists; no persistence | **PARTIAL** |

### 6.3 Spec Ambiguities Discovered

1. **Coherence: accessibility vs malleability**: Spec implied both. Implementation chose malleability only (coherence governs how easy to change, not how easy to retrieve). Documented in Phase 3 CONTEXT.md.

2. **Embeddedness weighting**: Spec said "weighted by embeddedness". Implementation changed to: embeddedness slows decay (well-connected patterns stay coherent longer) but doesn't directly weight retrieval scores.

3. **What triggers surprise**: Spec vague. Implementation: surprise = Hamming distance between expected (coherence-weighted prediction) and actual top result.

4. **How connections form**: Spec doesn't say. Implementation: manual `create_connection()` required; coactivation increments `connection_count` but doesn't add to `connection_map`.

5. **Criticality self-adjust frequency**: Spec doesn't specify. Implementation: every 10 operations (`_criticality_adjust_interval`).

---

## Part 7: Open Questions

### 7.1 Unresolved Design Questions

| Question | Context | Options Considered | Current State |
|----------|---------|-------------------|---------------|
| Should coherence affect retrieval ranking? | Phase 1-3 evolution | Yes (coherence-weighted) vs No (pure similarity) | No - pure similarity + coherence affects malleability only |
| Should connections auto-populate from coactivation? | Tunneling needs connections | Yes (coactivation creates them) vs No (manual) | No - connection_count increments but map not updated |
| Per-pattern vs per-bit coherence? | Spec unclear | Per-pattern simpler | Per-pattern |
| When should "reflect" trigger? | Spec describes operation | Time-based vs count-based vs LLM-judged | Not implemented |

### 7.2 Empirical Questions (need more testing)

| Question | How to Test | Priority |
|----------|-------------|----------|
| Does interference beat baselines on harder scenarios? | Create confounding distractors with same overlap but different phases | HIGH |
| Does coactivation actually improve retrieval over time? | A/B test with/without learning across sessions | HIGH |
| Does surprise re-coherence improve outcomes? | Measure retrieval quality with/without re-coherence | HIGH |
| Does deep hierarchy emerge from coactivation? | Long-term study with labeled concepts | MEDIUM |
| Sequence length limits before phase aliasing? | Stress test >10k items with position recovery | LOW |

### 7.3 Theoretical Questions (need more thought)

| Question | Why It Matters |
|----------|----------------|
| How should coherence interact with binding? | Currently independent - should bound patterns share coherence? |
| What makes a retrieval "surprising" semantically? | Current bit-based measure ignores meaning |
| How do we know if the substrate hypothesis is valid? | Need falsifiable tests that can actually invalidate |
| Does the system actually outperform RAG? | No benchmark comparison exists |

---

## Part 8: v2 Recommendations

### 8.1 Must Have (Critical for v2)

1. **Persistence layer** - Memory must survive process restart. Serialize patterns, coherence state, connection map to JSON/MessagePack.

2. **Harder hypothesis validation tests** - Current tests have ceiling effects. Need scenarios where:
   - Relevant patterns have same overlap but different phases (only interference can distinguish)
   - Distractors confuse bit-overlap methods
   - Statistical significance (p < 0.05) achievable

3. **Clarified API** - Merge `retrieve()`, `retrieve_pure_similarity()`, `retrieve_with_surprise()`, `retrieve_with_tunneling()` into single configurable method.

4. **Auto-connection from coactivation** - When patterns coactivate, add to `connection_map`, not just increment `connection_count`.

### 8.2 Should Have (Important but not blocking)

1. **Pattern pruning/forgetting** - Mechanism to remove low-value patterns; bound memory growth
2. **O(log n) retrieval** - LSH or approximate nearest neighbor for >10k patterns
3. **Real LLM integration tests** - Not just mock reranker
4. **Reflect operation** - LLM-driven summarization and reorganization as spec describes
5. **Split memory_store.py** - Separate storage from retrieval engine

### 8.3 Could Have (Nice to have)

1. **Multi-agent unified substrate** - Multiple foci on one memory
2. **N-gram encoding fix** - Proper phase assignment for sub-word features
3. **GPU acceleration** - For very large pattern counts
4. **Async LLM calls** - Non-blocking reranking
5. **Measurement/collapse semantics** - If philosophical model proves useful

### 8.4 Won't Have (Explicitly descoped)

1. **True quantum entanglement** - HRR binding is sufficient proxy
2. **Self-referential field implementation** - Too philosophical, unclear operationalization
3. **Per-bit coherence** - Complexity not justified
4. **Complex phase (multiple dimensions)** - Single phase dimension working well

### 8.5 Architectural Pivot Candidates

1. **Compose, don't embed**: MemoryStore should receive CoherenceManager, SurpriseDetector, CriticalityState as constructor params, not create them internally.

2. **Event-sourced coherence**: Record events (access, coactivation, surprise) and compute coherence lazily. Better debuggability, reproducibility.

3. **Connection graph as first-class object**: Separate connection management from MemoryStore entirely.

### 8.6 Carry-Forward Code

| Component | Disposition | Rationale |
|-----------|-------------|-----------|
| `patterns.py` | **KEEP AS-IS** | Clean, well-tested, 113 lines |
| `binding.py` | **KEEP AS-IS** | Simple, correct, 74 lines |
| `interference.py` | **KEEP AS-IS** | Works well, 211 lines |
| `coherence.py` | **KEEP AS-IS** | Formula validated, 298 lines |
| `tunneling.py` | **PORT** | Add auto-connection discovery |
| `criticality.py` | **KEEP AS-IS** | Works, benefit unvalidated but harmless |
| `surprise.py` | **KEEP AS-IS** | Works, 194 lines |
| `protocols.py` | **KEEP AS-IS** | Clean protocol definition |
| `text_encoder.py` | **KEEP**, fix ngram | 239 lines, works well |
| `evolving_pattern.py` | **KEEP AS-IS** | Clean, 172 lines |
| `coactivation.py` | **KEEP AS-IS** | Clean, 144 lines |
| `memory_store.py` | **REFACTOR** | Too large (715 lines), split it |
| `agent_memory.py` | **KEEP AS-IS** | Good facade, 235 lines |
| `llm_interface.py` | **KEEP AS-IS** | Good abstraction |
| `evaluation.py` | **KEEP AS-IS** | Useful metrics |
| Test utilities | **KEEP** | Comprehensive |

---

## Part 9: Documentation State

### 9.1 Documentation Inventory

| Document | Location | Currency | Quality | v2 Action |
|----------|----------|----------|---------|-----------|
| quantum_proposal.md | dicussions/ | CURRENT | GOOD | KEEP - authoritative spec |
| quantum_agents.md | dicussions/ | CURRENT | GOOD | KEEP - agentic spec |
| INTUITION.md | dicussions/ | CURRENT | GOOD | KEEP - validation target |
| TEST_SUMMARY.md | dicussions/ | CURRENT | GOOD | UPDATE for v2 |
| quantum_tests.md | dicussions/ | CURRENT | GOOD | KEEP |
| PROJECT.md | .planning/ | OUTDATED | ADEQUATE | ARCHIVE, create new |
| ROADMAP.md | .planning/ | COMPLETE | GOOD | ARCHIVE |
| REQUIREMENTS.md | .planning/ | CURRENT | GOOD | UPDATE for v2 |
| STATE.md | .planning/ | OUTDATED | ADEQUATE | ARCHIVE |
| ARCHITECTURE.md | .planning/codebase/ | CURRENT | GOOD | UPDATE for v2 |
| CONCERNS.md | .planning/codebase/ | CURRENT | GOOD | Many addressed |
| STRUCTURE.md | .planning/codebase/ | CURRENT | ADEQUATE | UPDATE for v2 |
| STACK.md | .planning/codebase/ | CURRENT | ADEQUATE | KEEP |
| TESTING.md | .planning/codebase/ | CURRENT | ADEQUATE | UPDATE for v2 |
| CONVENTIONS.md | .planning/codebase/ | CURRENT | ADEQUATE | KEEP |
| INTEGRATIONS.md | .planning/codebase/ | CURRENT | ADEQUATE | UPDATE for v2 |

### 9.2 Missing Documentation

1. **API Reference** - No consolidated API doc beyond docstrings
2. **Tutorial/Getting Started** - No quick-start guide
3. **Decision Log** - Key decisions scattered across ROADMAP commits
4. **Benchmark Results** - Performance claims not in reproducible benchmark suite
5. **Glossary** - Terms like "coherence", "crystallization", "embeddedness" need definitions
6. **Comparison to Alternatives** - No document comparing to RAG/vector DB

---

## Appendices

### Appendix A: Test Results Summary

```
============================= test session starts =============================
platform win32 -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0
rootdir: C:\Users\jinwi\programming_files_NEW\engram
collected 330 items

tests/agentic/ - 91 tests passed
tests/hypothesis_validation/ - 22 tests (19 passed, 3 skipped)
tests/quantum_substrate/ - 217 tests passed

=========== 327 passed, 3 skipped, 3 warnings in 384.04s (0:06:24) ============
```

### Appendix B: File Tree

```
src/
├── agentic/
│   ├── __init__.py
│   ├── agent_memory.py       # High-level API (235 lines)
│   ├── coactivation.py       # Learning rule (144 lines)
│   ├── evaluation.py         # Metrics framework
│   ├── evolving_pattern.py   # Learning-capable pattern (172 lines)
│   ├── llm_interface.py      # Reranker abstraction
│   ├── memory_store.py       # Storage + retrieval (715 lines) - REFACTOR CANDIDATE
│   └── text_encoder.py       # Text to pattern (239 lines)
└── quantum_substrate/
    ├── __init__.py
    ├── binding.py            # HRR operations (74 lines)
    ├── coherence.py          # Decay/refresh (298 lines)
    ├── criticality.py        # Order/chaos (131 lines)
    ├── interference.py       # Phase-aware retrieval (211 lines)
    ├── patterns.py           # Sparse complex patterns (113 lines)
    ├── protocols.py          # SubstratePattern protocol
    ├── surprise.py           # Mismatch detection (194 lines)
    └── tunneling.py          # Creative retrieval (218 lines)
```

### Appendix C: Key Code Snippets

**HRR Binding (the core insight):**
```python
# binding.py:12-28
def bind_hrr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bind two patterns via circular convolution."""
    return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b))

def unbind_hrr(bound: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Unbind pattern b from a bound pattern."""
    return torch.fft.ifft(torch.fft.fft(bound) * torch.conj(torch.fft.fft(b)))
```

**Coherence Decay Formula:**
```python
# coherence.py:94-97
effective_rate = self.config.decay_rate * crystallization_multiplier / pattern.embeddedness
decayed = pattern.coherence * math.exp(-effective_rate * dt)
return max(self.config.floor, decayed)
```

**Coactivation Learning (original bits only):**
```python
# coactivation.py:110
transferable = source.original_bits - target.bits  # Only original bits transfer
```

### Appendix D: Key Metrics from TEST_SUMMARY.md

**Wave 1 Results:**
- HRR binding: 100% accuracy up to 20 entities, 3 roles
- Phase sequence encoding: 100% accuracy up to 100 items
- Interference vs Jaccard: Phase adds distinguishable signal
- Scale: <1ms binding, 50ms retrieval for 5k patterns

**Wave 2 Results:**
- Sparse patterns: No degradation from dense
- 30+ role superposition: Far exceeds expected 4-7 limit
- 50% semantic overlap tolerated
- 30% noise resilience
- No catastrophic interference
- 6-item working memory (matches Miller's law)

**Hypothesis Validation Results:**
- All 9 INTUITION.md behaviors: VALIDATED
- Coherence dynamics: All 10 tests pass
- Interference vs baselines: 1 passed, 3 skipped (ceiling effects)

### Appendix E: References

- `dicussions/quantum_proposal.md` - Authoritative spec for quantum substrate mechanics
- `dicussions/quantum_agents.md` - Agentic memory layer design
- `dicussions/INTUITION.md` - Original behavioral intuitions
- `dicussions/TEST_SUMMARY.md` - Comprehensive test results
- `dicussions/quantum_tests.md` - Test design and validation approach
- `.planning/PROJECT.md` - Project requirements and decisions
- `.planning/ROADMAP.md` - Milestone planning and progress
- `.planning/REQUIREMENTS.md` - Traceable requirements
- `.planning/codebase/ARCHITECTURE.md` - Architecture analysis
- `.planning/codebase/CONCERNS.md` - Known issues and tech debt

---

*Audit completed: 2026-02-04*
