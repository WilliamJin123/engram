# Engram: Quantum-Inspired Memory Substrate

## What This Is

A quantum-inspired memory system for agentic AI that uses sparse distributed patterns with complex amplitudes (magnitude + phase), HRR binding, and coherence dynamics to enable associative memory with graceful decay, interference-based retrieval, and emergent organization. The substrate handles structure; LLMs handle semantics.

## Core Value

**Engineer the substrate. Let everything else emerge.** The system must implement the core quantum dynamics (coherence, interference, binding, phase) faithfully - everything built on top depends on these working correctly.

## Requirements

### Validated

- HRR binding accuracy: 100% role-filler recovery (20 entities tested) - v1.0
- Phase sequence encoding: 100% ordering to n=100 - v1.0
- Interference-based retrieval: phase-aware similarity working - v1.0
- Text encoding: deterministic hash-to-sparse pattern - v1.0
- Coactivation learning: bidirectional bit transfer with obesity tracking - v1.0
- Scale: <1ms binding, 50ms retrieval for 5k patterns, 11MB for 10k patterns - v1.0
- Coherence field on patterns (0-1 scale) - v1.0
- Time-based coherence decay toward classical stability - v1.0
- Access-based coherence refresh - v1.0
- Surprise/contradiction re-coherence mechanism - v1.0
- Coherence-modulated interference (low coherence = weak contribution) - v1.0
- Tunneling: high-coherence patterns activate weakly-related patterns - v1.0
- Criticality tuning: system-wide parameter for order vs chaos balance - v1.0
- Fix: deterministic bit transfer in coactivation (add RNG seed) - v1.0
- Fix: text encoding collision at small k values - v1.0
- SubstratePattern protocol for LLM-independent substrate layer - v1.0

### Active

**Current Milestone: v1.1 Harder Test Cases**

**Goal:** Stress-test retrieval mechanisms with realistic memory noise to resolve the v1.0 ceiling effect and validate that interference-based retrieval outperforms baselines under challenging conditions.

**Target features:**
- Noise generators: semantic near-misses and random clutter patterns
- Harder INTUITION.md behavior tests (tunneling, interference, coherence dynamics)
- Differentiation metrics: interference vs cosine vs random baselines
- Graceful degradation curves showing performance vs noise level
- Reusable test fixtures for noisy memory scenarios

### Out of Scope

- Persistence layer - essential for production but deferred to v2
- LSH indexing for O(1) retrieval - performance optimization, defer
- Real LLM integration testing - mock reranker sufficient for now
- Pattern pruning/forgetting - needed eventually, not blocking
- N-gram encoding (currently broken) - defer fix, leave disabled
- Patterns grow unbounded until max_bits - no backpressure signal (v2)

## Context

**Shipped v1.0 Coherence milestone:** 2026-02-04
**Started v1.1 Harder Test Cases:** 2026-02-04

**Current state:**
- 287 tests passing (coherence, tunneling, criticality, hypothesis validation)
- 11,445 lines of Python
- Full coherence dynamics implemented and validated
- Hypothesis validation: coherence dynamics and INTUITION.md behaviors confirmed

**v1.0 Hypothesis validation summary:**
- Coherence dynamics: VALIDATED (10/10 tests pass)
- INTUITION.md behaviors: VALIDATED (9/9 tests pass)
- Interference vs baselines: INCONCLUSIVE (ceiling effect in synthetic tests)

**v1.1 addresses:** The ceiling effect — synthetic tests were too easy, making all retrieval methods appear equivalent. Harder tests with memory noise will differentiate interference-based retrieval from baselines.

**Theoretical foundation:**
- `dicussions/quantum_proposal.md` - authoritative spec for substrate mechanics
- `dicussions/quantum_agents.md` - agentic memory layer design
- `dicussions/quantum_tests.md` - validation approach and results

## Constraints

- **Theory fidelity**: Implementation must match quantum_proposal.md spec
- **Test-first**: New dynamics must have validation tests
- **Backwards compatible**: Existing tests must continue to pass
- **PyTorch dependency**: Continue using torch for tensor operations

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| HRR over tensor product | Scalable binding (O(n) vs O(n^2)) | Good |
| Text stored, not embeddings | Avoids LLM bias, keeps substrate LLM-agnostic | Good |
| Original bits for binding, all bits for retrieval | Prevents transitive pollution while enabling learning | Good |
| Coherence per-pattern, not per-bit | Simpler; per-bit adds complexity without clear benefit | Good |
| Decay rate 0.05 default | Half-life ~14 ticks balances forgetting and retention | Good |
| Coherence floor 0.01 | Patterns never fully decohere | Good |
| Stability score = access_count / 10 | 10 accesses = max stability for crystallization | Good |
| Tunneling requires connection AND low overlap | Ensures creative but meaningful connections | Good |
| SubstratePattern protocol | Clean separation enables LLM-independent substrate | Good |

---
*Last updated: 2026-02-04 after v1.1 milestone start*
