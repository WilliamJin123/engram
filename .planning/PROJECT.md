# Engram: Quantum-Inspired Memory Substrate

## What This Is

A quantum-inspired memory system for agentic AI that uses sparse distributed patterns with complex amplitudes (magnitude + phase), HRR binding, and coherence dynamics to enable associative memory with graceful decay, interference-based retrieval, and emergent organization. The substrate handles structure; LLMs handle semantics.

## Core Value

**Engineer the substrate. Let everything else emerge.** The system must implement the core quantum dynamics (coherence, interference, binding, phase) faithfully — everything built on top depends on these working correctly.

## Requirements

### Validated

- HRR binding accuracy: 100% role-filler recovery (20 entities tested)
- Phase sequence encoding: 100% ordering to n=100
- Interference-based retrieval: phase-aware similarity working
- Text encoding: deterministic hash-to-sparse pattern
- Coactivation learning: bidirectional bit transfer with obesity tracking
- Scale: <1ms binding, 50ms retrieval for 5k patterns, 11MB for 10k patterns

### Active

- [ ] Coherence field on patterns (0-1 scale)
- [ ] Time-based coherence decay toward classical stability
- [ ] Access-based coherence refresh
- [ ] Surprise/contradiction re-coherence mechanism
- [ ] Coherence-modulated interference (low coherence = weak contribution)
- [ ] Tunneling: high-coherence patterns activate weakly-related patterns
- [ ] Criticality tuning: system-wide parameter for order vs chaos balance
- [ ] Fix: deterministic bit transfer in coactivation (add RNG seed)
- [ ] Fix: text encoding collision at small k values
- [ ] Project documentation state (PROJECT.md, REQUIREMENTS.md, ROADMAP.md, STATE.md)

### Out of Scope

- Persistence layer — essential for production but not this milestone
- LSH indexing for O(1) retrieval — performance optimization, defer
- Real LLM integration testing — mock reranker sufficient for now
- Pattern pruning/forgetting — needed eventually, not blocking coherence work
- N-gram encoding (currently broken) — defer fix, leave disabled

## Context

**Theoretical foundation:**
- `dicussions/quantum_proposal.md` — authoritative spec for substrate mechanics
- `dicussions/quantum_agents.md` — agentic memory layer design
- `dicussions/quantum_tests.md` — validation approach and results

**Current state:**
- 51 tests passing, all core dynamics validated
- Coherence is described in proposal but NOT implemented in code
- Patterns have phase and magnitude but no coherence field
- Interference retrieval treats all patterns equally (no coherence weighting)

**Known issues (from CONCERNS.md):**
- `coactivation.py` uses `random.sample()` without seeding — non-deterministic
- Text encoding at small k can cause token collisions
- Patterns grow unbounded until max_bits; no backpressure signal

## Constraints

- **Theory fidelity**: Implementation must match quantum_proposal.md spec — these documents are the authoritative design
- **Test-first**: New dynamics must have validation tests like the existing substrate tests
- **Backwards compatible**: Existing tests must continue to pass; don't break what works
- **PyTorch dependency**: Continue using torch for tensor operations; no new major deps

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| HRR over tensor product | Scalable binding (O(n) vs O(n²)) | Good |
| Text stored, not embeddings | Avoids LLM bias, keeps substrate LLM-agnostic | Good |
| Original bits for binding, all bits for retrieval | Prevents transitive pollution while enabling learning | Good |
| Coherence per-pattern, not per-bit | Simpler; per-bit coherence adds complexity without clear benefit | Pending |

---
*Last updated: 2026-01-31 after project initialization*
