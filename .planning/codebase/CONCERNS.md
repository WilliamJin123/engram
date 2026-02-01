# Codebase Concerns

**Analysis Date:** 2026-01-31

## Tech Debt

**Long sequence phase resolution:**
- Issue: Phase encoding degrades for sequences beyond ~1000 items. At n=10000, phase resolution (2π/n) becomes ~0.0006 radians, risking confusion between adjacent items.
- Files: `src/agentic/text_encoder.py`, `tests/agentic/test_long_sequences.py`
- Impact: Conversation history, logs, or any sequence of 10000+ items will have unreliable order recovery from phase values. Position accuracy drops unmeasured at extreme scales.
- Fix approach: Implement logarithmic phase encoding for long sequences, or split sequences into multiple patterns with hierarchy markers.

**Random bit transfer without determinism:**
- Issue: `_transfer_bits()` in `src/agentic/coactivation.py` uses `random.sample()` without seeding, making learning non-deterministic and non-reproducible.
- Files: `src/agentic/coactivation.py:98` (`random.sample()` call)
- Impact: Same coactivation events produce different bit transfers across runs. Cannot reproduce memory states for debugging or testing. Systems trained twice will diverge.
- Fix approach: Add optional RNG seed parameter to `CoactivationConfig` and pass through to coactivation functions. Use `torch.Generator` for consistency with other randomization.

**Text encoding collision risk at small k:**
- Issue: When k < 10 (sparse), `_token_to_bits()` generates k/10 bits per token, sometimes rounding to 1. Multiple tokens can hash to same bits, losing uniqueness.
- Files: `src/agentic/text_encoder.py:174` (`bits_per_token = max(1, self.k // 10)`)
- Impact: Patterns from different texts produce identical bits when tokens collide. Retrieval becomes unreliable. Tests pass with k=50 but would fail with k=5.
- Fix approach: Ensure minimum bits per token (e.g., 3) or use multi-hash strategy to guarantee diversity.

**Missing overflow/saturation handling in memory growth:**
- Issue: Patterns can grow indefinitely through coactivation until hitting `max_bits` limit. No backpressure; patterns simply stop learning when full.
- Files: `src/agentic/coactivation.py:93-95` (availability check), `src/agentic/agent_memory.py:229-234` (obesity metric)
- Impact: Memory saturation causes learning to halt silently. No warning to user. Pattern obesity can reach 2.0+ (200% of original bits). System assumes memory available but doesn't report when at capacity.
- Fix approach: Add saturation metrics to `AgentMemory`. Track patterns at max capacity. Optionally evict low-acquisition patterns or implement LRU replacement.

**Interference method assumes unit magnitude consistently:**
- Issue: Interference retrieval in `src/quantum_substrate/interference.py` assumes all magnitudes are unit (1.0). `create_related_pattern()` also hardcodes magnitude 1.0. Binding operations don't preserve magnitude bounds.
- Files: `src/quantum_substrate/interference.py:190`, `src/quantum_substrate/patterns.py:62-63`
- Impact: If magnitudes vary significantly, interference formula becomes non-normalized. Similarity scores can exceed reasonable bounds. Phase information assumed valid but undocumented for non-unit magnitudes.
- Fix approach: Normalize magnitudes explicitly. Document magnitude constraints. Add validation in pattern creation.

## Known Bugs

**Unreachable n-gram code path:**
- Symptoms: EncodedPattern includes phase data for n-grams but n-gram encoding disabled by default. Using ngrams=True causes phase calculation issues.
- Files: `src/agentic/text_encoder.py:118-125` (ngram addition), `src/agentic/text_encoder.py:135` (phase calculation assumes single phase per bit)
- Trigger: Set `EncodingConfig(use_ngrams=True)` and encode text
- Workaround: Leave ngrams disabled. Current implementation assigns phase 0.0 to all ngram bits, breaking phase uniqueness.

**Metadata pollution in pattern store:**
- Symptoms: Pattern metadata gets modified after storage (id added). Calling `store()` twice with same metadata object mutates it.
- Files: `src/agentic/memory_store.py:64` (`pattern.metadata["id"] = pattern_id`)
- Trigger: Store a pattern, then check if metadata dict was mutated (it was)
- Workaround: Always pass new metadata dicts; don't reuse instances.

**Obesity ratio can exceed theoretical bounds:**
- Symptoms: `obesity` property returns values > 2.0 when `max_bits=200` and original_bits < 100
- Files: `src/agentic/evolving_pattern.py:106` (obesity calculation)
- Trigger: Store text with k=30, then coactivate heavily with max_bits=200
- Workaround: Document that obesity > 1.0 means learning occurred; > 2.0 is permitted but rare.

## Security Considerations

**Text encoding depends on hash functions:**
- Risk: SHA256 is deterministic but if cryptographic properties assumed, could be misused. No validation of hash output range.
- Files: `src/agentic/text_encoder.py:173, 178-179, 221, 225-226, 231`
- Current mitigation: Hashing used only for deterministic bit allocation, not security.
- Recommendations: Document that hash is for reproducibility, not cryptographic security. Consider faster non-crypto hash if performance required.

**LLM reranker accepts arbitrary Python code via BaseLLMReranker:**
- Risk: `set_reranker()` accepts any object implementing `BaseLLMReranker` interface. No validation of reranker behavior. Could inject adversarial ranking.
- Files: `src/agentic/agent_memory.py:197-203`, `src/agentic/llm_interface.py`
- Current mitigation: Reranker interface is intentionally flexible for custom LLM integration.
- Recommendations: Document security implications. Consider reranker signature validation if used in multi-user contexts.

## Performance Bottlenecks

**Retrieval is O(n) with full pattern iteration:**
- Problem: Both Jaccard and interference retrieval iterate all stored patterns. No indexing.
- Files: `src/agentic/memory_store.py:98-112` (Jaccard), `src/agentic/memory_store.py:127-150` (interference)
- Cause: Patterns stored in dict with no spatial index. At 10k patterns, every query scans all.
- Improvement path: Implement LSH (locality-sensitive hashing) for Jaccard, or bin patterns by bit overlap ranges. Accept tradeoff of approximate nearest neighbor.

**Dense tensor conversion on every retrieval:**
- Problem: Every similarity query calls `to_dense()` or `to_dense_evolved()`, allocating full 1024+ dimensional tensors.
- Files: `src/agentic/memory_store.py:122-124, 129-131` (dense conversion), `src/agentic/evolving_pattern.py:74-99` (dense methods)
- Cause: HRR binding requires dense tensors, but retrieval uses sparse patterns for Jaccard. Unnecessary materialization.
- Improvement path: Compute Jaccard entirely on sparse bit sets (already done). Cache dense tensors if reused. Use sparse tensor libraries for interference method.

**Coactivation does pairwise bit transfer O(k²) between all pattern pairs:**
- Problem: With 100 patterns retrieved and k=50, coactivate() does ~5000 random.sample() calls.
- Files: `src/agentic/coactivation.py:64-68` (nested loop)
- Cause: All-pairs bit transfer is correct but expensive for large top_k.
- Improvement path: Limit pairwise updates to top-k nearest neighbors only, or use probabilistic sampling.

**TextEncoder creates new encoder instance for every pattern:**
- Problem: `from_text()` creates new TextEncoder(dim, k) each time. Hash computations repeated.
- Files: `src/agentic/evolving_pattern.py:68-71` (TextEncoder instantiation in class method)
- Cause: No encoder caching/singleton. Each pattern encodes independently.
- Improvement path: Cache encoder instance in MemoryStore or AgentMemory. Reuse across patterns.

## Fragile Areas

**Phase extraction from empty bit set:**
- Files: `tests/agentic/test_long_sequences.py:34` (list(patterns[i].phases.values())[0])
- Why fragile: Assumes at least one bit in phases dict. Empty patterns or patterns with no phase info cause IndexError.
- Safe modification: Check bit count before accessing phases. Add guards in test and phase recovery code.
- Test coverage: test_long_sequences assumes valid patterns. No tests for empty or sparse patterns (k=1).

**EvolvingPattern.to_dense() uses original_bits only for binding:**
- Files: `src/agentic/evolving_pattern.py:74-85` (to_dense), `src/agentic/binding.py` (bind operations)
- Why fragile: Design choice to use original_bits for binding integrity is not validated. If client code calls to_dense() expecting all bits, receives wrong result. Naming ambiguous.
- Safe modification: Add docstring clarification. Create explicit `to_dense_for_binding()` method. Add assertion in bind_hrr() to verify sparsity.
- Test coverage: Binding tests don't verify that acquired bits are excluded.

**Circular mean computation undefined for empty phase list:**
- Files: `src/agentic/text_encoder.py:190-203` (_circular_mean)
- Why fragile: Returns 0.0 for empty list, but doesn't handle all-ambiguous phases (sin_sum=0, cos_sum=0).
- Safe modification: Add test for antipodal phase distributions. Throw explicit error if result undefined.
- Test coverage: phase encoding tests don't cover edge cases like empty tokens or single-bit patterns.

**MockLLMReranker hardcoded keyword matching:**
- Files: `src/agentic/llm_interface.py:47-86` (MockLLMReranker)
- Why fragile: Simple keyword overlap used for testing. Query boost (lines 71-72) is hardcoded constant 10. If test expects semantic ranking, fails silently.
- Safe modification: Document that mock is intentionally naive. Consider adding configurable boost factors or example semantic ranker subclass.
- Test coverage: test_agent_memory.py uses mock in production tests, risking false confidence.

**Max bits budget enforced late in coactivation:**
- Files: `src/agentic/coactivation.py:93-96` (availability check after strength calculation)
- Why fragile: Patterns near capacity don't learn at all. Asymmetric learning: full pattern stops accepting bits but empty pattern keeps learning.
- Safe modification: Implement proportional learning strength based on remaining capacity. Or prioritize bidirectional transfers.
- Test coverage: test_agent_memory.py checks max_bits enforcement but not learning rate degradation.

## Scaling Limits

**Scalability to pattern counts:**
- Current capacity: ~100 patterns retrievable in <1s on commodity hardware
- Limit: O(n) retrieval becomes bottleneck above 10k patterns. Dense tensor operations (binding) use single-threaded PyTorch.
- Scaling path: Implement partitioned storage (sharding), approximate nearest neighbor (HNSW), or distributed pattern store.

**Phase encoding sequence limits:**
- Current capacity: ~1000-item sequences with >95% position recovery from phase
- Limit: 10000+ items show significant position confusion. Phase resolution becomes 0.0006 radians.
- Scaling path: Hierarchical phase encoding (coarse + fine), or separate sequence index decoupled from phase.

**Memory growth under learning:**
- Current capacity: Patterns grow to 2x-3x original bits under sustained coactivation
- Limit: max_bits enforced hard, causing learning to stop. No adaptive growth or eviction.
- Scaling path: Implement pattern merging for similar patterns, or LRU replacement for low-activity patterns.

**Dimensionality explosion:**
- Current capacity: dim=1024 is reasonable for k=50 sparsity
- Limit: Interference method requires O(dim) FFT operations for binding. At dim=100k, binding becomes slow.
- Scaling path: Use structured sparsity, random projections, or kernel methods instead of full FFT.

## Dependencies at Risk

**PyTorch as only tensor library:**
- Risk: Hard dependency on PyTorch 2.0+. FFT-based HRR binding relies on torch.fft. No fallback to NumPy or JAX.
- Impact: Windows/ARM deployment requires PyTorch binary availability. GPU memory required even for CPU-only systems (torch loads it).
- Migration plan: Wrap tensor operations in abstraction layer. Support NumPy backend for non-HRR patterns. Consider scipy.fft for binding.

**No async/IO layer:**
- Risk: All operations are blocking. MemoryStore retrieval can't be parallelized or distributed.
- Impact: Multi-user systems would serialize access. Reranking with real LLM API calls would hang entire system.
- Migration plan: Add async retrieval interface. Use asyncio or ThreadPoolExecutor for LLM calls. Document blocking nature.

## Missing Critical Features

**No persistence layer:**
- Problem: All patterns stored in memory. No checkpoint/restore. Loss on process exit.
- Blocks: Long-running agents, distributed systems, production deployments.
- Priority: High - essential for real applications.

**No pruning or forgetting mechanism:**
- Problem: Patterns never removed. Only grow via coactivation. Memory use monotonically increases.
- Blocks: Long-horizon agents, constantly learning scenarios, memory-constrained devices.
- Priority: High - required for sustainable learning.

**No pattern clustering or semantic organization:**
- Problem: Flat pattern store with linear search. No way to navigate or organize learned associations.
- Blocks: Interpretability, user control over memory, debugging learned structures.
- Priority: Medium - affects usability but not core functionality.

**No explicit similarity metric API:**
- Problem: Users must call internal _retrieve_*() methods or recall(). No public similarity(pattern1, pattern2) function.
- Blocks: Custom ranking, debugging pattern relationships, building applications on top.
- Priority: Medium - needed for flexibility.

## Test Coverage Gaps

**Untested edge cases in text encoding:**
- What's not tested: Empty text, single-word text, very long text (>10k tokens), special characters, unicode
- Files: `src/agentic/text_encoder.py`, `tests/agentic/test_text_encoder.py`
- Risk: Edge cases like empty text return empty pattern (dim=1024, bits={}) but code assumes bits exist.
- Priority: High - encoding is critical path.

**Untested phase recovery accuracy:**
- What's not tested: Can phases be reliably inverted to recover sequence order at scale?
- Files: `src/agentic/text_encoder.py:184-188` (phase encoding), `tests/agentic/test_long_sequences.py:28-57`
- Risk: test_long_sequences only validates explicit sequential phase assignments, not real encoded text phases.
- Priority: High - phase design depends on this.

**Untested LLM integration end-to-end:**
- What's not tested: Real LLM API calls, error handling for LLM failures, timeout behavior, token limits
- Files: `src/agentic/llm_interface.py`, `src/agentic/agent_memory.py:156-162`
- Risk: MockLLMReranker passes tests but real reranker may crash or timeout, blocking system.
- Priority: High - reranking is stated as critical feature.

**Untested catastrophic interference at scale:**
- What's not tested: Adding 10000+ patterns to store and verifying recall degrades gracefully
- Files: `src/agentic/memory_store.py`, `tests/quantum_substrate/scale/test_catastrophic_interference.py`
- Risk: Scaling claims (test assertions >= 80% recall) not validated above 100 patterns.
- Priority: Medium - performance claim not proven.

**Untested metadata handling edge cases:**
- What's not tested: Metadata with circular references, very large metadata dicts, metadata mutation after storage
- Files: `src/agentic/memory_store.py:64`, `src/agentic/evolving_pattern.py:42`
- Risk: No deep copy of metadata. User mutations after store() affect stored pattern.
- Priority: Medium - usability issue but not algorithmic.

---

*Concerns audit: 2026-01-31*
