# Architecture

**Analysis Date:** 2026-01-31

## Pattern Overview

**Overall:** Layered substrate architecture with quantum-inspired primitives supporting agentic memory systems.

**Key Characteristics:**
- Two-layer composition: quantum substrate (low-level patterns, binding, interference) and agentic layer (memory, retrieval, learning)
- Sparse, complex-valued pattern representation with phase information
- HRR (Holographic Reduced Representations) for role-filler binding
- Coactivation-based associative learning for pattern evolution
- LLM-agnostic substrate with pluggable semantic reranking interface
- Clear separation of concerns: substrate handles structure, agentic layer handles memory operations

## Layers

**Quantum Substrate (`src/quantum_substrate/`):**
- Purpose: Provides low-level pattern representation and operations inspired by quantum mechanics
- Location: `src/quantum_substrate/`
- Contains: Complex sparse patterns, circular convolution binding, interference-based retrieval
- Depends on: torch (for complex tensor operations and FFT)
- Used by: Agentic layer for pattern creation and similarity computation

**Agentic Memory (`src/agentic/`):**
- Purpose: High-level memory API integrating storage, retrieval, learning, and semantic understanding
- Location: `src/agentic/`
- Contains: AgentMemory orchestrator, MemoryStore, coactivation learning, text encoding, LLM interface
- Depends on: Quantum substrate + torch
- Used by: Applications requiring persistent, learnable memory with semantic retrieval

## Data Flow

**Store Flow:**

1. Application calls `AgentMemory.store(text, metadata)`
2. Text is encoded via `TextEncoder.encode(text)` → `EncodedPattern`
3. Pattern converted to `EvolvingPattern` with original_bits frozen
4. Pattern stored in `MemoryStore.patterns` dict with auto-generated or provided ID
5. Memory ID returned to application

**Recall Flow:**

1. Application calls `AgentMemory.recall(query, top_k=10, use_reranking=False)`
2. Query text encoded to pattern (via TextEncoder → EvolvingPattern)
3. Initial retrieval via `MemoryStore.retrieve(method="jaccard"|"interference", use_evolved=True)`
   - Jaccard method: set intersection/union on active bits (ignores phase)
   - Interference method: phase-aware similarity using complex amplitude addition
4. If `use_reranking=True` and reranker is set:
   - Candidates passed to `BaseLLMReranker.rerank(query, candidates, top_k)`
   - LLM-based semantic judgment reorders candidates
5. Top-k results returned as `RetrievalResult` list (pattern_id, EvolvingPattern, score)

**Learn Flow:**

1. Application calls `AgentMemory.learn(results)` with retrieval results
2. Patterns extracted from results
3. `coactivate(patterns, strength, config)` processes all pairs:
   - For each pair (p1, p2): transfer bits in both directions (if bidirectional=True)
   - `_transfer_bits(source, target)`:
     - Selects random subset of source's ORIGINAL bits (not acquired bits)
     - Applies obesity_decay: large patterns learn slower
     - Respects max_bits budget
     - Adds bits to target.bits and target.acquired_bits
     - Copies phase information from source
4. Patterns evolve in place (bits increase, obesity metric tracks growth)

**State Management:**
- Patterns maintained in `MemoryStore.patterns` dict keyed by pattern_id
- Each `EvolvingPattern` independently tracks: dim, bits (current), original_bits (frozen), acquired_bits, phases, metadata, acquisition_count
- No central state: all changes to patterns occur through direct method calls
- Reranker is optional and injected at initialization or via `set_reranker()`

## Key Abstractions

**ComplexSparsePattern (Quantum Substrate):**
- Purpose: Represent information as sparse complex-valued vectors with magnitude and phase
- Examples: `src/quantum_substrate/patterns.py`
- Pattern: Immutable dataclass with sorted indices, magnitudes, phases; provides to_dense() and from_dense() for conversion

**EvolvingPattern (Agentic Layer):**
- Purpose: Represent a memory that learns through coactivation by acquiring new bits over time
- Examples: `src/agentic/evolving_pattern.py`
- Pattern: Dataclass tracking original_bits (from encoding) separately from current bits; distinguishes bits used for binding vs. retrieval; computes obesity (current/original ratio)

**MemoryStore:**
- Purpose: Provide persistent storage and similarity-based retrieval for evolved patterns
- Examples: `src/agentic/memory_store.py`
- Pattern: Dictionary of patterns keyed by ID; supports two retrieval methods (Jaccard or interference); returns RetrievalResult tuples with score

**AgentMemory:**
- Purpose: Unified API orchestrating store, recall, and learn operations
- Examples: `src/agentic/agent_memory.py`
- Pattern: Facade over MemoryStore, coactivation, and LLM reranker; configurable via AgentMemoryConfig; optional LLM reranking via pluggable BaseLLMReranker interface

**TextEncoder:**
- Purpose: Deterministic encoding of text to sparse patterns with phase information
- Examples: `src/agentic/text_encoder.py`
- Pattern: Token-to-bit mapping via SHA256 hashing; phase encoding via word position; padding with deterministic bits to reach k sparsity; EncodedPattern output

**Coactivation:**
- Purpose: Learning rule for associative clustering via bit sharing among frequently-retrieved patterns
- Examples: `src/agentic/coactivation.py`
- Pattern: Functional module with CoactivationConfig; transfers ORIGINAL bits only to prevent transitive pollution; respects bidirectionality and obesity decay; random sampling for probabilistic bit transfer

## Entry Points

**AgentMemory Class:**
- Location: `src/agentic/agent_memory.py`
- Triggers: Application instantiation with dim, k, config, optional reranker
- Responsibilities: Provides store(), recall(), learn(), get() methods; tracks memory_count and average_obesity; orchestrates all agentic operations

**Quantum Substrate Functions:**
- Location: `src/quantum_substrate/binding.py`, `src/quantum_substrate/interference.py`
- Triggers: Called internally by agentic layer or directly by advanced applications
- Responsibilities: bind_hrr (circular convolution), unbind_hrr (correlation), similarity (cosine), jaccard_retrieval, interference_retrieval, create_related_pattern

## Error Handling

**Strategy:** Minimal error handling by design; assumes valid inputs and stable torch operations.

**Patterns:**
- Empty text handling: TextEncoder returns EncodedPattern with empty bits set
- Missing reranker: recall() checks `if use_reranking and self.reranker is not None` before calling rerank()
- Pattern not found: MemoryStore.get() returns None if pattern_id not in dict
- Coactivation on empty list: learn() returns early if no patterns provided

## Cross-Cutting Concerns

**Logging:** Not implemented; relies on pytest -v output for test visibility

**Validation:** Light validation in ComplexSparsePattern.__post_init__() for dimension bounds and index sorting; no input validation in agentic layer

**Authentication:** Not applicable; in-memory substrate with no external services

**Phase Management:** Circular mean computation in TextEncoder; phase normalization to [0, 2π) in patterns.py; phase preservation during coactivation bit transfer

---

*Architecture analysis: 2026-01-31*
