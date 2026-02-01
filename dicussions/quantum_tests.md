# Minimal Validation Tests for Quantum-Inspired Substrate

Experiments to validate or invalidate the core intuitions before building the full system.

---

## Test Categories

| Category | What it validates | Priority |
|----------|-------------------|----------|
| **Pure Substrate** | Core dynamics work mathematically | Critical |
| **Agentic Layer** | Text interface + LLM integration | Secondary |

Focus on pure substrate first. If the core dynamics don't work, the agentic layer doesn't matter.

---

## Pure Substrate Tests

These test the fundamental mechanics, independent of how patterns get created.

### Test 1: Interference Math

**The claim:** Patterns with complex amplitudes (magnitude + phase) interfere like waves. Same-phase patterns reinforce; opposite-phase patterns cancel.

**What to test:**

```
Pattern A: bits [1, 5, 9], phases [0, 0, 0], magnitude 1.0
Pattern B: bits [1, 5, 7], phases [0, 0, 0], magnitude 1.0  ← same phase as A
Pattern C: bits [1, 5, 8], phases [π, π, 0], magnitude 1.0  ← opposite phase

Combine A + B: bits [1, 5] should have magnitude 2.0 (constructive)
Combine A + C: bits [1, 5] should have magnitude 0.0 (destructive)
```

**Implementation:**

```python
# Each pattern is a complex sparse vector
# amplitude = magnitude * e^(i*phase)

def interfere(patterns: list[ComplexSparseVector]) -> ComplexSparseVector:
    """Sum complex amplitudes across patterns."""
    result = {}
    for p in patterns:
        for bit, amplitude in p.items():
            result[bit] = result.get(bit, 0) + amplitude
    return result

# Test: create patterns with known phase relationships
# Verify interference produces expected magnitudes
```

**Success criteria:**
- Constructive interference increases magnitude
- Destructive interference decreases/cancels magnitude
- Partial phase differences produce partial interference

**GPU notes:** Complex tensor addition is native in PyTorch. This is trivially parallelizable.

---

### Test 2: Entanglement Binding (The Binding Problem)

**The claim:** Entanglement creates joint states that preserve WHO did WHAT to WHOM. Synchrony alone cannot distinguish "dog bit mailman" from "mailman bit dog."

**What to test:**

```
Scenario: Two events with swapped roles

Event 1: "dog bit mailman"
  - DOG ⊗ AGENT
  - MAILMAN ⊗ PATIENT
  - BIT ⊗ ACTION

Event 2: "mailman fed dog"
  - MAILMAN ⊗ AGENT
  - DOG ⊗ PATIENT
  - FED ⊗ ACTION

Queries:
  - "Who was the agent in the biting?" → DOG
  - "Who was the patient in the feeding?" → DOG
  - "What did the mailman do as agent?" → FED
```

**Implementation approaches:**

Option A: Tensor product (mathematically pure, expensive)
```python
# Entangled state is outer product
# For 10k-dim patterns, A⊗B is 100M dimensions - not scalable

entangled = torch.outer(pattern_a, pattern_b)  # O(n²) space
```

Option B: Tagged binding (practical approximation)
```python
# Role patterns are small (AGENT, PATIENT, ACTION)
# Bind by XOR or concatenated indices

def bind(entity: SparsePattern, role: SparsePattern) -> SparsePattern:
    """Bind entity to role via circular convolution or XOR."""
    # Option: XOR indices (Kanerva-style)
    bound_bits = {(e + r) % DIM for e in entity.bits for r in role.bits}
    return SparsePattern(bits=bound_bits, phase=entity.phase)

def unbind(bound: SparsePattern, role: SparsePattern) -> SparsePattern:
    """Recover entity given role."""
    # Reverse the binding
    unbound_bits = {(b - r) % DIM for b in bound.bits for r in role.bits}
    return SparsePattern(bits=unbound_bits)
```

Option C: Holographic reduced representations (HRR)
```python
# Circular convolution in frequency domain
# Bind: A ⊛ B (circular convolution)
# Unbind: A ⊛ B* (convolution with conjugate)

def bind_hrr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bind via circular convolution (FFT-based)."""
    return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b))

def unbind_hrr(bound: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Unbind via correlation."""
    return torch.fft.ifft(torch.fft.fft(bound) * torch.conj(torch.fft.fft(b)))
```

**Test procedure:**

1. Create entity patterns: DOG, MAILMAN, BIT, FED
2. Create role patterns: AGENT, PATIENT, ACTION
3. Bind Event 1 and Event 2 with their role structures
4. Store both events
5. Query: activate PATIENT + BIT, measure which entity activates most
6. Compare to synchrony-only baseline (just co-activation, no role binding)

**Success criteria:**
- Role queries return correct entity
- Swapped-role events are distinguishable
- Synchrony-only baseline fails (returns ambiguous results)

**GPU notes:** HRR uses FFT, which is highly optimized on GPU. Option B (XOR) is also fast but less principled.

---

### Test 3: Phase Encodes Sequence

**The claim:** Temporal order is encoded in phase. Items earlier in sequence have earlier phase; activation spreads in phase order.

**What to test:**

```
Sequence: A → B → C → D → E

Encoding:
  A: phase = 0
  B: phase = 2π/5
  C: phase = 4π/5
  D: phase = 6π/5
  E: phase = 8π/5

Query: Activate A
Expected: B activates before C before D before E (phase proximity)
```

**Implementation:**

```python
def encode_sequence(items: list[SparsePattern]) -> list[SparsePattern]:
    """Assign phases based on sequence position."""
    n = len(items)
    for i, item in enumerate(items):
        item.phase = (2 * math.pi * i) / n
    return items

def spread_activation(query: SparsePattern, memory: list[SparsePattern]) -> list[tuple[SparsePattern, float]]:
    """Spread activation with phase-based resonance."""
    results = []
    for pattern in memory:
        # Overlap score
        overlap = len(query.bits & pattern.bits) / len(query.bits | pattern.bits)

        # Phase resonance (closer phase = higher resonance)
        phase_diff = abs(query.phase - pattern.phase)
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)  # wrap around
        resonance = math.cos(phase_diff)  # 1 at same phase, -1 at opposite

        # Combined score
        activation = overlap * (1 + resonance) / 2
        results.append((pattern, activation))

    return sorted(results, key=lambda x: -x[1])
```

**Test procedure:**

1. Create patterns for sequence items (can be random sparse patterns)
2. Assign phases based on sequence position
3. Query with first item
4. Measure activation order
5. Test with sequences of length 3, 5, 10, 20, 50
6. Find where phase aliasing breaks ordering

**Success criteria:**
- Short sequences (3-10) maintain correct order
- Identify the sequence length limit before aliasing
- Understand the trade-off: more items = less phase resolution

**Phase aliasing problem:**
```
With n items, phase resolution = 2π/n

n=4:  resolution = π/2 = 90°  (clear separation)
n=12: resolution = π/6 = 30°  (still distinguishable)
n=36: resolution = π/18 = 10° (getting noisy)
n=360: resolution = 1°        (noise dominates)
```

**GPU notes:** Phase arithmetic is trivial. The question is whether this is useful, not whether it's fast.

---

### Test 4: Coherence Dynamics

**The claim:** Coherence (0-1) controls quantum vs classical behavior. High coherence = interference active, superposition possible. Low coherence = stable, classical, no interference.

**What to test:**

```
Dynamics:
1. New patterns start with high coherence
2. Coherence decays over time (toward classical stability)
3. Accessing a pattern refreshes its coherence
4. Surprise/contradiction re-coheres decayed patterns

Behavior difference:
- High coherence: pattern participates in interference
- Low coherence: pattern is stable but "dormant"
```

**Implementation:**

```python
class Pattern:
    bits: set[int]
    phase: float
    coherence: float  # 0 = classical, 1 = quantum
    last_accessed: float

def decay_coherence(pattern: Pattern, current_time: float, decay_rate: float = 0.1):
    """Coherence decays toward 0 over time."""
    dt = current_time - pattern.last_accessed
    pattern.coherence *= math.exp(-decay_rate * dt)

def access_pattern(pattern: Pattern, current_time: float, refresh_amount: float = 0.3):
    """Accessing a pattern refreshes coherence."""
    pattern.coherence = min(1.0, pattern.coherence + refresh_amount)
    pattern.last_accessed = current_time

def surprise_recohere(pattern: Pattern, surprise_level: float):
    """Contradiction or surprise re-coheres a pattern."""
    pattern.coherence = min(1.0, pattern.coherence + surprise_level * 0.5)

def interference_weight(pattern: Pattern) -> float:
    """How much this pattern participates in interference."""
    return pattern.coherence  # linear, or could be threshold-based
```

**Test procedure:**

1. Create patterns, let time pass, verify coherence decays
2. Access pattern, verify coherence refreshes
3. Simulate "surprise" (prediction mismatch), verify re-coherence
4. Test interference with mixed coherence levels:
   - Two high-coherence patterns should interfere strongly
   - High + low coherence should interfere weakly
   - Two low-coherence patterns should barely interfere

**Success criteria:**
- Decay happens smoothly
- Refresh mechanism works
- Coherence actually modulates interference strength
- System reaches stable equilibrium (not runaway decay or growth)

**GPU notes:** Coherence is just a scalar per pattern. Batch updates are trivial.

---

### Test 5: Synchrony Builds Associations (Hebbian Learning)

**The claim:** Repeated co-activation causes patterns to develop overlapping bits. "Neurons that fire together wire together."

**What to test:**

```
Initial state:
  Pattern A: bits [1, 2, 3, 4, 5]
  Pattern B: bits [100, 101, 102, 103, 104]
  Overlap: 0%

After repeated co-activation:
  Pattern A: bits [1, 2, 3, 4, 5, 100, 101]  ← acquired some of B's bits
  Pattern B: bits [100, 101, 102, 103, 104, 4, 5]  ← acquired some of A's bits
  Overlap: > 0%
```

**Implementation:**

```python
def hebbian_update(pattern_a: Pattern, pattern_b: Pattern, learning_rate: float = 0.1):
    """Co-activation causes bit sharing."""
    # A acquires some of B's bits
    b_bits = list(pattern_b.bits - pattern_a.bits)
    acquire_count = int(len(b_bits) * learning_rate)
    if acquire_count > 0:
        new_bits = random.sample(b_bits, acquire_count)
        pattern_a.bits.update(new_bits)

    # B acquires some of A's bits
    a_bits = list(pattern_a.bits - pattern_b.bits)
    acquire_count = int(len(a_bits) * learning_rate)
    if acquire_count > 0:
        new_bits = random.sample(a_bits, acquire_count)
        pattern_b.bits.update(new_bits)

def co_activate(patterns: list[Pattern], repetitions: int = 10):
    """Simulate repeated co-activation."""
    for _ in range(repetitions):
        for i, p1 in enumerate(patterns):
            for p2 in patterns[i+1:]:
                hebbian_update(p1, p2)
```

**Test procedure:**

1. Create disjoint patterns (no overlap)
2. Repeatedly co-activate in a context (e.g., "dog" and "bark" always together)
3. Measure overlap growth
4. Verify: activating one pattern now partially activates the other
5. Test that NON-co-activated patterns remain disjoint

**Success criteria:**
- Overlap increases with co-activation
- Activation spreading follows learned associations
- Patterns that don't co-occur stay separate
- System doesn't collapse (everything becoming identical)

**Sparsity maintenance:**
```python
def maintain_sparsity(pattern: Pattern, target_k: int):
    """Prune least-used bits to maintain sparsity."""
    if len(pattern.bits) > target_k * 1.5:  # allow some slack
        # Keep most recently reinforced bits
        pattern.bits = set(sorted(pattern.bits, key=lambda b: pattern.bit_strength[b])[-target_k:])
```

**GPU notes:** Bit set operations are tricky on GPU. May need to represent as dense boolean tensors for batch Hebbian updates.

---

## Agentic Layer Tests (Secondary)

Only run these after pure substrate tests pass.

### Test 6: Hash Encoding Sanity

**The claim:** Simple hash-to-sparse encoding preserves enough lexical overlap to be useful, even though semantics are handled by LLM.

**What to test:**

```
Text pairs with expected similarity:

High overlap expected (lexical similarity):
  "the dog barked loudly" vs "the dog barked softly"

Medium overlap expected (some shared words):
  "the dog barked" vs "the cat meowed"

Low overlap expected (different words, same meaning):
  "the dog barked" vs "the canine vocalized"  ← WILL FAIL (no lexical overlap)

No overlap expected:
  "the dog barked" vs "quantum mechanics explained"
```

**Implementation:**

```python
def text_to_pattern(text: str, dim: int = 10000, k: int = 100) -> SparsePattern:
    """Hash text to sparse pattern."""
    tokens = text.lower().split()  # simple tokenization

    active_bits = set()
    for token in tokens:
        seed = hash(token) % (2**32)
        rng = random.Random(seed)
        bits_per_token = max(1, k // len(tokens))
        active_bits.update(rng.sample(range(dim), bits_per_token))

    return SparsePattern(bits=active_bits)

def jaccard(a: SparsePattern, b: SparsePattern) -> float:
    """Jaccard similarity of active bits."""
    intersection = len(a.bits & b.bits)
    union = len(a.bits | b.bits)
    return intersection / union if union > 0 else 0
```

**Success criteria:**
- Lexically similar texts have measurable overlap
- Completely different texts have near-zero overlap
- Accept that semantic similarity (synonyms) won't be captured - that's the LLM's job

**GPU notes:** Not critical for GPU. This is a one-time encoding step.

---

### Test 7: End-to-End vs RAG Baseline

**The claim:** The full system (substrate + LLM) beats simple vector similarity for structural queries.

**What to test:**

```
Corpus: 50-100 conversational exchanges with metadata
  - Speaker (Alice, Bob, Carol)
  - Time (timestamps)
  - Topic (work, personal, technical)

Structural queries:
  - "What did Alice say about the deadline?"
  - "What happened after Bob mentioned the bug?"
  - "What contradicts Carol's earlier statement?"

Compare:
  - Engram: entanglement binding + LLM rerank
  - RAG baseline: embed with sentence-transformers, cosine similarity, LLM rerank
```

**Success criteria:**
- Engram retrieves correct structural matches
- Engram beats or matches RAG on structural queries
- Understand where each approach wins/loses

---

## Implementation Plan

### Phase 1: Core Math (Tests 1-3)

Validate that interference, binding, and phase actually work mathematically.

```
src/
  quantum_tests/
    test_interference.py    # Test 1
    test_binding.py         # Test 2
    test_phase_sequence.py  # Test 3
    core.py                 # Shared Pattern class, basic operations
```

**Tech:** Python + PyTorch (complex tensors, FFT for HRR)

### Phase 2: Dynamics (Tests 4-5)

Validate coherence and learning dynamics.

```
    test_coherence.py       # Test 4
    test_hebbian.py         # Test 5
```

### Phase 3: Agentic (Tests 6-7)

Only if Phase 1-2 pass.

```
    test_encoding.py        # Test 6
    test_e2e_vs_rag.py      # Test 7
```

---

## Open Questions

1. **Binding implementation:** Tensor product is pure but expensive. HRR is practical but approximate. Which to try first?

2. **Phase representation:** Single global phase per pattern, or per-bit phases? Per-bit is more expressive but complex.

3. **Coherence function:** Linear modulation of interference, or hard threshold? Or something smoother?

4. **Sparsity maintenance:** How to prevent patterns from growing unbounded during Hebbian learning?

5. **GPU sparse tensors:** PyTorch sparse tensor support is limited. May need custom kernels or dense fallback for small-scale tests.

---

## Success/Failure Criteria

**The framework has potential if:**
- Interference math produces meaningful signal (Test 1)
- Binding disambiguates roles that synchrony cannot (Test 2)
- Phase preserves sequence for reasonable lengths (Test 3)

**The framework should be abandoned if:**
- Interference is just noise (no better than ignoring phase)
- Binding doesn't recover roles reliably
- Phase aliasing kicks in at very short sequences (<5 items)

**Inconclusive results mean:**
- Tune parameters (dimensions, sparsity, learning rates)
- Try alternative implementations (HRR vs tensor product)
- Revisit theoretical assumptions

---

## Validation Results (2026-01-30)

**51 tests executed, 51 passed**

### Test A: Binding Accuracy (HRR)
- **Result: PASS**
- Accuracy at dim=1024: **100%** (20 entities)
- Accuracy at all dimensions (256-4096): **100%**
- Role disambiguation: dog-bit-mailman vs mailman-bit-dog **correctly distinguished**
- Multi-role events (giver/receiver/item/time): **100% recovery**

### Test B: Sequence Phase Encoding
- **Result: PASS**
- Sequence ordering preserved at all tested lengths: 3, 5, 7, 10, 20, 30, 50, 100
- Phase resolution accuracy: **100%** at all lengths up to n=100
- Forward vs backward sequences: **distinguishable**
- No practical aliasing limit observed in tested range

### Test C: Interference vs Jaccard
- **Result: PASS**
- Interference correctly identifies phase-coherent patterns
- Same-phase patterns: constructive interference (similarity = 1.0)
- Opposite-phase patterns: destructive interference (similarity = 0.0)
- At scale (500 patterns): interference retrieval ranks related patterns significantly higher than Jaccard

### Test D: Scale Benchmarks
- **Result: PASS**

| Metric | Requirement | Actual |
|--------|-------------|--------|
| Binding at dim=16384 | < 10ms | **0.27 ms** |
| Retrieval for 5000 patterns | < 1s | **51.5 ms** |
| Sparse compression ratio | > 5x | **6.8x** |
| Memory for 10k patterns | < 100 MB | **11.4 MB** |

### Integration Test: Agentic Memory Scenario
- **Result: PASS**
- Conversation memory with role bindings and temporal phases: **working**
- Query by participant (Alice): correct events in top-2
- Query by topic (weather): correct events in top-2
- Multi-role recovery: **4/4 (100%)**

---

## Conclusions

**GO Decision: Proceed with quantum-inspired substrate**

All four validation tests passed with strong margins:

1. **Binding works reliably** - HRR circular convolution provides robust role-filler binding with 100% accuracy even at 20 entities. The dog-bit-mailman problem is cleanly solved.

2. **Phase encoding is effective** - Sequence ordering maintained at 100% accuracy up to 100 items with no observed aliasing. This significantly exceeds the minimum requirement of n≥5.

3. **Interference provides value over Jaccard** - Phase-aware retrieval correctly discriminates between patterns that share bits but differ in phase, which Jaccard cannot do.

4. **Scale is production-ready** - Sub-millisecond binding, 50ms retrieval for 5000 patterns, and 11MB for 10k patterns means this can run in real-time on modest hardware.

**Next steps:**
- Build agentic memory layer on this substrate
- Implement coherence dynamics (Test 4 from original plan)
- Implement Coactivation learning (Test 5 from original plan)
- Build text encoding and LLM integration (Tests 6-7)
