# Phase 4: Codebase Cleanup - Research

**Researched:** 2026-02-01
**Domain:** Determinism fixes (seeded RNG, hash function improvements)
**Confidence:** HIGH

## Summary

This phase addresses two determinism bugs in the existing codebase:
1. **FIX-01**: Non-deterministic coactivation due to use of global `random.sample()`
2. **FIX-02**: Text encoding collisions at small k values due to hash distribution issues

The coactivation fix requires adding seeded RNG support following the pattern already established in `tunneling.py`. The text encoding fix requires improving the hash function to reduce collisions for short strings at small k values.

**Key findings:**
- The tunneling module already demonstrates the correct pattern for seeded RNG (optional `rng: random.Random | None` parameter)
- Python's `random.Random` class provides isolated instances with independent state - exactly what we need
- SHA-256 truncation at 8 hex chars (32 bits) provides only ~4 billion unique values before modulo, which can cause collisions in small dimension spaces
- The fix should use more hash bits for better distribution before the modulo operation

**Primary recommendation:** Follow the existing `tunneling.py` pattern for RNG seeding; use full 64-bit hash extraction for better bit distribution in text encoding.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| random.Random | stdlib | Seeded pseudo-random number generation | Isolated instances with independent state; Mersenne Twister algorithm |
| hashlib | stdlib | Deterministic hashing | SHA-256 is available and sufficient; no external dependencies needed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-rng | 1.0+ | Test seeded fixtures | Optional - manual seeding via `random.Random(seed)` is sufficient |
| torch.Generator | PyTorch 2.0+ | Seeded RNG for tensor operations | Already used in quantum_substrate for pattern generation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| hashlib.sha256 | xxhash | 5-10x faster but adds dependency; SHA-256 is fast enough for text encoding |
| hashlib.sha256 | hashlib.blake2b | Faster with similar security; could consider if performance matters |

**Installation:**
```bash
# No new dependencies needed - using stdlib random and hashlib
```

## Architecture Patterns

### Pattern 1: Seeded RNG Parameter (from tunneling.py)
**What:** Optional `rng` parameter that defaults to global random when None
**When to use:** Any function that uses randomness and needs to be testable
**Example:**
```python
# Source: src/quantum_substrate/tunneling.py lines 91, 123
import random

def attempt_tunneling(
    source: "EvolvingPattern",
    # ... other params ...
    rng: random.Random | None = None,  # For deterministic testing
) -> TunnelingResult:
    """..."""
    _rng = rng if rng is not None else random

    # Use _rng.random(), _rng.sample(), _rng.choice() everywhere
    if _rng.random() >= probability:
        return ...
```

### Pattern 2: Session-Level Seed with Per-Call Override
**What:** Store a session seed that can be overridden per-call
**When to use:** When you want reproducibility across multiple operations
**Example:**
```python
# Session seed stored at module/class level
class CoactivationSession:
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else None

    def coactivate(
        self,
        patterns: list,
        seed: int | None = None,  # Override session seed
    ):
        # Per-call seed takes precedence, then session, then global
        if seed is not None:
            rng = random.Random(seed)
        elif self._rng is not None:
            rng = self._rng
        else:
            rng = random  # Global random (non-deterministic)

        # Use rng throughout
```

### Pattern 3: Metadata Tracking for Reproducibility
**What:** Store the seed used in pattern metadata for debugging
**When to use:** When you need to reproduce specific operations
**Example:**
```python
def coactivate(patterns, seed=None, ...):
    effective_seed = seed if seed is not None else "random"

    # Store in pattern metadata
    for pattern in patterns:
        pattern.metadata.setdefault("coactivation_seeds", [])
        pattern.metadata["coactivation_seeds"].append(effective_seed)
```

### Pattern 4: Better Hash Distribution for Bit Indices
**What:** Use more hash bits before modulo to improve distribution
**When to use:** When mapping strings to small integer spaces
**Example:**
```python
# Current (problematic for small dim):
h = hashlib.sha256(token.encode()).hexdigest()
bit_idx = int(h[:8], 16) % self.dim  # Only 32 bits

# Better (use 64 bits for better distribution):
h = hashlib.sha256(token.encode()).digest()
bit_idx = int.from_bytes(h[:8], 'big') % self.dim  # Full 64 bits
```

### Anti-Patterns to Avoid
- **Using global `random` module directly:** Creates non-deterministic behavior that can't be tested
- **Using `random.seed()` globally:** Affects all code using random; prefer isolated instances
- **Truncating hash too early:** Using only 32 bits (8 hex chars) increases collision probability
- **Not validating k parameter:** Small k values (< 10) are problematic for uniqueness

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Seeded RNG | Custom PRNG | `random.Random(seed)` | Isolated state, well-tested Mersenne Twister |
| Hash function | FNV/custom | `hashlib.sha256` | Cryptographic quality distribution, stdlib |
| Test reproducibility | Manual state management | pytest fixtures with seeded `random.Random` | Already established pattern in test suite |

**Key insight:** The codebase already has the correct pattern in `tunneling.py` - just apply it consistently to `coactivation.py`.

## Common Pitfalls

### Pitfall 1: Using Global Random State
**What goes wrong:** Tests pass sometimes and fail others; can't reproduce bugs
**Why it happens:** `random.sample()` uses global state affected by all code
**How to avoid:** Pass `random.Random(seed)` instance explicitly
**Warning signs:** Tests that fail intermittently; "it worked on my machine"

### Pitfall 2: Hash Truncation Collisions
**What goes wrong:** Different tokens map to same bit indices at small dimensions
**Why it happens:** Using only 32 bits of hash (8 hex chars) before modulo
**How to avoid:** Use 64 bits minimum; more bits = better distribution
**Warning signs:** Similar single-word inputs producing identical patterns

### Pitfall 3: Modifying RNG State Between Samples
**What goes wrong:** Intermediate operations change what bits get sampled
**Why it happens:** Using same RNG instance for multiple unrelated random operations
**How to avoid:** Isolate RNG per logical operation; consider deterministic ordering
**Warning signs:** Same seed produces different results after refactoring

### Pitfall 4: Seed Collision with None
**What goes wrong:** `random.Random(None)` seeds from system time, not reproducible
**Why it happens:** Passing `None` thinking it means "use this seed"
**How to avoid:** Check explicitly for `None` before creating seeded instance
**Warning signs:** Tests using `seed=None` not reproducing same results

### Pitfall 5: Small k with Large Corpus
**What goes wrong:** Unique words produce identical bit patterns
**Why it happens:** k=5 means only 5 bits to distinguish all words
**How to avoid:** Error on k < 10; document minimum k requirements
**Warning signs:** Collision tests failing; different texts returning identical scores

## Code Examples

Verified patterns from existing codebase:

### Seeded RNG Usage (from tunneling.py)
```python
# Source: src/quantum_substrate/tunneling.py
import random

def attempt_tunneling(
    source: "EvolvingPattern",
    source_id: str,
    all_patterns: dict[str, "EvolvingPattern"],
    connections: dict[str, set[str]],
    config: TunnelingConfig,
    creative_mode: bool = False,
    criticality_amplification: float = 1.0,
    rng: random.Random | None = None,  # For deterministic testing
) -> TunnelingResult:
    _rng = rng if rng is not None else random

    # ... later in function:
    if _rng.random() >= probability:
        return TunnelingResult(tunneled=False, ...)

    # Weighted selection
    r = _rng.random() * total_weight
```

### Test with Seeded RNG (from test_tunneling.py)
```python
# Source: tests/quantum_substrate/tunneling/test_tunneling.py
def test_high_coherence_can_tunnel(self):
    config = TunnelingConfig(baseline_probability=0.9, ...)
    p1 = EvolvingPattern.from_text("quantum physics", dim=1024, k=50)
    p2 = EvolvingPattern.from_text("medieval history", dim=1024, k=50)
    p1.coherence = 0.9

    rng = random.Random(42)  # Deterministic seed
    result = attempt_tunneling(
        p1, "p1", {"p1": p1, "p2": p2}, connections, config,
        creative_mode=True, rng=rng
    )
```

### Current Hash Function (to be improved)
```python
# Source: src/agentic/text_encoder.py lines 171-182
def _token_to_bits(self, token: str) -> set[int]:
    """Convert a token to a set of bit indices."""
    base_hash = hashlib.sha256(token.encode()).hexdigest()
    bits_per_token = max(1, self.k // 10)

    bits = set()
    for i in range(bits_per_token):
        h = hashlib.sha256(f"{base_hash}:{i}".encode()).hexdigest()
        bit_idx = int(h[:8], 16) % self.dim  # Only 32 bits - collision risk
        bits.add(bit_idx)

    return bits
```

### Improved Hash Function Pattern
```python
# Recommendation: Use 64 bits for better distribution
def _token_to_bits(self, token: str) -> set[int]:
    """Convert a token to a set of bit indices."""
    base_hash = hashlib.sha256(token.encode()).digest()  # bytes, not hex
    bits_per_token = max(1, self.k // 10)

    bits = set()
    for i in range(bits_per_token):
        h = hashlib.sha256(f"{token}:{i}".encode()).digest()
        # Use 8 bytes = 64 bits for better distribution
        bit_idx = int.from_bytes(h[:8], 'big') % self.dim
        bits.add(bit_idx)

    return bits
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Global `random.sample()` | Isolated `random.Random(seed)` instances | Best practice since Python 2.4 | Deterministic, testable |
| 8 hex char (32-bit) hash truncation | 8 byte (64-bit) binary extraction | Standard practice | Better distribution, fewer collisions |

**Deprecated/outdated:**
- Using `random.seed()` for test reproducibility: Creates global state pollution
- Using `hash()` builtin: Not stable across Python runs (PYTHONHASHSEED randomization)

## Open Questions

None - the requirements and approach are well-defined:

1. **RNG seeding approach:** Settled in CONTEXT.md - session-level seed with per-call override
2. **Hash function choice:** Settled - improve existing SHA-256 usage (no new dependencies)
3. **k validation:** Settled - error on k < 10

## Sources

### Primary (HIGH confidence)
- Python random module documentation: https://docs.python.org/3/library/random.html - Random class usage
- Existing codebase: `src/quantum_substrate/tunneling.py` - working seeded RNG pattern
- Existing codebase: `tests/quantum_substrate/tunneling/test_tunneling.py` - test patterns
- Python hashlib documentation: https://docs.python.org/3/library/hashlib.html - hash function usage

### Secondary (MEDIUM confidence)
- [Random Seeds and Reproducibility](https://medium.com/data-science/random-seeds-and-reproducibility-933da79446e3) - best practices
- [pytest-rng GitHub](https://github.com/nengo/pytest-rng) - fixture patterns

### Tertiary (LOW confidence)
- Web search on xxhash vs SHA-256 - performance comparisons (not needed, SHA-256 is sufficient)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - using Python stdlib only, well-documented
- Architecture: HIGH - patterns already exist in codebase (tunneling.py)
- Pitfalls: HIGH - common issues with seeded RNG are well-known

**Research date:** 2026-02-01
**Valid until:** 2026-03-01 (stable domain, patterns won't change)
