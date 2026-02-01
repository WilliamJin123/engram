# Phase 1: Coherence Foundation - Research

**Researched:** 2026-02-01
**Domain:** Quantum-inspired coherence dynamics for sparse distributed memory
**Confidence:** HIGH

## Summary

This phase implements the coherence field for patterns in the quantum-inspired memory substrate. Each pattern gains a coherence value (0-1 scalar) that decays over time (measured in operations) and refreshes on access. The implementation follows exponential decay dynamics from cognitive psychology's forgetting curve research, adapted to the quantum decoherence metaphor established in the project's architecture.

The existing codebase has a working sparse pattern system (`ComplexSparsePattern` and `EvolvingPattern`) with binding and interference retrieval. This phase adds coherence as a new property, implements decay triggered by a global operation tick mechanism, and refreshes coherence proportionally to activation strength on access operations (store, retrieve, coactivation, binding).

**Primary recommendation:** Extend `EvolvingPattern` with a coherence field and last-access tick, implement a `CoherenceManager` to track the global tick counter and apply decay/refresh across all patterns, and use structural embeddedness (degree centrality) to modulate per-pattern decay rates.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >=2.0.0 | Tensor operations, complex numbers | Already in project; handles math efficiently |
| dataclasses | stdlib | Pattern data structures | Already used; clean field extension |
| math | stdlib | exp() for decay calculations | Standard library, no dependencies |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=7.0.0 | Mathematical validation of decay | Already configured in project |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom tick counter | Wall-clock time | Decision: operations as time is locked per CONTEXT.md |
| Graph library (networkx) | Manual degree tracking | Overkill; simple dict-based counting sufficient for embeddedness |

**Installation:**
```bash
# No new dependencies required - all already present
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── quantum_substrate/
│   ├── patterns.py           # ComplexSparsePattern (unchanged)
│   ├── coherence.py          # NEW: CoherenceManager, decay/refresh logic
│   ├── binding.py            # HRR binding (unchanged)
│   └── interference.py       # May need coherence-weighting (Phase 2)
├── agentic/
│   ├── evolving_pattern.py   # Add coherence field
│   ├── memory_store.py       # Hook operations to CoherenceManager
│   └── coactivation.py       # Trigger refresh on coactivation
```

### Pattern 1: Coherence as Computed Property with Lazy Decay

**What:** Store `last_access_tick` and compute effective coherence on read using exponential decay formula.

**When to use:** When patterns are accessed infrequently relative to total operations.

**Example:**
```python
# Source: Cognitive psychology forgetting curve + quantum decoherence metaphor
@dataclass
class EvolvingPattern:
    # ... existing fields ...
    _coherence: float = 1.0           # Base coherence value
    last_access_tick: int = 0         # When pattern was last accessed

    def get_coherence(self, current_tick: int, decay_rate: float, floor: float = 0.01) -> float:
        """Compute effective coherence with decay since last access."""
        dt = current_tick - self.last_access_tick
        if dt <= 0:
            return min(self._coherence, 1.0)

        # Exponential decay: coherence *= exp(-decay_rate * dt)
        decayed = self._coherence * math.exp(-decay_rate * dt)
        return max(decayed, floor)  # Never below floor
```

### Pattern 2: Centralized Tick Manager

**What:** Single `CoherenceManager` instance tracks global tick counter, applies decay, and manages refresh.

**When to use:** When coordination between patterns and operations is needed.

**Example:**
```python
# Source: Design decision from CONTEXT.md - operations as time unit
class CoherenceManager:
    def __init__(self, decay_rate: float = 0.1, floor: float = 0.01):
        self.tick = 0
        self.decay_rate = decay_rate
        self.floor = floor
        self.pattern_connections: dict[str, set[str]] = {}  # For embeddedness

    def advance_tick(self) -> None:
        """Called once per operation."""
        self.tick += 1

    def decay_and_refresh(
        self,
        patterns: dict[str, EvolvingPattern],
        activated: dict[str, float],  # pattern_id -> activation_strength (0-1)
    ) -> None:
        """Apply decay to all, then refresh activated patterns."""
        # Step 1: Decay all patterns
        for pid, pattern in patterns.items():
            pattern._coherence = pattern.get_coherence(
                self.tick,
                self._effective_decay_rate(pid),
                self.floor
            )
            pattern.last_access_tick = self.tick

        # Step 2: Refresh activated patterns proportionally
        for pid, strength in activated.items():
            pattern = patterns[pid]
            # Refresh proportional to activation; cap at 1.0
            pattern._coherence = min(1.0, pattern._coherence + strength)
```

### Pattern 3: Structural Embeddedness via Degree Centrality

**What:** Track pattern connections (bindings, coactivations) and use degree as embeddedness metric.

**When to use:** To implement "highly connected patterns decay slower" decision.

**Example:**
```python
# Source: Graph centrality literature + CONTEXT.md decision
def _effective_decay_rate(self, pattern_id: str) -> float:
    """Decay rate modulated by structural embeddedness."""
    # More connections = lower decay rate
    degree = len(self.pattern_connections.get(pattern_id, set()))
    # Embeddedness factor: 1.0 for isolated, decreasing for connected
    # Using sigmoid-like scaling to bound effect
    embeddedness_factor = 1.0 / (1.0 + 0.1 * degree)
    return self.decay_rate * embeddedness_factor
```

### Anti-Patterns to Avoid
- **Wall-clock time decay:** Don't use `time.time()` - operations are the time unit per decision.
- **Per-operation decay on all patterns:** Don't iterate all patterns every operation. Use lazy decay computation on access.
- **Coherence overshoot:** Always `min(1.0, coherence + refresh)` to enforce cap.
- **Forgetting the floor:** Always `max(floor, coherence)` to prevent full decoherence.
- **Separate decay and refresh timing:** Decision states "decay happens first, then refresh" in same operation.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Exponential decay math | Custom approximation | `math.exp(-rate * dt)` | Precision, numerical stability |
| Graph degree counting | Graph traversal | Simple `dict[str, set[str]]` | Only need degree, not full graph |
| Tick synchronization | Per-pattern timers | Single manager tick | Consistency, testability |
| Activation propagation | Recursive graph walk | Bounded depth with decay factor | Performance (decision: limit multi-hop) |

**Key insight:** The coherence system is simpler than it appears. The core is just exponential decay + additive refresh. Complexity comes from:
1. Where to hook into operations (store, retrieve, coactivate, bind)
2. Computing structural embeddedness efficiently
3. Propagating indirect refresh through bindings

## Common Pitfalls

### Pitfall 1: Forgetting Order of Operations
**What goes wrong:** Refresh before decay, causing different steady-state behavior.
**Why it happens:** Natural to think "access refreshes" first.
**How to avoid:** CONTEXT.md is explicit: "decay all patterns first, then refresh activated patterns."
**Warning signs:** Frequently-accessed patterns never decay at all.

### Pitfall 2: Activation Strength Ambiguity
**What goes wrong:** Unclear what "activation strength" means for different operation types.
**Why it happens:** Store vs. retrieve vs. coactivate have different semantics.
**How to avoid:** Define clearly:
- Store: activation = 1.0 (explicit creation is full refresh)
- Retrieve: activation = similarity score (0-1)
- Coactivate: activation = coactivation strength from retrieval scores
- Bind: activation = 1.0 for both bound patterns
**Warning signs:** Tests fail because different operations refresh differently.

### Pitfall 3: Indirect Refresh Explosion
**What goes wrong:** Multi-hop propagation refreshes too many patterns.
**Why it happens:** Without depth limit, binding chains propagate indefinitely.
**How to avoid:**
- Limit propagation depth (2-3 hops max per CONTEXT.md discretion)
- Decay propagation strength: hop 1 = 15%, hop 2 = 2.25% (15% of 15%)
**Warning signs:** Unrelated patterns stay high-coherence.

### Pitfall 4: Test Non-Determinism
**What goes wrong:** Decay tests fail intermittently.
**Why it happens:** Tests don't control the tick counter state.
**How to avoid:** Always reset CoherenceManager in test fixtures; use explicit tick values.
**Warning signs:** Tests pass locally, fail in CI.

### Pitfall 5: Coherence-Weighted Retrieval Too Early
**What goes wrong:** Implementing coherence effects on interference in Phase 1.
**Why it happens:** Natural extension once coherence exists.
**How to avoid:** Phase 1 is just the field + decay + refresh. Phase 2 is coherence-weighted interference.
**Warning signs:** Scope creep, tests checking retrieval ranking changes.

## Code Examples

Verified patterns from official sources and project analysis:

### Extending EvolvingPattern with Coherence
```python
# Source: Existing evolving_pattern.py + design from research
@dataclass
class EvolvingPattern:
    dim: int
    bits: set[int]
    original_bits: frozenset[int]
    acquired_bits: set[int]
    phases: dict[int, float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    acquisition_count: int = 0
    # NEW coherence fields
    _coherence: float = field(default=1.0)  # Starts fully quantum
    last_access_tick: int = field(default=0)

    @property
    def coherence(self) -> float:
        """Current coherence value (may be stale until decay applied)."""
        return self._coherence
```

### Exponential Decay Implementation
```python
# Source: Forgetting curve research (Ebbinghaus) + CONTEXT.md spec
def apply_decay(coherence: float, dt: int, decay_rate: float, floor: float = 0.01) -> float:
    """Apply exponential decay: coherence *= exp(-decay_rate * dt).

    Args:
        coherence: Current coherence value (0-1).
        dt: Time elapsed in ticks.
        decay_rate: Decay constant (higher = faster decay).
        floor: Minimum coherence value (never fully decohere).

    Returns:
        Decayed coherence value, at least floor.

    Mathematical spec from CONTEXT.md:
        coherence(t) = coherence(0) * exp(-decay_rate * t)
    """
    if dt <= 0:
        return coherence
    decayed = coherence * math.exp(-decay_rate * dt)
    return max(decayed, floor)
```

### Mathematical Test for Decay Validation
```python
# Source: TEST-01 requirement + spec validation pattern
def test_decay_follows_exponential_formula():
    """TEST-01: Validate decay follows spec mathematically."""
    initial = 1.0
    decay_rate = 0.1
    floor = 0.01

    for dt in [1, 5, 10, 20, 50]:
        result = apply_decay(initial, dt, decay_rate, floor)
        expected = max(initial * math.exp(-decay_rate * dt), floor)

        assert abs(result - expected) < 1e-10, \
            f"dt={dt}: got {result}, expected {expected}"
```

### Refresh with Proportional Activation
```python
# Source: CONTEXT.md "refresh proportional to activation strength"
def apply_refresh(coherence: float, activation_strength: float) -> float:
    """Refresh coherence proportionally to activation.

    Args:
        coherence: Current coherence value (0-1).
        activation_strength: How strongly pattern was activated (0-1).

    Returns:
        Refreshed coherence, capped at 1.0.
    """
    # Hard cap at 1.0 per CONTEXT.md decision
    return min(1.0, coherence + activation_strength)
```

### Structural Embeddedness Tracking
```python
# Source: Graph centrality literature + CONTEXT.md discretion
class ConnectionTracker:
    """Track pattern connections for embeddedness calculation."""

    def __init__(self):
        self._connections: dict[str, set[str]] = {}

    def add_connection(self, pattern_a: str, pattern_b: str) -> None:
        """Record bidirectional connection between patterns."""
        self._connections.setdefault(pattern_a, set()).add(pattern_b)
        self._connections.setdefault(pattern_b, set()).add(pattern_a)

    def degree(self, pattern_id: str) -> int:
        """Get degree (number of connections) for pattern."""
        return len(self._connections.get(pattern_id, set()))

    def embeddedness_factor(self, pattern_id: str, scale: float = 0.1) -> float:
        """Compute embeddedness factor for decay rate modulation.

        Higher degree -> lower factor -> slower decay.
        """
        d = self.degree(pattern_id)
        return 1.0 / (1.0 + scale * d)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Static patterns | Evolving patterns | Already implemented | Enables learning |
| No time concept | Operation-based ticks | Phase 1 | Enables decay |
| Binary active/inactive | Coherence spectrum | Phase 1 | Enables quantum/classical continuum |

**Deprecated/outdated:**
- The existing `test_decay.py` has placeholder code that simulates coherence separately from patterns. This should be replaced with actual integration into `EvolvingPattern`.

## Open Questions

Things that couldn't be fully resolved:

1. **Exact decay rate constant value**
   - What we know: Cognitive literature uses various rates; quantum decoherence is domain-specific
   - What's unclear: What value produces desired behavior (patterns decay to floor in ~100-1000 operations?)
   - Recommendation: Start with `decay_rate = 0.05` (half-life ~14 operations), tune empirically

2. **Exact coherence floor value**
   - What we know: CONTEXT.md says "small" (e.g., 0.01)
   - What's unclear: How low is low enough without causing numerical issues?
   - Recommendation: Use `floor = 0.01` per CONTEXT.md suggestion

3. **Multi-hop propagation depth limit**
   - What we know: CONTEXT.md says "weak (~10-20% of direct), multi-hop with decay"
   - What's unclear: When does propagation overhead exceed value?
   - Recommendation: Limit to 2 hops initially (15% -> 2.25%), measure performance

4. **Embeddedness scale factor**
   - What we know: Degree centrality is the metric; higher = slower decay
   - What's unclear: How much should 10 connections slow decay vs 1?
   - Recommendation: Start with `scale = 0.1` (10 connections -> 50% decay rate), tune empirically

## Sources

### Primary (HIGH confidence)
- Existing codebase analysis: `src/quantum_substrate/patterns.py`, `src/agentic/evolving_pattern.py`
- CONTEXT.md decisions: All decay/refresh behavior specifications
- quantum_proposal.md: Coherence/decoherence architecture

### Secondary (MEDIUM confidence)
- [Forgetting Curve - Wikipedia](https://en.wikipedia.org/wiki/Forgetting_curve): Exponential decay model for memory
- [Graph Centrality - Wikipedia](https://en.wikipedia.org/wiki/Centrality): Degree centrality for embeddedness
- [Sparse Distributed Memory - Wikipedia](https://en.wikipedia.org/wiki/Sparse_distributed_memory): SDM decay models

### Tertiary (LOW confidence)
- [Distributed Associative Memory with Memory Refreshing Loss](https://www.sciencedirect.com/science/article/abs/pii/S0893608021003014): Neural network memory refresh patterns
- [QuTiP - Quantum Toolbox in Python](https://www.sciencedirect.com/science/article/pii/S0370157325002704): Decoherence simulation patterns (for metaphor only)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Already in project, no new dependencies
- Architecture: HIGH - Clear design from CONTEXT.md decisions
- Pitfalls: MEDIUM - Based on common patterns, some project-specific unknowns
- Code examples: HIGH - Derived directly from existing code and locked decisions

**Research date:** 2026-02-01
**Valid until:** 2026-03-01 (30 days - stable domain, locked decisions)
