# Phase 3: Advanced Dynamics - Research

**Researched:** 2026-02-01
**Domain:** Coherence semantics (malleability), tunneling for creative retrieval, criticality tuning
**Confidence:** MEDIUM-HIGH

## Summary

Phase 3 fundamentally changes what coherence means: from accessibility (high coherence = high retrieval weight) to malleability (high coherence = easy to change, low coherence = crystallized/stable). This shift aligns with neuroscience where synaptic strength (accessibility) is separate from synaptic plasticity (malleability). The phase also adds tunneling for creative/exploratory retrieval and a criticality parameter that tunes the balance between ordered and chaotic system behavior.

The existing Phase 2 implementation uses coherence to weight retrieval results - high coherence patterns dominate. Phase 3 inverts this: retrieval is based purely on similarity (bit overlap), while coherence governs only how much a pattern can change when faced with surprise/contradiction. This means well-embedded (highly connected) but crystallized (low coherence) patterns remain easily retrievable.

**Primary recommendation:** (1) Modify retrieval to use pure similarity without coherence weighting; (2) Add crystallization dynamics where patterns become low-coherence based on age + stability; (3) Implement tunneling allowing high-coherence patterns to probabilistically activate weakly-related patterns; (4) Add a global criticality parameter (0-1) that self-adjusts based on retrieval quality and surprise frequency.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >=2.0.0 | Tensor operations, random sampling | Already in project |
| math | stdlib | exp, log for decay curves and probability | Standard library |
| random | stdlib | Tunneling probabilistic selection | Already used in coactivation |
| dataclasses | stdlib | TunnelingResult, CriticalityState | Already used throughout |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=7.0.0 | TEST-05 validation | Already configured |
| typing | stdlib | Protocol, TypedDict for interfaces | Clarity |
| collections.abc | stdlib | Sequence, Mapping type hints | Already in use |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Random tunneling selection | numpy.random | Unnecessary dependency; random.choices sufficient |
| Complex criticality model | Simple float | Decision: start simple, one global parameter |
| Graph library for connections | Dict-based adjacency | networkx overkill for connection traversal |

**Installation:**
```bash
# No new dependencies required
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── quantum_substrate/
│   ├── coherence.py          # MODIFY: Add crystallization dynamics
│   ├── tunneling.py          # NEW: Tunneling mechanics
│   ├── criticality.py        # NEW: Criticality parameter management
│   └── surprise.py           # UNCHANGED
├── agentic/
│   ├── memory_store.py       # MODIFY: Change retrieval weighting
│   └── evolving_pattern.py   # MODIFY: Add stability tracking
```

### Pattern 1: Pure Similarity Retrieval (Removing Coherence Weighting)

**What:** Retrieve patterns based purely on bit overlap (Jaccard/interference score), not coherence.

**When to use:** All retrieval operations - coherence no longer affects accessibility.

**Example:**
```python
# Source: CONTEXT.md decision - "Remove embeddedness from explicit weighting"
# Connectivity naturally affects retrieval through network pathways

def _retrieve_by_similarity(
    self,
    query: EvolvingPattern,
    patterns: dict[str, EvolvingPattern],
    top_k: int,
    use_evolved: bool = True,
) -> list[RetrievalResult]:
    """Retrieve using pure similarity - no coherence weighting.

    Patterns are ranked purely by how well they match the query.
    Low-coherence (crystallized) patterns are just as retrievable
    as high-coherence patterns if they have similar content.
    """
    query_bits = query.bits if use_evolved else set(query.original_bits)

    results = []
    for pattern_id, pattern in patterns.items():
        pattern_bits = pattern.bits if use_evolved else set(pattern.original_bits)

        # Pure Jaccard similarity - no coherence factor
        intersection = len(query_bits & pattern_bits)
        union = len(query_bits | pattern_bits)
        score = intersection / union if union > 0 else 0.0

        results.append(RetrievalResult(
            pattern_id=pattern_id,
            pattern=pattern,
            score=score,
        ))

    results.sort(key=lambda r: -r.score)
    return results[:top_k]
```

### Pattern 2: Crystallization Dynamics

**What:** Patterns with high stability (unchanged over time, frequently accessed) crystallize to low coherence.

**When to use:** During coherence decay - patterns that haven't changed become harder to change.

**Example:**
```python
# Source: CONTEXT.md - "Crystallization driven by age + stability"
# "Patterns unchanged recently, frequently accessed" become crystallized

@dataclass
class StabilityTracker:
    """Track pattern stability for crystallization."""
    last_modified_tick: int = 0  # When pattern bits last changed
    access_count: int = 0  # Times retrieved since last change

    @property
    def stability_score(self) -> float:
        """Higher = more stable = crystallizes faster.

        Stability increases with:
        - Time since last modification
        - Frequent access without modification
        """
        # Access count without change indicates stability
        return min(1.0, self.access_count / 10.0)  # Cap at 10 accesses

def apply_crystallization_decay(
    self,
    pattern: EvolvingPattern,
    stability: StabilityTracker,
    base_decay_rate: float = 0.05,
) -> float:
    """Apply decay that accelerates with stability.

    Stable patterns crystallize faster (coherence drops faster).
    This is OPPOSITE to Phase 2 where high access = high coherence.

    Args:
        pattern: Pattern to decay.
        stability: Stability tracking for this pattern.
        base_decay_rate: Normal decay rate.

    Returns:
        New coherence value.
    """
    # Stability accelerates crystallization
    # More stable = faster decay toward floor
    effective_rate = base_decay_rate * (1.0 + stability.stability_score)

    dt = self._current_tick - pattern.last_access_tick
    if dt <= 0:
        return pattern.coherence

    # Embeddedness still slows decay (well-connected patterns more robust)
    effective_rate = effective_rate / pattern.embeddedness

    decayed = pattern.coherence * math.exp(-effective_rate * dt)
    pattern.coherence = max(self.config.floor, decayed)

    return pattern.coherence
```

### Pattern 3: Coherence-Scaled Contradiction Response

**What:** When surprise/contradiction affects a pattern, the response scales with coherence.

**When to use:** When applying re-coherence or pattern modification after surprise.

**Example:**
```python
# Source: CONTEXT.md - "Scale response by coherence"
# "Low-coherence patterns change proportionally less"

def apply_contradiction_response(
    pattern: EvolvingPattern,
    surprise: SurpriseResult,
    change_coefficient: float = 0.5,
) -> float:
    """Apply pattern change scaled by coherence (malleability).

    High coherence = malleable = changes easily
    Low coherence = crystallized = resists change

    Returns:
        Actual change magnitude applied.
    """
    if surprise.magnitude < 0.001:
        return 0.0

    # Potential change is based on surprise magnitude
    potential_change = surprise.magnitude * change_coefficient

    # Actual change is scaled by coherence (malleability)
    # Low coherence = small change, high coherence = full change
    actual_change = potential_change * pattern.coherence

    # Apply change (e.g., re-coherence boost, but could also be bit modification)
    headroom = 1.0 - pattern.coherence
    pattern.coherence += min(actual_change, headroom)

    return actual_change
```

### Pattern 4: Tunneling Mechanics

**What:** High-coherence patterns can probabilistically activate weakly-related patterns via connection paths.

**When to use:** During retrieval when creative/exploratory activation is desired.

**Example:**
```python
# Source: CONTEXT.md tunneling decisions:
# - "Always available at baseline, with explicit creative mode for amplification"
# - "Tunnel targets require BOTH: low bit overlap AND at least one connection path"
# - "Tunneling strength scales linearly with source coherence"

@dataclass
class TunnelingConfig:
    """Configuration for tunneling behavior."""
    baseline_probability: float = 0.1  # Base chance of tunnel activation
    creative_mode_multiplier: float = 3.0  # Amplification in creative mode
    min_source_coherence: float = 0.3  # Source must have this coherence
    max_bit_overlap: float = 0.2  # Target must have low overlap

@dataclass
class TunnelingResult:
    """Result of tunneling attempt."""
    tunneled: bool
    source_pattern_id: str
    target_pattern_id: str | None
    tunnel_strength: float  # How strong the activation was

def attempt_tunneling(
    source: EvolvingPattern,
    source_id: str,
    all_patterns: dict[str, EvolvingPattern],
    connections: dict[str, set[str]],  # pattern_id -> connected_pattern_ids
    config: TunnelingConfig,
    creative_mode: bool = False,
) -> TunnelingResult | None:
    """Attempt tunneling from high-coherence source to weak-relation target.

    Tunneling enables creative/exploratory retrieval by activating
    patterns that share few bits but have connection paths.

    Args:
        source: Pattern attempting to tunnel from.
        source_id: ID of source pattern.
        all_patterns: All patterns for finding targets.
        connections: Adjacency map of pattern connections.
        config: Tunneling parameters.
        creative_mode: If True, amplify tunneling probability.

    Returns:
        TunnelingResult if attempt made, None if source too low-coherence.
    """
    # Source must have sufficient coherence to tunnel
    if source.coherence < config.min_source_coherence:
        return None

    # Find candidate targets: low overlap but connected
    source_bits = source.bits
    candidates = []

    for target_id, target in all_patterns.items():
        if target_id == source_id:
            continue

        # Check connection path exists
        if target_id not in connections.get(source_id, set()):
            # Could also check multi-hop paths, but start with direct
            continue

        # Check low bit overlap
        overlap = len(source_bits & target.bits)
        union = len(source_bits | target.bits)
        overlap_ratio = overlap / union if union > 0 else 0.0

        if overlap_ratio < config.max_bit_overlap:
            candidates.append((target_id, target, overlap_ratio))

    if not candidates:
        return TunnelingResult(
            tunneled=False,
            source_pattern_id=source_id,
            target_pattern_id=None,
            tunnel_strength=0.0,
        )

    # Tunneling probability scales with source coherence
    base_prob = config.baseline_probability * source.coherence
    if creative_mode:
        base_prob *= config.creative_mode_multiplier

    # Probabilistic selection weighted by inverse overlap (more different = more interesting)
    import random
    if random.random() < base_prob:
        # Select target - prefer lower overlap
        weights = [1.0 - overlap_ratio for _, _, overlap_ratio in candidates]
        target_id, target, _ = random.choices(candidates, weights=weights, k=1)[0]

        tunnel_strength = source.coherence  # Linear scaling per CONTEXT.md

        return TunnelingResult(
            tunneled=True,
            source_pattern_id=source_id,
            target_pattern_id=target_id,
            tunnel_strength=tunnel_strength,
        )

    return TunnelingResult(
        tunneled=False,
        source_pattern_id=source_id,
        target_pattern_id=None,
        tunnel_strength=0.0,
    )
```

### Pattern 5: Auto-Creative Mode Triggers

**What:** System automatically enters creative mode when retrieval is struggling.

**When to use:** When retrieval confidence is low or repeated failures occur.

**Example:**
```python
# Source: CONTEXT.md - "Auto-creative triggers: low retrieval confidence
# AND repeated retrieval failures"

@dataclass
class CreativeModeTrigger:
    """Track conditions for auto-creative mode."""
    recent_scores: list[float]  # Last N top retrieval scores
    failure_threshold: float = 0.3  # Score below this = "failure"
    consecutive_failures_needed: int = 3

    def should_activate_creative(self, latest_score: float) -> bool:
        """Check if creative mode should auto-activate.

        Activates when:
        - Current score is low (below threshold)
        - AND we've had multiple recent low scores
        """
        self.recent_scores.append(latest_score)
        if len(self.recent_scores) > 10:
            self.recent_scores.pop(0)

        if latest_score >= self.failure_threshold:
            return False

        # Count recent consecutive failures
        consecutive = 0
        for score in reversed(self.recent_scores):
            if score < self.failure_threshold:
                consecutive += 1
            else:
                break

        return consecutive >= self.consecutive_failures_needed
```

### Pattern 6: Criticality Parameter

**What:** Global parameter (0-1) that controls order vs chaos balance, self-adjusting based on feedback.

**When to use:** Affects tunneling probability, noise in retrieval, pattern flexibility.

**Example:**
```python
# Source: CONTEXT.md criticality decisions:
# - "Edge of chaos model: 0.5 is optimal"
# - "Self-organizing: adjusts based on retrieval quality + surprise frequency"
# - "Global scope only"

@dataclass
class CriticalityState:
    """Global criticality parameter and adjustment logic."""
    value: float = 0.5  # Start at optimal edge-of-chaos
    adjustment_rate: float = 0.01  # How fast to adjust
    dampening: float = 0.9  # Resist rapid swings

    # Feedback tracking
    recent_retrieval_quality: list[float] = None  # Higher = better matches
    recent_surprise_frequency: list[bool] = None  # True if surprise occurred

    def __post_init__(self):
        if self.recent_retrieval_quality is None:
            self.recent_retrieval_quality = []
        if self.recent_surprise_frequency is None:
            self.recent_surprise_frequency = []

    def record_retrieval(self, top_score: float, had_surprise: bool) -> None:
        """Record retrieval outcome for self-adjustment."""
        self.recent_retrieval_quality.append(top_score)
        self.recent_surprise_frequency.append(had_surprise)

        # Keep window size bounded
        if len(self.recent_retrieval_quality) > 20:
            self.recent_retrieval_quality.pop(0)
        if len(self.recent_surprise_frequency) > 20:
            self.recent_surprise_frequency.pop(0)

    def self_adjust(self) -> float:
        """Self-adjust criticality based on feedback.

        - Poor matches (low quality) -> increase chaos (higher criticality)
        - Too many surprises -> decrease chaos (lower criticality)
        - Too few surprises -> increase chaos (system too rigid)

        Returns:
            New criticality value.
        """
        if len(self.recent_retrieval_quality) < 5:
            return self.value  # Not enough data

        # Compute signals
        avg_quality = sum(self.recent_retrieval_quality) / len(self.recent_retrieval_quality)
        surprise_rate = sum(self.recent_surprise_frequency) / len(self.recent_surprise_frequency)

        # Target surprise rate is around 0.2-0.4 (edge of chaos)
        target_surprise_rate = 0.3

        adjustment = 0.0

        # Poor retrieval quality -> more chaos
        if avg_quality < 0.5:
            adjustment += self.adjustment_rate

        # Too many surprises -> too chaotic -> less chaos
        if surprise_rate > 0.5:
            adjustment -= self.adjustment_rate
        # Too few surprises -> too rigid -> more chaos
        elif surprise_rate < 0.1:
            adjustment += self.adjustment_rate

        # Apply dampening
        adjustment *= self.dampening

        # Update with bounds
        self.value = max(0.0, min(1.0, self.value + adjustment))

        return self.value

    @property
    def tunneling_amplification(self) -> float:
        """Higher criticality = more tunneling."""
        # At criticality 0.5, amplification = 1.0
        # At criticality 1.0, amplification = 2.0
        # At criticality 0.0, amplification = 0.5
        return 0.5 + self.value

    @property
    def noise_level(self) -> float:
        """Higher criticality = more noise in retrieval."""
        return self.value * 0.1  # 0-10% noise
```

### Anti-Patterns to Avoid

- **Coherence affecting retrieval ranking:** Phase 3 removes this. Coherence = malleability only.
- **Tunneling without connection requirement:** Must have connection path to prevent random jumps.
- **Criticality extremes:** 0.0 and 1.0 should be problematic, not just less optimal.
- **Forgetting mechanism for low-connection patterns:** Per CONTEXT.md, zero/low-connection patterns remain retrievable by similarity.
- **Complex crystallization curves:** Start simple (linear/exponential), tune later.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Random weighted selection | Manual probability calculation | random.choices with weights | Built-in, tested |
| Connection graph traversal | Custom BFS/DFS | Dict-based adjacency lookup | Simple, already have connections |
| Exponential decay | Manual power calculation | math.exp | Numerical stability |
| Rolling window tracking | Custom circular buffer | List with pop(0) | Simple, sufficient for small windows |
| Parameter clamping | Custom if/else chains | max(min(val, upper), lower) | Clear, idiomatic |

**Key insight:** The complexity in Phase 3 is behavioral (what coherence means, how tunneling works) not computational. The actual operations are simple - it's the semantics that matter.

## Common Pitfalls

### Pitfall 1: Confusing Malleability with Accessibility

**What goes wrong:** Implementing coherence as retrieval weight instead of change resistance.
**Why it happens:** Phase 2 used coherence for retrieval weighting - mental model persists.
**How to avoid:** In retrieval code, explicitly comment "no coherence weighting - pure similarity."
**Warning signs:** Tests show low-coherence patterns not appearing in results.

### Pitfall 2: Tunneling to Unconnected Patterns

**What goes wrong:** Tunneling activates completely unrelated patterns.
**Why it happens:** Forgetting the "must have connection path" requirement.
**How to avoid:** Check connection map before considering a pattern as tunnel target.
**Warning signs:** Tunneling produces nonsensical associations.

### Pitfall 3: Criticality Oscillation

**What goes wrong:** Criticality swings rapidly between extremes.
**Why it happens:** Adjustment rate too high, insufficient dampening.
**How to avoid:** Use dampening factor (e.g., 0.9), small adjustment rate, rolling window for signals.
**Warning signs:** System behavior changes dramatically between operations.

### Pitfall 4: Crystallization Without Stability Tracking

**What goes wrong:** All old patterns crystallize regardless of whether they changed.
**Why it happens:** Using only age, not stability (unchanged + accessed).
**How to avoid:** Track last_modified_tick separately from last_access_tick.
**Warning signs:** Patterns that are frequently updated still crystallize.

### Pitfall 5: Creative Mode Always On

**What goes wrong:** Tunneling overwhelms normal retrieval.
**Why it happens:** Auto-creative threshold too sensitive, or baseline probability too high.
**How to avoid:** Require multiple consecutive failures, keep baseline probability low (0.1).
**Warning signs:** Retrieval results are mostly tunneled, not directly matched.

### Pitfall 6: Ignoring Embeddedness in Crystallization

**What goes wrong:** Well-connected patterns crystallize as fast as isolated ones.
**Why it happens:** Only using stability, ignoring embeddedness.
**How to avoid:** Embeddedness still slows decay (per Phase 1) - it affects rate, not direction.
**Warning signs:** Central hub patterns become rigid as fast as peripheral patterns.

## Code Examples

### Modified Retrieval Without Coherence Weighting

```python
# Source: CONTEXT.md - "Pure similarity as primary retrieval mechanism"
def retrieve(
    self,
    query: str,
    top_k: int = 10,
    recency_boost: bool = False,  # Optional enhancement per CONTEXT.md
) -> list[RetrievalResult]:
    """Retrieve patterns by similarity only.

    Coherence does NOT affect ranking - only malleability.
    Recency boost is optional per CONTEXT.md.
    """
    self.coherence_manager.advance_tick()
    self.coherence_manager.decay_all(self.patterns.values())

    query_pattern = EvolvingPattern.from_text(query, dim=self.dim, k=self.k)
    query_bits = query_pattern.bits

    results = []
    for pattern_id, pattern in self.patterns.items():
        # Pure Jaccard similarity
        intersection = len(query_bits & pattern.bits)
        union = len(query_bits | pattern.bits)
        score = intersection / union if union > 0 else 0.0

        # Optional recency boost
        if recency_boost:
            recency = 1.0 / (1.0 + (self.coherence_manager.current_tick - pattern.last_access_tick))
            score = score * 0.9 + recency * 0.1  # 90% similarity, 10% recency

        results.append(RetrievalResult(
            pattern_id=pattern_id,
            pattern=pattern,
            score=score,
        ))

    results.sort(key=lambda r: -r.score)

    # Refresh retrieved patterns (doesn't affect ranking, just updates state)
    for result in results[:top_k]:
        self.coherence_manager.apply_refresh(result.pattern, activation_strength=result.score)

    return results[:top_k]
```

### Retrieval with Tunneling Integration

```python
# Source: CONTEXT.md - "Always available at baseline, with explicit creative mode"
def retrieve_with_tunneling(
    self,
    query: str,
    top_k: int = 10,
    creative_mode: bool | None = None,  # None = auto-detect
) -> tuple[list[RetrievalResult], list[TunnelingResult]]:
    """Retrieve with optional tunneling for creative activation.

    Returns both direct matches and any tunneled activations.
    """
    # Normal retrieval first
    results = self.retrieve(query, top_k=top_k)

    # Determine creative mode
    if creative_mode is None:
        # Auto-detect based on retrieval quality
        top_score = results[0].score if results else 0.0
        creative_mode = self.creative_trigger.should_activate_creative(top_score)

    tunneled = []

    if results:
        # Attempt tunneling from top results
        for result in results[:3]:  # Top 3 can tunnel
            tunnel_result = attempt_tunneling(
                source=result.pattern,
                source_id=result.pattern_id,
                all_patterns=self.patterns,
                connections=self.connection_map,
                config=self.tunneling_config,
                creative_mode=creative_mode,
            )
            if tunnel_result and tunnel_result.tunneled:
                tunneled.append(tunnel_result)

                # Add tunneled pattern to results if not already present
                target_id = tunnel_result.target_pattern_id
                if target_id not in {r.pattern_id for r in results}:
                    target = self.patterns[target_id]
                    results.append(RetrievalResult(
                        pattern_id=target_id,
                        pattern=target,
                        score=tunnel_result.tunnel_strength * 0.5,  # Reduced score
                    ))

    # Re-sort and return
    results.sort(key=lambda r: -r.score)
    return results[:top_k], tunneled
```

### TEST-05: Tunneling Enables Creative Retrieval

```python
# Source: TEST-05 requirement
class TestTunnelingCreativeRetrieval:
    """TEST-05: Validate tunneling enables creative/exploratory activation."""

    def test_high_coherence_can_tunnel(self):
        """High-coherence patterns can activate weakly-related patterns."""
        from agentic.memory_store import MemoryStore

        store = MemoryStore()

        # Create connected patterns with low overlap
        id_cats = store.store("cats are furry pets")
        id_dogs = store.store("dogs are loyal companions")  # Different topic

        # Create connection between them
        store.create_connection(id_cats, id_dogs)

        # Refresh cats to high coherence
        store.coherence_manager.apply_refresh(store.patterns[id_cats], 1.0)

        # Query about cats - dogs might tunnel in
        results, tunneled = store.retrieve_with_tunneling(
            "cats furry",
            creative_mode=True
        )

        # With tunneling, dogs might appear despite low overlap
        if tunneled:
            assert any(t.target_pattern_id == id_dogs for t in tunneled), \
                "Tunneling should reach connected pattern"

    def test_low_coherence_cannot_tunnel(self):
        """Low-coherence patterns cannot initiate tunneling."""
        from agentic.memory_store import MemoryStore
        from quantum_substrate.coherence import CoherenceConfig

        config = CoherenceConfig(decay_rate=0.5, floor=0.01)
        store = MemoryStore(coherence_config=config)

        id_a = store.store("pattern A content")
        id_b = store.store("pattern B different")
        store.create_connection(id_a, id_b)

        # Decay A to low coherence
        for _ in range(50):
            store.coherence_manager.advance_tick()
        store.coherence_manager.apply_decay(store.patterns[id_a])

        assert store.patterns[id_a].coherence < 0.1, "Should be low coherence"

        # Tunneling from low-coherence should fail
        results, tunneled = store.retrieve_with_tunneling(
            "pattern A",
            creative_mode=True
        )

        # A should be in results (similarity-based) but shouldn't tunnel
        a_tunneled = [t for t in tunneled if t.source_pattern_id == id_a]
        assert all(not t.tunneled for t in a_tunneled), \
            "Low-coherence pattern should not tunnel"

    def test_tunneling_requires_connection(self):
        """Tunneling only reaches connected patterns."""
        from agentic.memory_store import MemoryStore

        store = MemoryStore()

        id_a = store.store("topic A about cats")
        id_b = store.store("topic B completely different")  # No connection

        # No connection created
        store.coherence_manager.apply_refresh(store.patterns[id_a], 1.0)

        results, tunneled = store.retrieve_with_tunneling(
            "cats topic A",
            creative_mode=True
        )

        # B should not appear in tunneled results (no connection)
        assert not any(t.target_pattern_id == id_b for t in tunneled), \
            "Cannot tunnel to unconnected pattern"
```

## State of the Art

| Old Approach (Phase 2) | New Approach (Phase 3) | Impact |
|------------------------|------------------------|--------|
| Coherence = retrieval weight | Coherence = malleability only | Crystallized patterns still retrievable |
| Explicit embeddedness weighting | Connectivity emerges from network | Simpler retrieval, richer dynamics |
| No creative retrieval | Tunneling for exploration | System can make creative leaps |
| Fixed system parameters | Self-adjusting criticality | System adapts to content domain |

**Key paradigm shift:** Phase 2 coherence was "quantum visibility" - high coherence meant the pattern dominated results. Phase 3 coherence is "quantum uncertainty" - high coherence means the pattern is malleable/uncertain, low coherence means crystallized/stable.

## Research Findings

### Self-Organized Criticality (SOC)

Research from [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12213684/) and [Frontiers in Systems Neuroscience](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2025.1590743/full) (2025) confirms the "edge of chaos" model:

- Brain criticality hypothesis: optimal information processing at the boundary between order and disorder
- SOC mechanism maintains critical state through internal parameter regulation (e.g., synaptic plasticity)
- Network structure influences the time scales at which plasticity achieves criticality
- Hebbian learning can disrupt criticality unless alternated with recovery periods

**Implication for Phase 3:** The criticality parameter should self-adjust slowly with dampening. Rapid changes would disrupt the critical state.

### Memory Plasticity vs Accessibility

Research on [synaptic consolidation](https://arxiv.org/html/2405.16922v2) and [memory reconsolidation](https://www.sciencedirect.com/science/article/abs/pii/S0301008217302174) supports the malleability/accessibility distinction:

- Synaptic strength (accessibility) is separate from synaptic plasticity (malleability)
- Active memory is malleable and can integrate new information
- Memory stability exists in tension with plasticity
- Well-consolidated memories are both accessible AND resistant to change

**Implication for Phase 3:** The design correctly separates accessibility (from similarity/connections) and malleability (from coherence).

### Tunneling and Creative Retrieval

Research on [quantum associative memory](https://arxiv.org/html/2408.14272v1) describes metastable patterns that allow temporary associations before convergence - similar to our tunneling concept. The quantum framework enables "exploratory possibilities through superposition-based pattern exploration."

**Implication for Phase 3:** Tunneling is theoretically grounded in quantum dynamics. The "always available at baseline" decision aligns with this research.

### Edge of Chaos Computation

Research from [MDPI](https://www.mdpi.com/2079-7737/10/8/702) on optimal neural performance shows:

- Maximum Lyapunov exponent near zero indicates critical point
- Optimal performance occurs just below chaos onset
- Power-law eigenspectrum decay indicates criticality
- Too ordered = rigid, too chaotic = unstable

**Implication for Phase 3:** Criticality value 0.5 as "optimal" is well-supported. Self-adjustment toward this value makes sense.

## Open Questions

### 1. Crystallization Decay Curve (Claude's Discretion)

- **What we know:** Stability accelerates crystallization
- **What's unclear:** Exact functional form - linear, exponential, sigmoid?
- **Recommendation:** Start with linear acceleration (stability_score * base_rate), can tune to exponential if patterns crystallize too slowly.

### 2. Auto-Creative Thresholds (Claude's Discretion)

- **What we know:** Low retrieval confidence AND repeated failures trigger creative mode
- **What's unclear:** What score counts as "low"? How many failures needed?
- **Recommendation:** Start with score < 0.3 as "low", 3 consecutive failures to trigger. Expose as config parameters.

### 3. Criticality Adjustment Rate (Claude's Discretion)

- **What we know:** Self-organizing, based on feedback signals
- **What's unclear:** How fast should it adjust? Too fast = oscillation, too slow = unresponsive
- **Recommendation:** Start with adjustment_rate=0.01, dampening=0.9. Monitor in tests.

### 4. Linear vs Exponential Tunneling (Claude's Discretion)

- **What we know:** CONTEXT.md specifies linear scaling with coherence
- **What's unclear:** Whether exponential might work better for some scenarios
- **Recommendation:** Implement linear per decision, add exponent parameter for experimentation.

### 5. Connection Map Structure

- **What we know:** Tunneling requires connection paths
- **What's unclear:** Should connections be directional? Weighted?
- **Recommendation:** Start with bidirectional unweighted (simplest). The existing coactivation already creates connections; leverage that data structure.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/quantum_substrate/coherence.py`, `src/agentic/memory_store.py`, `src/agentic/coactivation.py`
- `03-CONTEXT.md`: All locked decisions for coherence semantics, tunneling, criticality
- `quantum_proposal.md`: Theoretical foundation for coherence dynamics, tunneling concept
- Phase 1 & 2 implementations: Established patterns for coherence, surprise, embeddedness

### Secondary (MEDIUM confidence)
- [Network structure influences self-organized criticality](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2025.1590743/full) - SOC in neural networks
- [Theories of synaptic memory consolidation](https://arxiv.org/html/2405.16922v2) - Stability/plasticity dilemma
- [Theoretical framework for quantum associative memories](https://arxiv.org/html/2408.14272v1) - Metastable patterns
- [Optimal Input Representation at Edge of Chaos](https://pmc.ncbi.nlm.nih.gov/articles/PMC8389338/) - Criticality measurement

### Tertiary (LOW confidence)
- [Sparse Distributed Memory - Wikipedia](https://en.wikipedia.org/wiki/Sparse_distributed_memory) - SDM fundamentals
- [Self-organized criticality in neural networks - arXiv](https://arxiv.org/abs/2107.03402) - SOC theory
- General neuroscience literature on metaplasticity and synaptic tagging

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new dependencies
- Architecture: HIGH - Clear design from CONTEXT.md decisions
- Coherence semantics: HIGH - Well-defined in context document
- Tunneling mechanics: MEDIUM - Clear concept, implementation details TBD
- Criticality self-adjustment: MEDIUM - Theory sound, tuning needed empirically
- Pitfalls: MEDIUM - Based on common patterns and research

**Research date:** 2026-02-01
**Valid until:** 2026-03-01 (30 days - stable domain with locked decisions)
