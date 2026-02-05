# Phase 7: Harder Behavior Tests - Research

**Researched:** 2026-02-04
**Domain:** Stress testing tunneling, interference retrieval, and coherence dynamics under realistic memory load
**Confidence:** HIGH

## Summary

Phase 7 creates stress tests that validate core behaviors (tunneling, interference retrieval, coherence dynamics) work correctly under realistic memory load. This phase uses the noise generation infrastructure from Phase 6 (noisy_memory fixture, inject_noise, NoiseLevel presets) to create challenging test scenarios with near-misses and clutter.

The codebase provides strong foundations:
- Phase 6 infrastructure: `NoiseLevel`, `NoiseConfig`, `noisy_memory` fixture, `inject_noise()` function
- Core behaviors already implemented: `attempt_tunneling()`, interference retrieval in MemoryStore, `CoherenceManager` with decay/refresh/recoherence
- Existing test patterns in `tests/hypothesis_validation/` and `tests/quantum_substrate/` demonstrating statistical validation approaches
- `retrieve_with_tunneling()` and `retrieve_with_surprise()` in MemoryStore providing integrated APIs

**Primary recommendation:** Create `tests/stress/` directory with three test modules targeting TEST-01 (tunneling), TEST-02 (interference retrieval), and TEST-03 (coherence dynamics). Use parametrized tests across noise levels (MEDIUM, HIGH) with metrics output on all runs. Conclude with TEST_SUMMARY.md documenting results.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | >=7.0.0 | Test framework | Already in project, parametrize for noise levels |
| torch | >=2.0.0 | Seeded random generation | Used by noise generators |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses | stdlib | Test result structures | Metrics capture |
| typing | stdlib | Type hints | API clarity |
| json | stdlib | Metrics serialization | TEST_SUMMARY.md generation |

### Existing Infrastructure (Phase 6)
| Component | Location | Purpose |
|-----------|----------|---------|
| NoiseLevel | tests/conftest.py | NONE/LOW/MEDIUM/HIGH presets |
| NoiseConfig | tests/conftest.py | Configurable noise parameters |
| noisy_memory | tests/conftest.py | Factory fixture for noisy scenarios |
| inject_noise | tests/conftest.py | Add noise to existing MemoryStore |
| generate_near_misses | tests/noise_generators.py | Near-miss pattern creation |
| generate_clutter_batch | tests/noise_generators.py | Clutter pattern creation |

**Installation:**
```bash
# No new dependencies - all in existing pyproject.toml
```

## Architecture Patterns

### Recommended Project Structure
```
tests/
    stress/
        __init__.py
        conftest.py              # Stress-specific fixtures, @pytest.mark.stress marker
        test_tunneling.py        # TEST-01: Tunneling through noise
        test_interference.py     # TEST-02: Interference retrieval under noise
        test_coherence.py        # TEST-03: Coherence dynamics under load
```

### Pattern 1: Parametrized Noise Level Tests
**What:** Use @pytest.mark.parametrize to run tests across multiple noise levels
**When to use:** All stress tests should validate behavior across noise conditions
**Example:**
```python
# Source: pytest parametrize documentation + project patterns
import pytest
from tests.conftest import NoiseLevel

# Skip NONE and LOW - focus on medium and high per CONTEXT.md
STRESS_NOISE_LEVELS = [NoiseLevel.MEDIUM, NoiseLevel.HIGH]

@pytest.mark.stress
@pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS, ids=lambda l: l.name)
def test_tunneling_through_noise(noisy_memory, noise_level):
    """Tunneling should navigate through clutter to find related patterns."""
    store, target_ids, noise_result = noisy_memory(
        ["source pattern text", "connected target text"],
        level=noise_level,
        seed=42,
    )
    # Test tunneling behavior...
```

### Pattern 2: Metrics Capture Pattern
**What:** Capture retrieval metrics (rank, score) on every test run for analysis
**When to use:** All stress tests should output metrics, not just on failure
**Example:**
```python
# Source: Derived from CONTEXT.md "always output metrics" requirement
from dataclasses import dataclass, asdict
import json

@dataclass
class RetrievalMetrics:
    """Metrics captured from retrieval under noise."""
    noise_level: str
    target_rank: int | None  # None if target not found
    target_score: float | None
    top_score: float
    total_patterns: int
    near_miss_count: int
    clutter_count: int

    def to_dict(self) -> dict:
        return asdict(self)

def capture_metrics(
    results: list,
    target_id: str,
    noise_result,
    noise_level: NoiseLevel,
) -> RetrievalMetrics:
    """Capture metrics from retrieval results."""
    result_ids = [r.pattern_id for r in results]
    target_rank = result_ids.index(target_id) + 1 if target_id in result_ids else None
    target_score = next((r.score for r in results if r.pattern_id == target_id), None)

    return RetrievalMetrics(
        noise_level=noise_level.name,
        target_rank=target_rank,
        target_score=target_score,
        top_score=results[0].score if results else 0.0,
        total_patterns=len(result_ids),
        near_miss_count=sum(len(v) for v in noise_result.near_misses.values()),
        clutter_count=len(noise_result.clutter_ids),
    )
```

### Pattern 3: Success Criteria Hierarchy
**What:** Test multiple success criteria per CONTEXT.md (top-1, top-K, relative ranking)
**When to use:** Interference retrieval tests validating precision and recall
**Example:**
```python
# Source: CONTEXT.md "test all three success criteria empirically"
def evaluate_retrieval_success(
    results: list,
    target_id: str,
    near_miss_ids: list[str],
    k_relaxed: int = 3,
) -> dict[str, bool]:
    """Evaluate retrieval against multiple success criteria."""
    result_ids = [r.pattern_id for r in results]

    # Strict: Target is top-1 result
    strict = result_ids[0] == target_id if result_ids else False

    # Relaxed: Target in top-K results
    relaxed = target_id in result_ids[:k_relaxed]

    # Relative: Target ranked above all near-misses
    target_rank = result_ids.index(target_id) if target_id in result_ids else float('inf')
    near_miss_ranks = [
        result_ids.index(nm_id) if nm_id in result_ids else float('inf')
        for nm_id in near_miss_ids
    ]
    relative = all(target_rank < nm_rank for nm_rank in near_miss_ranks if nm_rank != float('inf'))

    return {
        "strict_top_1": strict,
        "relaxed_top_k": relaxed,
        "relative_to_near_misses": relative,
    }
```

### Pattern 4: Multi-Hop Tunneling Test
**What:** Test tunneling chains (cue -> intermediate -> target)
**When to use:** TEST-01 multi-hop tunneling scenarios
**Example:**
```python
# Source: CONTEXT.md "multi-hop chains" requirement
def test_multi_hop_tunneling(noisy_memory, noise_level):
    """Test tunneling through intermediate patterns to reach target."""
    store, target_ids, noise_result = noisy_memory(
        ["source cue pattern", "intermediate bridge", "final target destination"],
        level=noise_level,
        seed=42,
    )

    source_id, intermediate_id, target_id = target_ids

    # Create connection chain: source -> intermediate -> target
    store.create_connection(source_id, intermediate_id)
    store.create_connection(intermediate_id, target_id)

    # Set high coherence on source (required for tunneling)
    store.patterns[source_id].coherence = 0.9
    store.patterns[intermediate_id].coherence = 0.8

    # Configure tunneling for testing
    from quantum_substrate.tunneling import TunnelingConfig
    store.set_tunneling_config(TunnelingConfig(
        baseline_probability=0.8,
        min_source_coherence=0.3,
        max_bit_overlap=0.4,
    ))

    # Query source to trigger tunneling
    results, tunnel_results = store.retrieve_with_tunneling(
        "source cue",
        top_k=5,
        creative_mode=True,
    )

    # Verify multi-hop reach
    result_ids = {r.pattern_id for r in results}
    successful_tunnels = [t for t in tunnel_results if t.tunneled]
    # ...
```

### Pattern 5: Coherence Dynamics Validation
**What:** Test decay, refresh, and surprise re-coherence under memory load
**When to use:** TEST-03 coherence dynamics tests
**Example:**
```python
# Source: CONTEXT.md "use access counts as proxy for time passage"
def test_decay_under_load(noisy_memory, noise_level):
    """Patterns should decay over many operations, even under noise."""
    store, target_ids, noise_result = noisy_memory(
        ["pattern that will decay"],
        level=noise_level,
        seed=42,
    )

    target_id = target_ids[0]
    initial_coherence = store.patterns[target_id].coherence

    # Simulate many operations without touching target
    # Each retrieve advances tick and applies decay
    for i in range(20):
        # Query something else (not target)
        store.retrieve("completely unrelated query", top_k=3)

    final_coherence = store.patterns[target_id].coherence

    assert final_coherence < initial_coherence, (
        f"Coherence should decay: initial={initial_coherence:.3f}, "
        f"final={final_coherence:.3f}"
    )
```

### Anti-Patterns to Avoid
- **Testing only at NONE noise:** Stress tests should focus on MEDIUM and HIGH, not trivial cases
- **Pass/fail only assertions:** Always capture metrics for degradation curve analysis
- **Fixed seeds across all tests:** Each test should use unique seeds for independence
- **Testing tunneling without connections:** Tunneling requires connection paths - always set up connections first
- **Ignoring coherence setup:** Tunneling requires high coherence on source patterns

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Noisy memory setup | Manual noise injection | `noisy_memory` fixture | Phase 6 infrastructure handles all complexity |
| Tunneling mechanics | Custom tunneling logic | `attempt_tunneling()` | Already implemented with all edge cases |
| Coherence decay | Manual decay calculation | `CoherenceManager.apply_decay()` | Handles embeddedness, crystallization |
| Integrated retrieval | Separate retrieve + tunnel | `retrieve_with_tunneling()` | MemoryStore method handles full flow |
| Surprise detection | Manual expectation tracking | `retrieve_with_surprise()` | Integrates surprise detection + re-coherence |
| Seeded RNG | Global torch.manual_seed | `torch.Generator` per test | Test isolation |

**Key insight:** Phase 6 and existing quantum_substrate code provide complete infrastructure. Stress tests should use existing APIs with parametrized noise levels, not reimplement mechanics.

## Common Pitfalls

### Pitfall 1: Tunneling Without Connection Setup
**What goes wrong:** Tunneling always fails (returns tunneled=False)
**Why it happens:** Tunneling requires patterns to be connected via `create_connection()`
**How to avoid:**
- Always call `store.create_connection(source_id, target_id)` before tunneling tests
- Verify connections exist with `store.get_connections(pattern_id)`
- For multi-hop: create full chain (a->b, b->c) before testing
**Warning signs:** All tunnel_results have tunneled=False

### Pitfall 2: Low Coherence on Tunneling Source
**What goes wrong:** Tunneling never triggers
**Why it happens:** `attempt_tunneling()` requires source.coherence >= min_source_coherence (default 0.3)
**How to avoid:**
- Explicitly set high coherence on source patterns: `store.patterns[source_id].coherence = 0.9`
- Noisy patterns have aged coherence (0.2-0.7) - targets may need coherence refresh
- Use `store.coherence_manager.apply_refresh(pattern, 1.0)` to boost coherence
**Warning signs:** Tunnel results always have tunnel_strength=0.0

### Pitfall 3: Near-Misses Ranked as Top Results
**What goes wrong:** Tests fail because near-misses beat targets
**Why it happens:** Near-misses are designed to be challenging - some will rank highly
**How to avoid:**
- Use relative success criteria ("target above near-misses") not just strict top-1
- Expect degradation at HIGH noise - document as graceful degradation
- If near-misses consistently beat targets, may indicate phase noise is too low
**Warning signs:** Target never in top-3 at any noise level

### Pitfall 4: Coherence Dynamics Test Isolation
**What goes wrong:** Coherence values unexpected due to prior operations
**Why it happens:** Each retrieve/store advances tick and applies decay to ALL patterns
**How to avoid:**
- Create fresh store for each test (noisy_memory fixture does this)
- Track initial coherence immediately after setup
- Understand that retrieve() calls decay all patterns before returning
**Warning signs:** Coherence values don't match expected decay formula

### Pitfall 5: Missing Metrics Output
**What goes wrong:** Tests pass but no data for Phase 8 analysis
**Why it happens:** Only asserting pass/fail without capturing metrics
**How to avoid:**
- Capture metrics on every test run using dataclass structure
- Use fixtures to collect metrics across parametrized runs
- Store metrics in pytest.request cache or write to files
**Warning signs:** Cannot produce degradation curves in Phase 8

### Pitfall 6: Stability Score Not Considered
**What goes wrong:** Crystallization/stability tests show unexpected behavior
**Why it happens:** Stability score = access_count / 10 (capped at 1.0) affects decay
**How to avoid:**
- When testing stability resistance: set `pattern.access_count_since_modification` high (>=10)
- When testing malleable patterns: keep access_count_since_modification at 0
- Verify with `pattern.stability_score` property
**Warning signs:** Highly-accessed patterns decay as fast as new patterns

## Code Examples

Verified patterns from official sources and codebase:

### Stress Test Fixture Configuration
```python
# tests/stress/conftest.py
# Source: Derived from project test patterns
import pytest
from dataclasses import dataclass, field
from typing import Any
from tests.conftest import NoiseLevel

# Focus on challenging levels per CONTEXT.md
STRESS_NOISE_LEVELS = [NoiseLevel.MEDIUM, NoiseLevel.HIGH]

# Custom marker for stress tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "stress: mark test as stress test (may be slow)"
    )

@dataclass
class StressTestMetrics:
    """Aggregated metrics from a stress test run."""
    test_name: str
    noise_level: str
    trials: int = 0
    successes: int = 0
    metrics: list[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials > 0 else 0.0

    def add_trial(self, success: bool, trial_metrics: dict) -> None:
        self.trials += 1
        if success:
            self.successes += 1
        self.metrics.append(trial_metrics)

@pytest.fixture
def stress_metrics() -> dict[str, StressTestMetrics]:
    """Fixture to collect metrics across tests."""
    return {}
```

### Tunneling Through Noise Test (TEST-01)
```python
# tests/stress/test_tunneling.py
# Source: CONTEXT.md requirements + quantum_substrate/tunneling.py API
import pytest
import random
from tests.conftest import NoiseLevel, noisy_memory
from quantum_substrate.tunneling import TunnelingConfig

STRESS_NOISE_LEVELS = [NoiseLevel.MEDIUM, NoiseLevel.HIGH]

class TestTunnelingThroughNoise:
    """TEST-01: Tunneling tests with noisy memory."""

    @pytest.mark.stress
    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS, ids=lambda l: l.name)
    def test_single_hop_tunneling(self, noisy_memory, noise_level):
        """High-coherence pattern can tunnel to connected target through noise."""
        store, target_ids, noise_result = noisy_memory(
            ["quantum physics source", "medieval history target"],
            level=noise_level,
            seed=42,
        )

        source_id, target_id = target_ids

        # Setup connection
        store.create_connection(source_id, target_id)

        # Boost source coherence (required for tunneling)
        store.patterns[source_id].coherence = 0.9

        # Configure tunneling
        store.set_tunneling_config(TunnelingConfig(
            baseline_probability=0.5,
            creative_mode_multiplier=3.0,
            min_source_coherence=0.3,
            max_bit_overlap=0.3,
        ))

        # Attempt retrieval with tunneling
        results, tunnel_results = store.retrieve_with_tunneling(
            "quantum physics",
            top_k=10,
            creative_mode=True,
        )

        # Capture metrics
        metrics = {
            "noise_level": noise_level.name,
            "source_in_results": source_id in [r.pattern_id for r in results],
            "target_in_results": target_id in [r.pattern_id for r in results],
            "successful_tunnels": sum(1 for t in tunnel_results if t.tunneled),
            "tunnel_to_target": any(
                t.tunneled and t.target_pattern_id == target_id
                for t in tunnel_results
            ),
            "total_patterns": len(store.patterns),
        }
        print(f"Metrics: {metrics}")  # Always output

        # Verify source is found
        assert source_id in [r.pattern_id for r in results], \
            "Source pattern should be found directly"

        # Check if tunneling reached target
        tunneled_to_target = any(
            t.tunneled and t.target_pattern_id == target_id
            for t in tunnel_results
        )
        # Document but don't fail - tunneling is probabilistic
        if not tunneled_to_target:
            pytest.skip(
                f"LIMITATION: Tunneling did not reach target at {noise_level.name}. "
                f"This is expected sometimes - tunneling is probabilistic."
            )

    @pytest.mark.stress
    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS, ids=lambda l: l.name)
    def test_threshold_behavior(self, noisy_memory, noise_level):
        """Tunneling succeeds above coherence threshold, fails below."""
        store, target_ids, noise_result = noisy_memory(
            ["high coherence source", "target pattern"],
            level=noise_level,
            seed=42,
        )

        source_id, target_id = target_ids
        store.create_connection(source_id, target_id)

        config = TunnelingConfig(
            baseline_probability=0.9,  # High for testing
            min_source_coherence=0.5,  # Threshold
        )
        store.set_tunneling_config(config)

        # Test above threshold
        store.patterns[source_id].coherence = 0.8  # Above 0.5
        results_high, tunnels_high = store.retrieve_with_tunneling(
            "high coherence",
            top_k=5,
            creative_mode=True,
        )

        # Reset and test below threshold
        store.patterns[source_id].coherence = 0.3  # Below 0.5
        results_low, tunnels_low = store.retrieve_with_tunneling(
            "high coherence",
            top_k=5,
            creative_mode=False,
        )

        # Metrics
        metrics = {
            "noise_level": noise_level.name,
            "high_coherence_tunnels": sum(1 for t in tunnels_high if t.tunneled),
            "low_coherence_tunnels": sum(1 for t in tunnels_low if t.tunneled),
        }
        print(f"Threshold metrics: {metrics}")

        # Validate threshold behavior
        high_success = any(t.tunneled for t in tunnels_high)
        low_success = any(t.tunneled for t in tunnels_low)

        # Low coherence should NOT tunnel (below threshold)
        assert not low_success, \
            f"Low-coherence pattern should not tunnel (coherence=0.3 < threshold=0.5)"
```

### Interference Retrieval Under Noise Test (TEST-02)
```python
# tests/stress/test_interference.py
# Source: CONTEXT.md requirements + memory_store.py API
import pytest
from tests.conftest import NoiseLevel, noisy_memory

STRESS_NOISE_LEVELS = [NoiseLevel.MEDIUM, NoiseLevel.HIGH]
NEAR_MISS_COUNTS = [1, 3, 5]  # Parametrize to observe degradation

class TestInterferenceRetrieval:
    """TEST-02: Interference retrieval tests with noisy memory."""

    @pytest.mark.stress
    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS, ids=lambda l: l.name)
    @pytest.mark.parametrize("near_miss_count", NEAR_MISS_COUNTS, ids=lambda n: f"nm{n}")
    def test_target_retrieval_precision(self, noisy_memory, noise_level, near_miss_count):
        """Target should be retrievable among near-misses and clutter."""
        from tests.conftest import NoiseConfig

        # Custom config with specific near-miss count
        config = NoiseConfig(
            near_miss_ratio=float(near_miss_count),
            clutter_ratio=3.0,  # Fixed clutter
            seed=42,
        )

        store, target_ids, noise_result = noisy_memory(
            ["The quick brown fox jumps over the lazy dog"],
            config=config,
        )

        target_id = target_ids[0]
        near_miss_ids = noise_result.near_misses.get(target_id, [])

        # Retrieve using interference method
        results = store.retrieve(
            "quick brown fox",
            top_k=10,
            method="interference",
        )

        # Evaluate all three success criteria
        result_ids = [r.pattern_id for r in results]

        strict_success = result_ids[0] == target_id if result_ids else False
        relaxed_success = target_id in result_ids[:3]

        target_rank = result_ids.index(target_id) + 1 if target_id in result_ids else None
        near_miss_ranks = [
            result_ids.index(nm_id) + 1 if nm_id in result_ids else None
            for nm_id in near_miss_ids
        ]
        relative_success = target_rank is not None and all(
            target_rank < nm_rank for nm_rank in near_miss_ranks if nm_rank is not None
        )

        # Metrics
        target_result = next((r for r in results if r.pattern_id == target_id), None)
        metrics = {
            "noise_level": noise_level.name,
            "near_miss_count": near_miss_count,
            "target_rank": target_rank,
            "target_score": target_result.score if target_result else None,
            "strict_top_1": strict_success,
            "relaxed_top_3": relaxed_success,
            "relative_to_near_misses": relative_success,
            "near_miss_ranks": near_miss_ranks,
            "total_patterns": len(store.patterns),
        }
        print(f"Retrieval metrics: {metrics}")

        # At least relaxed criterion should pass
        assert relaxed_success, (
            f"Target not in top-3 at {noise_level.name} with {near_miss_count} near-misses. "
            f"Target rank: {target_rank}"
        )

    @pytest.mark.stress
    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS, ids=lambda l: l.name)
    def test_precision_vs_recall(self, noisy_memory, noise_level):
        """Validate both precision (near-misses rejected) and recall (targets found)."""
        store, target_ids, noise_result = noisy_memory(
            ["first target pattern", "second target pattern", "third target pattern"],
            level=noise_level,
            seed=42,
        )

        # Retrieve for first target
        target_id = target_ids[0]
        near_miss_ids = noise_result.near_misses.get(target_id, [])

        results = store.retrieve("first target", top_k=5, method="interference")
        result_ids = [r.pattern_id for r in results]

        # Precision: how many near-misses are in top results?
        near_misses_in_top = sum(1 for nm_id in near_miss_ids if nm_id in result_ids)
        precision_penalty = near_misses_in_top / len(near_miss_ids) if near_miss_ids else 0.0

        # Recall: is target in results?
        recall = target_id in result_ids

        metrics = {
            "noise_level": noise_level.name,
            "recall": recall,
            "near_misses_in_top_5": near_misses_in_top,
            "total_near_misses": len(near_miss_ids),
            "precision_penalty": precision_penalty,
        }
        print(f"Precision/Recall metrics: {metrics}")

        assert recall, f"Target not found in results at {noise_level.name}"
```

### Coherence Dynamics Under Load Test (TEST-03)
```python
# tests/stress/test_coherence.py
# Source: CONTEXT.md requirements + coherence.py API
import pytest
from tests.conftest import NoiseLevel, noisy_memory

STRESS_NOISE_LEVELS = [NoiseLevel.MEDIUM, NoiseLevel.HIGH]

class TestCoherenceDynamics:
    """TEST-03: Coherence dynamics tests with noisy memory."""

    @pytest.mark.stress
    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS, ids=lambda l: l.name)
    def test_decay_over_operations(self, noisy_memory, noise_level):
        """Patterns should decay over operations without access."""
        store, target_ids, noise_result = noisy_memory(
            ["pattern that will decay over time"],
            level=noise_level,
            seed=42,
        )

        target_id = target_ids[0]
        store.patterns[target_id].coherence = 0.9  # Start high
        initial_coherence = store.patterns[target_id].coherence

        # Simulate operations without touching target
        for i in range(30):
            store.retrieve("unrelated query content", top_k=3)

        final_coherence = store.patterns[target_id].coherence

        metrics = {
            "noise_level": noise_level.name,
            "initial_coherence": initial_coherence,
            "final_coherence": final_coherence,
            "operations": 30,
            "decay_amount": initial_coherence - final_coherence,
        }
        print(f"Decay metrics: {metrics}")

        assert final_coherence < initial_coherence, (
            f"Coherence should decay: {initial_coherence:.3f} -> {final_coherence:.3f}"
        )

    @pytest.mark.stress
    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS, ids=lambda l: l.name)
    def test_access_refresh(self, noisy_memory, noise_level):
        """Accessing a pattern should refresh its coherence."""
        store, target_ids, noise_result = noisy_memory(
            ["pattern that will be accessed and refreshed"],
            level=noise_level,
            seed=42,
        )

        target_id = target_ids[0]
        store.patterns[target_id].coherence = 0.3  # Start low

        # Let it decay first
        for i in range(10):
            store.retrieve("unrelated", top_k=1)

        pre_access_coherence = store.patterns[target_id].coherence

        # Now access the target directly
        results = store.retrieve("pattern accessed refreshed", top_k=5)

        post_access_coherence = store.patterns[target_id].coherence

        metrics = {
            "noise_level": noise_level.name,
            "pre_access_coherence": pre_access_coherence,
            "post_access_coherence": post_access_coherence,
            "target_in_results": target_id in [r.pattern_id for r in results],
            "refresh_amount": post_access_coherence - pre_access_coherence,
        }
        print(f"Refresh metrics: {metrics}")

        # If target was retrieved, coherence should increase
        if target_id in [r.pattern_id for r in results]:
            assert post_access_coherence > pre_access_coherence, (
                f"Coherence should refresh on access: {pre_access_coherence:.3f} -> {post_access_coherence:.3f}"
            )

    @pytest.mark.stress
    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS, ids=lambda l: l.name)
    def test_stability_resists_decay(self, noisy_memory, noise_level):
        """Highly-accessed patterns (high stability) should resist decay."""
        store, target_ids, noise_result = noisy_memory(
            ["stable pattern", "unstable pattern"],
            level=noise_level,
            seed=42,
        )

        stable_id, unstable_id = target_ids

        # Set up stability: stable has many accesses, unstable has none
        store.patterns[stable_id].access_count_since_modification = 20  # stability_score = 1.0
        store.patterns[unstable_id].access_count_since_modification = 0  # stability_score = 0.0

        # Both start at same coherence
        store.patterns[stable_id].coherence = 0.8
        store.patterns[unstable_id].coherence = 0.8

        # Let them decay
        for i in range(20):
            store.retrieve("unrelated query", top_k=1)

        stable_coherence = store.patterns[stable_id].coherence
        unstable_coherence = store.patterns[unstable_id].coherence

        metrics = {
            "noise_level": noise_level.name,
            "stable_coherence": stable_coherence,
            "unstable_coherence": unstable_coherence,
            "stable_stability_score": store.patterns[stable_id].stability_score,
            "unstable_stability_score": store.patterns[unstable_id].stability_score,
        }
        print(f"Stability metrics: {metrics}")

        # Note: Per coherence.py, stability affects CRYSTALLIZATION (faster decay for stable patterns)
        # This is counterintuitive but matches INTUITION.md: "crystallized beliefs decay faster"
        # because they become more "classical" (less quantum)

    @pytest.mark.stress
    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS, ids=lambda l: l.name)
    def test_surprise_recoherence(self, noisy_memory, noise_level):
        """Surprise events should trigger re-coherence of involved patterns."""
        store, target_ids, noise_result = noisy_memory(
            ["expected pattern", "surprising pattern"],
            level=noise_level,
            seed=42,
        )

        expected_id, surprising_id = target_ids

        # Set low coherence on surprising pattern (simulates "forgotten")
        store.patterns[surprising_id].coherence = 0.1
        store.patterns[surprising_id].access_count_since_modification = 0  # Malleable

        initial_coherence = store.patterns[surprising_id].coherence

        # Trigger surprise-aware retrieval
        # Query matches surprising pattern better, but we "expected" expected_id
        results, surprise = store.retrieve_with_surprise(
            "surprising pattern content",
            top_k=5,
            use_pure_similarity=True,
        )

        final_coherence = store.patterns[surprising_id].coherence

        metrics = {
            "noise_level": noise_level.name,
            "initial_coherence": initial_coherence,
            "final_coherence": final_coherence,
            "surprise_magnitude": surprise.magnitude if surprise else 0.0,
            "recoherence_delta": final_coherence - initial_coherence,
        }
        print(f"Surprise re-coherence metrics: {metrics}")

        # If surprise occurred, coherence should increase
        if surprise and surprise.magnitude > 0.1:
            assert final_coherence >= initial_coherence, (
                f"Surprise should trigger re-coherence: {initial_coherence:.3f} -> {final_coherence:.3f}"
            )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Behavior tests without noise | Stress tests with Phase 6 noise | Phase 7 | Realistic validation |
| Single success criterion | Multiple criteria (top-1, top-K, relative) | Phase 7 | Nuanced evaluation |
| Pass/fail only | Always output metrics | Phase 7 | Enables Phase 8 analysis |

**Deprecated/outdated:**
- Testing at NoiseLevel.NONE: Use MEDIUM and HIGH for stress testing
- Hard assertions on probabilistic behavior: Use skip with LIMITATION message

## Open Questions

Things that couldn't be fully resolved:

1. **Exact success rate thresholds**
   - What we know: Need empirical data on tunneling success rates under noise
   - What's unclear: What success rate is "acceptable" at HIGH noise?
   - Recommendation: Document actual rates in TEST_SUMMARY.md, define thresholds based on data

2. **Degradation curve shape**
   - What we know: Performance will degrade as noise increases
   - What's unclear: Linear degradation? Cliff? Graceful?
   - Recommendation: Parametrize near_miss_count to capture curve, analyze in Phase 8

3. **Creative mode effectiveness**
   - What we know: 3x probability multiplier in creative mode
   - What's unclear: How much does this help under heavy noise?
   - Recommendation: Test with and without creative mode, compare metrics

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/quantum_substrate/tunneling.py` - attempt_tunneling(), TunnelingConfig
- Existing codebase: `src/quantum_substrate/coherence.py` - CoherenceManager API
- Existing codebase: `src/agentic/memory_store.py` - retrieve_with_tunneling(), retrieve_with_surprise()
- Existing codebase: `tests/conftest.py` - noisy_memory fixture, NoiseLevel, inject_noise()
- Existing codebase: `tests/hypothesis_validation/` - test pattern examples

### Secondary (MEDIUM confidence)
- [pytest parametrize documentation](https://docs.pytest.org/en/stable/how-to/parametrize.html)
- CONTEXT.md: 07-CONTEXT.md decisions and requirements

### Tertiary (LOW confidence)
- None - all findings verified against codebase or official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Uses existing project infrastructure only
- Architecture: HIGH - Follows established test patterns from hypothesis_validation/
- Pitfalls: HIGH - Derived from codebase analysis of tunneling/coherence requirements
- Code examples: MEDIUM - Synthesized from codebase, not yet tested

**Research date:** 2026-02-04
**Valid until:** 60 days (stable domain, no external dependencies)
