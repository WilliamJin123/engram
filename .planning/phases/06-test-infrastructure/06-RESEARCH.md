# Phase 6: Test Infrastructure - Research

**Researched:** 2026-02-04
**Domain:** Test infrastructure for noise-based stress testing of retrieval mechanisms
**Confidence:** HIGH

## Summary

This phase builds a noise generation toolkit for stress-testing the quantum-inspired memory retrieval system. The research focuses on implementing noise generators that create realistic interference patterns: semantic near-misses (patterns similar to targets but semantically unrelated) and random clutter (background noise with realistic sparsity distribution).

The codebase already has strong foundations for this work:
- `ComplexSparsePattern.random()` for seeded pattern generation
- `create_related_pattern()` for controlled overlap and phase relationships
- Established fixture patterns in `conftest.py` files (factory functions, seeded RNG)
- `EvolvingPattern` with coherence dynamics for realistic aging

**Primary recommendation:** Implement noise generation as a factory fixture in `tests/conftest.py` that returns a `NoiseContext` object containing the generated patterns and metadata, using the existing `torch.Generator` seeding pattern for reproducibility.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | >=7.0.0 | Test framework | Already in project, excellent fixture system |
| torch | >=2.0.0 | Pattern generation | Already used for `ComplexSparsePattern.random()` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses | stdlib | Configuration/result objects | NoiseConfig, NoiseLevel presets |
| enum | stdlib | Named noise level presets | NoiseLevel.LOW, MEDIUM, HIGH |
| typing | stdlib | Type hints for API clarity | All public interfaces |

### Not Needed
| Instead of | Why Not Needed |
|------------|----------------|
| factory_boy | Overkill - simple factory fixtures sufficient |
| pytest-factoryboy | Not needed - existing patterns work well |
| hypothesis | Different purpose (property testing, not noise generation) |

**Installation:**
```bash
# No new dependencies required - all in existing pyproject.toml
```

## Architecture Patterns

### Recommended Project Structure
```
tests/
    conftest.py                 # NoiseConfig, NoiseLevel, noisy_memory fixture
    noise_utils.py              # Near-miss generator, clutter generator (optional)
    quantum_substrate/
        ...existing tests...
    agentic/
        ...existing tests...
    hypothesis_validation/
        ...existing tests...
```

### Pattern 1: Factory Fixture with NoiseContext
**What:** Fixture returns a factory function that creates memories with configurable noise
**When to use:** Tests needing parameterized noise configurations
**Example:**
```python
# Source: Derived from existing conftest.py patterns + pytest docs
from dataclasses import dataclass
from enum import Enum
from typing import Callable
import torch

class NoiseLevel(Enum):
    """Named noise level presets."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class NoiseConfig:
    """Configuration for noise generation."""
    near_miss_ratio: float = 0.0  # Near-misses per target
    clutter_ratio: float = 0.0     # Clutter patterns per target
    near_miss_overlap: float = 0.4  # Bit overlap with target
    near_miss_phase_noise: float = 1.5  # Phase variance (radians)
    seed: int = 42

    @classmethod
    def from_preset(cls, level: NoiseLevel, seed: int = 42) -> "NoiseConfig":
        """Create config from named preset."""
        presets = {
            NoiseLevel.NONE: cls(seed=seed),
            NoiseLevel.LOW: cls(near_miss_ratio=0.5, clutter_ratio=1.0, seed=seed),
            NoiseLevel.MEDIUM: cls(near_miss_ratio=1.0, clutter_ratio=3.0, seed=seed),
            NoiseLevel.HIGH: cls(near_miss_ratio=2.0, clutter_ratio=5.0, seed=seed),
        }
        return presets[level]

@dataclass
class NoiseContext:
    """Result of noise generation - holds patterns and metadata."""
    targets: list[EvolvingPattern]
    near_misses: list[EvolvingPattern]  # Grouped by target
    clutter: list[EvolvingPattern]
    config: NoiseConfig
    generator: torch.Generator

@pytest.fixture
def noisy_memory() -> Callable[..., NoiseContext]:
    """Factory for creating noisy memory contexts."""
    def _create(
        targets: list[str],
        config: NoiseConfig | None = None,
        level: NoiseLevel = NoiseLevel.MEDIUM,
    ) -> NoiseContext:
        if config is None:
            config = NoiseConfig.from_preset(level)
        gen = torch.Generator().manual_seed(config.seed)
        # ... generate near-misses and clutter ...
        return NoiseContext(...)
    return _create
```

### Pattern 2: Near-Miss Generation with Bit Overlap Control
**What:** Generate patterns that share bits with target but have different phases
**When to use:** Testing retrieval discrimination between similar patterns
**Example:**
```python
# Source: Based on existing create_related_pattern() in interference.py
def generate_near_miss(
    target: EvolvingPattern,
    overlap_frac: float = 0.4,
    phase_noise: float = 1.5,  # ~pi/2 - moderate disruption
    generator: torch.Generator | None = None,
    coherence_range: tuple[float, float] = (0.2, 0.7),
) -> EvolvingPattern:
    """Create a near-miss pattern for a target.

    Near-miss shares bit overlap but has:
    - Different phase relationships (won't constructively interfere)
    - Aged coherence (simulates realistic memory)
    - Different text/metadata (semantically unrelated)
    """
    # Use target's bit positions partially
    target_bits = list(target.bits)
    k = len(target_bits)
    num_shared = int(overlap_frac * k)

    # Select random subset of target bits to share
    perm = torch.randperm(k, generator=generator)
    shared_bits = {target_bits[i] for i in perm[:num_shared].tolist()}

    # Generate new bits for non-shared positions
    available = [i for i in range(target.dim) if i not in target.bits]
    new_bits = set(
        available[i]
        for i in torch.randperm(len(available), generator=generator)[:k - num_shared].tolist()
    )

    # Create with perturbed phases and aged coherence
    all_bits = shared_bits | new_bits
    phases = {
        bit: (target.phases.get(bit, 0.0) +
              torch.randn(1, generator=generator).item() * phase_noise) % (2 * math.pi)
        for bit in all_bits
    }

    # Aged coherence (not fresh 1.0)
    coherence = coherence_range[0] + (
        torch.rand(1, generator=generator).item() *
        (coherence_range[1] - coherence_range[0])
    )

    return EvolvingPattern(
        dim=target.dim,
        bits=all_bits,
        original_bits=frozenset(all_bits),
        acquired_bits=set(),
        phases=phases,
        text=f"near_miss_{target.text[:20]}",  # Distinct text
        coherence=coherence,
        last_access_tick=random.randint(1, 50),  # Aged
        # ... other fields
    )
```

### Pattern 3: Clutter Generation with Sparsity Matching
**What:** Generate random clutter that matches real pattern statistics
**When to use:** Background noise that participates in coherence dynamics
**Example:**
```python
# Source: Based on ComplexSparsePattern.random() and TextEncoder patterns
def generate_clutter(
    dim: int,
    k: int,
    count: int,
    generator: torch.Generator,
    relatedness_distribution: tuple[float, float, float] = (0.5, 0.3, 0.2),  # distant, mid, close
    target_bits: set[int] | None = None,
) -> list[EvolvingPattern]:
    """Generate clutter patterns with varying relatedness to targets.

    Per CONTEXT.md: Clutter isn't purely unrelated - some patterns
    will be closer to targets than others, creating realistic interference.

    Args:
        dim: Pattern dimensionality
        k: Sparsity (active bits per pattern)
        count: Number of clutter patterns
        generator: Seeded RNG
        relatedness_distribution: (distant, mid-range, close) ratios
        target_bits: Combined bits from all targets (for relatedness control)
    """
    clutter = []

    # Distribute clutter across relatedness levels
    n_distant = int(count * relatedness_distribution[0])
    n_mid = int(count * relatedness_distribution[1])
    n_close = count - n_distant - n_mid

    for i in range(count):
        if i < n_distant:
            # Distant: minimal overlap with targets
            overlap_frac = 0.0 if target_bits else 0.1
        elif i < n_distant + n_mid:
            # Mid-range: moderate overlap
            overlap_frac = 0.2
        else:
            # Close: higher overlap (but still clutter)
            overlap_frac = 0.3

        pattern = _generate_single_clutter(
            dim, k, generator, overlap_frac, target_bits
        )
        clutter.append(pattern)

    return clutter
```

### Anti-Patterns to Avoid
- **Fresh coherence on all noise:** Near-misses and clutter should have varied, aged coherence to simulate realistic memory state
- **Purely random clutter:** Clutter should match sparsity distribution (k bits active) not random bit density
- **Unseeded generation:** ALL noise must be seeded for reproducibility - no optional seeding
- **Global state for RNG:** Use `torch.Generator` instances, not global torch.manual_seed()

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Related pattern creation | Custom bit manipulation | `create_related_pattern()` | Already handles overlap+phase correctly |
| Random indices | Manual random sampling | `torch.randperm(generator=gen)` | Proper seeding, efficient |
| Phase perturbation | Custom noise addition | `torch.randn(generator=gen) * noise` | Reproducible Gaussian noise |
| Circular mean of phases | Manual trig | `TextEncoder._circular_mean()` | Already implemented correctly |
| Pattern from text | Custom encoding | `EvolvingPattern.from_text()` | Uses TextEncoder, deterministic |

**Key insight:** The existing codebase has sophisticated pattern generation infrastructure. Near-miss generation should wrap `create_related_pattern()` or use its approach. Clutter generation should use `ComplexSparsePattern.random()` patterns converted to `EvolvingPattern`.

## Common Pitfalls

### Pitfall 1: Non-Reproducible Noise
**What goes wrong:** Tests flake because noise differs between runs
**Why it happens:** Using unseeded RNG or mixing global/local seeds
**How to avoid:**
- ALWAYS require seed in NoiseConfig (no defaults that use time-based seeds)
- Pass `torch.Generator` explicitly to all generation functions
- Store generator in NoiseContext for potential continuation
**Warning signs:** Tests pass locally but fail in CI, different results on re-run

### Pitfall 2: Uniform Noise Coherence
**What goes wrong:** All noise patterns have same coherence (e.g., all 1.0 or all 0.5)
**Why it happens:** Not simulating realistic memory aging
**How to avoid:**
- Near-misses: coherence sampled from range like (0.2, 0.7)
- Clutter: coherence sampled from broader range (0.1, 0.9)
- Set varied `last_access_tick` values
**Warning signs:** Coherence dynamics tests don't show realistic behavior

### Pitfall 3: Wrong Sparsity for Clutter
**What goes wrong:** Clutter has different bit density than real patterns
**Why it happens:** Using random bit vectors instead of matching k active bits
**How to avoid:**
- Clutter must have exactly k bits active (same as real patterns)
- Use same dim/k parameters as MemoryStore
**Warning signs:** Clutter dominates retrieval or has no interference effect

### Pitfall 4: Near-Misses That Are Actually Hits
**What goes wrong:** Near-misses are so similar they should be retrieved
**Why it happens:** Too high overlap fraction, too low phase noise
**How to avoid:**
- Keep overlap_frac <= 0.5 (typical: 0.3-0.4)
- Keep phase_noise >= 1.0 radians (~60 degrees disruption)
- Near-misses should rank LOWER than targets on interference retrieval
**Warning signs:** Near-misses consistently rank in top-3 with targets

### Pitfall 5: Clutter Without Coherence Dynamics
**What goes wrong:** Clutter is "inert" - doesn't participate in memory dynamics
**Why it happens:** Creating clutter as data only, not full EvolvingPattern
**How to avoid:**
- Clutter must be EvolvingPattern with coherence, last_access_tick, etc.
- Clutter should be stored in MemoryStore (not just pattern list)
**Warning signs:** Clutter doesn't decay, doesn't affect coherence calculations

## Code Examples

Verified patterns from official sources and codebase:

### Complete Noise Generation Fixture
```python
# Source: Synthesized from codebase patterns
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any
import math
import torch

from agentic.evolving_pattern import EvolvingPattern
from agentic.memory_store import MemoryStore


class NoiseLevel(Enum):
    """Named noise level presets."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class NoiseConfig:
    """Configuration for noise generation.

    Attributes:
        near_miss_ratio: Near-miss patterns per target (e.g., 1.0 = 1 per target)
        clutter_ratio: Clutter patterns per target
        near_miss_overlap: Fraction of bits shared with target (0-1)
        near_miss_phase_noise: Phase perturbation in radians
        coherence_range: (min, max) coherence for noise patterns
        seed: RNG seed (REQUIRED - always seeded)
    """
    near_miss_ratio: float = 0.0
    clutter_ratio: float = 0.0
    near_miss_overlap: float = 0.4
    near_miss_phase_noise: float = 1.5
    coherence_range: tuple[float, float] = (0.2, 0.7)
    seed: int = 42

    @classmethod
    def from_preset(cls, level: NoiseLevel, seed: int = 42) -> "NoiseConfig":
        """Create config from named preset."""
        presets = {
            NoiseLevel.NONE: cls(seed=seed),
            NoiseLevel.LOW: cls(
                near_miss_ratio=0.5,
                clutter_ratio=1.0,
                seed=seed
            ),
            NoiseLevel.MEDIUM: cls(
                near_miss_ratio=1.0,
                clutter_ratio=3.0,
                seed=seed
            ),
            NoiseLevel.HIGH: cls(
                near_miss_ratio=2.0,
                clutter_ratio=5.0,
                seed=seed
            ),
        }
        return presets[level]


@dataclass
class NoiseResult:
    """Result of noise injection."""
    near_misses: dict[str, list[str]]  # target_id -> list of near_miss_ids
    clutter_ids: list[str]
    config: NoiseConfig


def inject_noise(
    store: MemoryStore,
    target_ids: list[str],
    config: NoiseConfig,
) -> NoiseResult:
    """Inject noise patterns into an existing memory store.

    Creates near-miss and clutter patterns based on config,
    storing them in the memory store.

    Returns:
        NoiseResult with IDs of created noise patterns.
    """
    gen = torch.Generator().manual_seed(config.seed)

    near_misses: dict[str, list[str]] = {}
    clutter_ids: list[str] = []

    # Generate near-misses for each target
    for target_id in target_ids:
        target = store.get(target_id)
        if target is None:
            continue

        near_misses[target_id] = []
        num_near_misses = int(config.near_miss_ratio)
        if torch.rand(1, generator=gen).item() < (config.near_miss_ratio % 1):
            num_near_misses += 1

        for i in range(num_near_misses):
            nm_pattern = _create_near_miss(
                target, config, gen, i
            )
            nm_id = store.store(
                nm_pattern.text,
                metadata={"noise_type": "near_miss", "target": target_id}
            )
            # Adjust the stored pattern's coherence (store() sets to 1.0)
            store.patterns[nm_id].coherence = nm_pattern.coherence
            store.patterns[nm_id].last_access_tick = nm_pattern.last_access_tick
            near_misses[target_id].append(nm_id)

    # Generate clutter
    total_clutter = int(len(target_ids) * config.clutter_ratio)
    all_target_bits = set()
    for tid in target_ids:
        t = store.get(tid)
        if t:
            all_target_bits |= t.bits

    for i in range(total_clutter):
        clutter_pattern = _create_clutter(
            store.dim, store.k, config, gen, all_target_bits, i
        )
        c_id = store.store(
            clutter_pattern.text,
            metadata={"noise_type": "clutter"}
        )
        store.patterns[c_id].coherence = clutter_pattern.coherence
        store.patterns[c_id].last_access_tick = clutter_pattern.last_access_tick
        clutter_ids.append(c_id)

    return NoiseResult(
        near_misses=near_misses,
        clutter_ids=clutter_ids,
        config=config,
    )


def _create_near_miss(
    target: EvolvingPattern,
    config: NoiseConfig,
    gen: torch.Generator,
    index: int,
) -> EvolvingPattern:
    """Create a single near-miss pattern."""
    k = len(target.bits)
    num_shared = int(config.near_miss_overlap * k)

    target_bits = list(target.bits)
    perm = torch.randperm(k, generator=gen)
    shared_bits = {target_bits[i] for i in perm[:num_shared].tolist()}

    available = [i for i in range(target.dim) if i not in target.bits]
    new_bits = set(
        available[i]
        for i in torch.randperm(len(available), generator=gen)[:k - num_shared].tolist()
    )

    all_bits = shared_bits | new_bits

    phases = {}
    for bit in all_bits:
        base_phase = target.phases.get(bit, torch.rand(1, generator=gen).item() * 2 * math.pi)
        noise = torch.randn(1, generator=gen).item() * config.near_miss_phase_noise
        phases[bit] = (base_phase + noise) % (2 * math.pi)

    cmin, cmax = config.coherence_range
    coherence = cmin + torch.rand(1, generator=gen).item() * (cmax - cmin)

    return EvolvingPattern(
        dim=target.dim,
        bits=all_bits,
        original_bits=frozenset(all_bits),
        acquired_bits=set(),
        phases=phases,
        text=f"near_miss_{index}_of_{target.text[:30]}",
        coherence=coherence,
        last_access_tick=int(torch.randint(1, 50, (1,), generator=gen).item()),
        connection_count=0,
        last_modified_tick=0,
        access_count_since_modification=int(torch.randint(0, 5, (1,), generator=gen).item()),
    )


def _create_clutter(
    dim: int,
    k: int,
    config: NoiseConfig,
    gen: torch.Generator,
    target_bits: set[int],
    index: int,
) -> EvolvingPattern:
    """Create a single clutter pattern with varying relatedness."""
    # Determine relatedness level (distant/mid/close)
    r = torch.rand(1, generator=gen).item()
    if r < 0.5:  # 50% distant
        overlap_frac = 0.1
    elif r < 0.8:  # 30% mid-range
        overlap_frac = 0.2
    else:  # 20% closer
        overlap_frac = 0.3

    # Generate bits with controlled overlap to targets
    if target_bits:
        num_overlap = int(overlap_frac * k)
        target_list = list(target_bits)
        if len(target_list) >= num_overlap:
            perm = torch.randperm(len(target_list), generator=gen)
            overlapping = {target_list[i] for i in perm[:num_overlap].tolist()}
        else:
            overlapping = set(target_list)
            num_overlap = len(overlapping)
    else:
        overlapping = set()
        num_overlap = 0

    # Generate remaining bits from non-target positions
    available = [i for i in range(dim) if i not in target_bits]
    needed = k - num_overlap
    new_bits = set(
        available[i]
        for i in torch.randperm(len(available), generator=gen)[:needed].tolist()
    )

    all_bits = overlapping | new_bits

    # Random phases
    phases = {
        bit: torch.rand(1, generator=gen).item() * 2 * math.pi
        for bit in all_bits
    }

    # Broader coherence range for clutter
    coherence = 0.1 + torch.rand(1, generator=gen).item() * 0.8

    return EvolvingPattern(
        dim=dim,
        bits=all_bits,
        original_bits=frozenset(all_bits),
        acquired_bits=set(),
        phases=phases,
        text=f"clutter_{index}",
        coherence=coherence,
        last_access_tick=int(torch.randint(1, 100, (1,), generator=gen).item()),
        connection_count=0,
        last_modified_tick=0,
        access_count_since_modification=int(torch.randint(0, 10, (1,), generator=gen).item()),
    )
```

### Test Usage Example
```python
# Source: Pattern derived from existing hypothesis_validation tests
def test_retrieval_under_noise(memory_store_factory, statistical_comparator):
    """Test that retrieval remains accurate under various noise levels."""
    from tests.conftest import NoiseLevel, NoiseConfig, inject_noise

    scores_no_noise = []
    scores_high_noise = []

    for trial in range(50):
        # No noise baseline
        store = memory_store_factory(seed=trial)
        target_id = store.store("The quick brown fox jumps over the lazy dog")
        results = store.retrieve("quick fox", top_k=3)
        scores_no_noise.append(1.0 if results[0].pattern_id == target_id else 0.0)

        # High noise condition
        store = memory_store_factory(seed=trial)
        target_id = store.store("The quick brown fox jumps over the lazy dog")
        inject_noise(
            store,
            [target_id],
            NoiseConfig.from_preset(NoiseLevel.HIGH, seed=trial + 1000)
        )
        results = store.retrieve("quick fox", top_k=3)
        scores_high_noise.append(1.0 if results[0].pattern_id == target_id else 0.0)

    # Statistical comparison
    result = statistical_comparator(scores_no_noise, scores_high_noise)
    # Document but don't fail - we expect some degradation
    print(f"No noise: {sum(scores_no_noise)/len(scores_no_noise):.2%}")
    print(f"High noise: {sum(scores_high_noise)/len(scores_high_noise):.2%}")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Random patterns only | Near-miss + clutter | Phase 6 | Realistic stress testing |
| Ad-hoc noise in tests | Centralized fixture | Phase 6 | Reproducibility, consistency |
| Unseeded generation | Always seeded | Phase 6 | Reproducible test results |

**Deprecated/outdated:**
- Direct `torch.manual_seed()` calls: Use `torch.Generator` instances instead for test isolation
- pytest-randomly for noise: Not appropriate for deterministic noise generation (different purpose)

## Open Questions

Things that couldn't be fully resolved:

1. **Exact noise level ratios**
   - What we know: Need NONE/LOW/MEDIUM/HIGH presets with separate near-miss and clutter ratios
   - What's unclear: Optimal ratio values depend on retrieval degradation targets
   - Recommendation: Start with proposed values (LOW: 0.5/1.0, MEDIUM: 1.0/3.0, HIGH: 2.0/5.0), tune based on Phase 7 behavior tests

2. **Near-miss overlap percentage**
   - What we know: Must be < 0.5 to avoid being "too good"
   - What's unclear: Exact threshold depends on interference retrieval discrimination
   - Recommendation: Default 0.4, make configurable

3. **Continuous injection API**
   - What we know: CONTEXT.md requires "both setup-time clutter and continuous injection during test execution"
   - What's unclear: Exact API for continuous injection (callback? context manager? method?)
   - Recommendation: Implement as `inject_noise(store, targets, config)` function callable at any time, plus `NoisyMemoryContext` context manager for setup-time

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/quantum_substrate/patterns.py` - ComplexSparsePattern.random()
- Existing codebase: `src/quantum_substrate/interference.py` - create_related_pattern()
- Existing codebase: `tests/quantum_substrate/conftest.py` - fixture patterns
- Existing codebase: `tests/hypothesis_validation/conftest.py` - factory fixture pattern
- [pytest fixtures documentation](https://docs.pytest.org/en/stable/how-to/fixtures.html)

### Secondary (MEDIUM confidence)
- [Five Advanced Pytest Fixture Patterns](https://www.inspiredpython.com/article/five-advanced-pytest-fixture-patterns) - Factory patterns
- [PyTorch Reproducibility docs](https://docs.pytorch.org/docs/stable/notes/randomness.html) - Generator seeding
- [Fiddler AI Blog: Advanced Pytest Patterns](https://www.fiddler.ai/blog/advanced-pytest-patterns-harnessing-the-power-of-parametrization-and-factory-methods) - Parametrized fixtures

### Tertiary (LOW confidence)
- None - all findings verified against codebase or official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Uses existing project dependencies only
- Architecture: HIGH - Follows established codebase patterns
- Pitfalls: HIGH - Derived from codebase analysis and domain knowledge
- Code examples: MEDIUM - Synthesized from codebase, not yet tested

**Research date:** 2026-02-04
**Valid until:** 60 days (stable domain, no external dependencies)
