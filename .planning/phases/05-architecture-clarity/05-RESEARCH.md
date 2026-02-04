# Phase 5: Architecture Clarity - Research

**Researched:** 2026-02-04
**Domain:** Python module architecture, statistical hypothesis testing, LLM-independence patterns
**Confidence:** HIGH

## Summary

This phase requires three distinct but related efforts: (1) refactoring the substrate layer to remove agentic dependencies, enabling true LLM-independence; (2) documenting and enforcing architectural boundaries between `quantum_substrate/` and `agentic/`; and (3) designing hypothesis validation tests that can genuinely validate OR invalidate the quantum memory theory.

The current codebase has a significant architectural issue: `quantum_substrate/` modules (`coherence.py`, `surprise.py`, `tunneling.py`) have TYPE_CHECKING imports from `agentic.evolving_pattern`. This creates a circular conceptual dependency where the "primitive" layer references the "semantic" layer. The fix involves defining protocol/interface types in substrate that `EvolvingPattern` implements, inverting the dependency.

For hypothesis testing, the research identifies scipy.stats as the standard library for statistical comparisons, with specific focus on Welch's t-test (`ttest_ind` with `equal_var=False`) and effect size metrics (Cohen's d via pingouin). The 8 INTUITION.md behaviors map naturally to testable predictions.

**Primary recommendation:** Define a `SubstratePattern` protocol in substrate that specifies required attributes (`bits`, `coherence`, `embeddedness`, etc.) which agentic layer patterns implement, enabling substrate to be truly independent.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.0.0 | Tensor operations, math | Already in use, GPU-ready |
| pytest | >=7.0.0 | Test framework | Already in use, fixtures support |
| typing/Protocol | stdlib | Interface definitions | No runtime cost, IDE support |

### Supporting (for hypothesis tests)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.stats | >=1.10 | Statistical tests | Comparing retrieval methods |
| pingouin | >=0.5 | Effect size (Cohen's d) | Measuring magnitude of advantage |
| numpy | >=1.20 | Array operations | Statistical computations |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.stats | statsmodels | statsmodels heavier but more features |
| Protocol types | ABC | ABC requires inheritance; Protocol is structural |
| pingouin | manual Cohen's d | pingouin handles edge cases properly |

**Installation:**
```bash
# Add to pyproject.toml [project.optional-dependencies]
pip install scipy pingouin
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── quantum_substrate/           # Primitive layer - NO LLM dependencies
│   ├── __init__.py              # Public API exports
│   ├── protocols.py             # NEW: SubstratePattern protocol
│   ├── patterns.py              # ComplexSparsePattern
│   ├── binding.py               # HRR operations
│   ├── interference.py          # Phase-aware retrieval
│   ├── coherence.py             # Decay/refresh dynamics
│   ├── surprise.py              # Surprise detection
│   ├── tunneling.py             # Creative retrieval
│   └── criticality.py           # System-wide tuning
│
├── agentic/                     # Semantic layer - builds on substrate
│   ├── __init__.py              # Public API exports
│   ├── evolving_pattern.py      # Implements SubstratePattern
│   ├── text_encoder.py          # Hash-based encoding
│   ├── llm_interface.py         # LLM reranking abstraction
│   ├── coactivation.py          # Learning dynamics
│   ├── memory_store.py          # High-level store
│   └── evaluation.py            # LLM evaluation
│
└── tests/
    ├── substrate/               # Tests for substrate alone
    │   └── test_*_no_llm.py     # Verify LLM-independence
    ├── agentic/                 # Tests for agentic layer
    └── hypothesis_validation/   # NEW: Quantum memory hypothesis tests
        ├── test_interference_beats_baselines.py
        ├── test_coherence_dynamics.py
        └── test_intuition_behaviors.py
```

### Pattern 1: Protocol-Based Dependency Inversion
**What:** Define protocols in substrate that agentic types implement
**When to use:** When lower layer needs to operate on higher layer types
**Example:**
```python
# src/quantum_substrate/protocols.py
from typing import Protocol, Set, Dict

class SubstratePattern(Protocol):
    """Protocol for patterns that substrate operations can work with.

    Any class implementing these attributes/methods can be used with
    coherence, surprise, tunneling without import dependency.
    """
    @property
    def bits(self) -> Set[int]: ...

    @property
    def original_bits(self) -> frozenset[int]: ...

    @property
    def phases(self) -> Dict[int, float]: ...

    @property
    def coherence(self) -> float: ...

    @coherence.setter
    def coherence(self, value: float) -> None: ...

    @property
    def last_access_tick(self) -> int: ...

    @last_access_tick.setter
    def last_access_tick(self, value: int) -> None: ...

    @property
    def embeddedness(self) -> float: ...

    @property
    def stability_score(self) -> float: ...

    def record_access(self) -> None: ...
```

### Pattern 2: Statistical Test Structure
**What:** Paired comparison of quantum vs baseline methods
**When to use:** Hypothesis validation tests
**Example:**
```python
# tests/hypothesis_validation/test_interference_beats_baselines.py
import pytest
from scipy.stats import ttest_ind
import pingouin as pg

def test_interference_beats_jaccard_statistically():
    """Interference retrieval outperforms Jaccard on precision.

    Hypothesis: Phase-aware interference finds more relevant results
    than bit-overlap-only Jaccard similarity.

    Pass criteria:
    - p < 0.05 (statistically significant)
    - Cohen's d > 0.3 (small-to-medium effect size)
    """
    # Generate N independent test scenarios
    N_TRIALS = 100
    interference_scores = []
    jaccard_scores = []

    for seed in range(N_TRIALS):
        # ... setup scenario with known ground truth ...
        score_int = run_interference_retrieval(scenario)
        score_jac = run_jaccard_retrieval(scenario)
        interference_scores.append(score_int)
        jaccard_scores.append(score_jac)

    # Statistical comparison
    result = ttest_ind(interference_scores, jaccard_scores,
                       equal_var=False, alternative='greater')

    # Effect size
    d = pg.compute_effsize(interference_scores, jaccard_scores,
                           paired=False, eftype='cohen')

    # Pass/fail criteria
    assert result.pvalue < 0.05, f"Not significant: p={result.pvalue}"
    assert d > 0.3, f"Effect too small: d={d}"
```

### Pattern 3: Behavior Test with Invalidation Criteria
**What:** Tests that can FAIL and document what that means
**When to use:** INTUITION.md behavior tests
**Example:**
```python
def test_generalization_emerges():
    """Repeated exposure creates shared concept representation.

    THEORY PREDICTION:
    - Exposing system to "cats are furry", "dogs are furry", "rabbits are furry"
    - Should create emergent representation where "furry" activates all three

    PASS CRITERIA:
    - Query "furry" retrieves all three above baseline probability
    - The quantum method beats random retrieval significantly

    INVALIDATION:
    - If quantum method equals or loses to random baseline, mark as limitation
    - Document what substrate capability would enable this behavior
    """
    # ... test implementation ...

    if not quantum_beats_baseline:
        pytest.skip(
            "LIMITATION: Generalization not observed with current substrate. "
            "Required capability: [document what's missing]"
        )
```

### Anti-Patterns to Avoid
- **Circular imports:** Never have substrate import from agentic at runtime
- **Force-passing tests:** Never adjust thresholds to make tests pass
- **Single-run comparisons:** Always use multiple trials with statistics
- **Confirmation bias:** Design tests that CAN fail, not just succeed

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Statistical significance | Manual p-value calc | `scipy.stats.ttest_ind` | Handles edge cases, degrees of freedom |
| Effect size | `(m1-m2)/sd` formula | `pingouin.compute_effsize` | Handles pooled SD, edge cases |
| Protocol types | Manual ABC | `typing.Protocol` | Structural typing, no inheritance needed |
| Random sampling | Manual seed handling | `random.Random(seed)` | Proper state isolation |

**Key insight:** Statistical testing has many subtle pitfalls (multiple comparisons, effect size vs significance, etc.). Use established libraries.

## Common Pitfalls

### Pitfall 1: TYPE_CHECKING Imports Create False Separation
**What goes wrong:** Using `if TYPE_CHECKING: from agentic import X` makes code look independent but still creates conceptual coupling
**Why it happens:** Quick fix for type hints without restructuring
**How to avoid:** Define protocols in the lower layer that higher layer implements
**Warning signs:** Any `from agentic` in substrate code, even under TYPE_CHECKING

### Pitfall 2: Statistical Significance Without Effect Size
**What goes wrong:** p < 0.05 but difference is trivially small (1% improvement)
**Why it happens:** Large sample sizes make tiny differences "significant"
**How to avoid:** Always report and threshold on effect size (Cohen's d)
**Warning signs:** Claiming success based on p-value alone

### Pitfall 3: Non-Reproducible Statistical Tests
**What goes wrong:** Tests pass/fail randomly depending on RNG state
**Why it happens:** Not seeding random generators properly
**How to avoid:** Use fixtures that provide seeded RNG; run sufficient trials
**Warning signs:** Tests that flake in CI, different results on re-run

### Pitfall 4: Testing Implementation Instead of Behavior
**What goes wrong:** Test passes because it tests the code path, not the theory
**Why it happens:** Testing "interference_retrieval returns list" vs "interference beats Jaccard"
**How to avoid:** Frame tests as theory predictions with clear pass/fail criteria
**Warning signs:** Tests that never fail even when algorithm is broken

### Pitfall 5: Forgetting the "Invalidation" Part
**What goes wrong:** All tests designed to pass; no way to disprove theory
**Why it happens:** Natural bias toward confirmation
**How to avoid:** Explicitly define what result would invalidate the hypothesis
**Warning signs:** No tests with `if not condition: pytest.skip("LIMITATION: ...")`

## Code Examples

Verified patterns for this phase:

### SubstratePattern Protocol Definition
```python
# src/quantum_substrate/protocols.py
from __future__ import annotations
from typing import Protocol, Set, Dict, FrozenSet, runtime_checkable

@runtime_checkable
class SubstratePattern(Protocol):
    """Protocol for patterns usable by substrate operations.

    Substrate modules (coherence, surprise, tunneling) operate on any
    object implementing this protocol, without importing agentic layer.
    """
    dim: int

    @property
    def bits(self) -> Set[int]:
        """Current active bit indices (original + acquired)."""
        ...

    @property
    def original_bits(self) -> FrozenSet[int]:
        """Frozen bits from initial encoding."""
        ...

    @property
    def phases(self) -> Dict[int, float]:
        """Phase (0 to 2pi) at each active bit."""
        ...

    @property
    def coherence(self) -> float:
        """Current coherence value (0-1)."""
        ...

    @coherence.setter
    def coherence(self, value: float) -> None:
        ...

    @property
    def last_access_tick(self) -> int:
        """Global tick when pattern was last accessed."""
        ...

    @last_access_tick.setter
    def last_access_tick(self, value: int) -> None:
        ...

    @property
    def embeddedness(self) -> float:
        """Structural embeddedness based on connections."""
        ...

    @property
    def stability_score(self) -> float:
        """Stability for crystallization dynamics."""
        ...

    def record_access(self) -> None:
        """Record an access without modification."""
        ...
```

### Welch's t-test for Method Comparison
```python
# Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
from scipy.stats import ttest_ind

def compare_methods(method_a_scores: list[float],
                   method_b_scores: list[float],
                   significance_level: float = 0.05) -> dict:
    """Compare two methods with Welch's t-test.

    Args:
        method_a_scores: Scores from method A across trials
        method_b_scores: Scores from method B across trials
        significance_level: Alpha threshold for significance

    Returns:
        Dict with t-statistic, p-value, and whether A > B significantly
    """
    result = ttest_ind(
        method_a_scores,
        method_b_scores,
        equal_var=False,  # Welch's t-test, no equal variance assumption
        alternative='greater'  # Test if A > B
    )

    return {
        'statistic': result.statistic,
        'pvalue': result.pvalue,
        'significant': result.pvalue < significance_level,
        'df': result.df,
    }
```

### Effect Size Calculation
```python
# Source: https://pingouin-stats.org/build/html/generated/pingouin.compute_effsize.html
import pingouin as pg

def compute_effect_size(group_a: list[float],
                        group_b: list[float]) -> dict:
    """Compute Cohen's d effect size between groups.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    d = pg.compute_effsize(group_a, group_b, paired=False, eftype='cohen')

    if abs(d) < 0.2:
        interpretation = 'negligible'
    elif abs(d) < 0.5:
        interpretation = 'small'
    elif abs(d) < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'

    return {
        'cohens_d': d,
        'interpretation': interpretation,
    }
```

### INTUITION.md Behavior Test Template
```python
# tests/hypothesis_validation/test_intuition_behaviors.py
import pytest
from typing import Optional

class TestGeneralization:
    """Test: Repeated exposure creates shared concept representation.

    INTUITION.md: "Generalization: repeated exposure -> shared concept with attributes"
    """

    def test_generalization_emerges_with_quantum_retrieval(self, store, rng):
        """Quantum retrieval shows generalization better than baseline."""
        # Setup: Expose to multiple instances sharing an attribute
        store.store("cats are furry animals")
        store.store("dogs are furry pets")
        store.store("rabbits are furry creatures")

        # Test: Query the shared attribute
        quantum_results = store.retrieve("furry", method="interference")
        baseline_results = store.retrieve("furry", method="jaccard")

        # Measure: How many of the 3 are in top-3?
        quantum_hits = sum(1 for r in quantum_results[:3]
                          if 'furry' in r.text)
        baseline_hits = sum(1 for r in baseline_results[:3]
                           if 'furry' in r.text)

        # Pass criteria
        if quantum_hits > baseline_hits:
            assert True  # Quantum shows generalization
        elif quantum_hits == baseline_hits:
            pytest.skip(
                "LIMITATION: Generalization not differentiated from baseline. "
                "Substrate may need: explicit attribute binding or "
                "coactivation-based attribute sharing."
            )
        else:
            pytest.fail(
                f"INVALIDATION: Quantum ({quantum_hits}) worse than baseline "
                f"({baseline_hits}). Hypothesis weakened."
            )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ABC for interfaces | typing.Protocol | Python 3.8+ | Structural typing, no inheritance required |
| Manual t-tests | scipy.stats | Stable | Handles edge cases, multiple variants |
| p-value only | p-value + effect size | Best practice | Meaningful comparisons |

**Deprecated/outdated:**
- Using `if TYPE_CHECKING:` to import from higher layers: still creates conceptual coupling
- Testing with single trials: need multiple runs for statistical validity

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal trial count for statistical tests**
   - What we know: More trials = more power, but diminishing returns
   - What's unclear: How many trials needed for this specific domain
   - Recommendation: Start with N=100 trials, adjust based on observed variance

2. **Serendipity measurement**
   - What we know: Context says "precision AND serendipity"
   - What's unclear: How to objectively measure "discovers non-obvious valid connections"
   - Recommendation: Define serendipity as "result is relevant but has low direct similarity"

3. **Baseline calibration for INTUITION.md tests**
   - What we know: Each behavior should beat "non-quantum alternative"
   - What's unclear: What's the right baseline for each behavior?
   - Recommendation: Use Jaccard similarity as primary baseline; random retrieval as floor

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `src/quantum_substrate/*.py`, `src/agentic/*.py`
- [SciPy ttest_ind documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html) - statistical testing API
- [Pingouin compute_effsize](https://pingouin-stats.org/build/html/generated/pingouin.compute_effsize.html) - effect size calculation

### Secondary (MEDIUM confidence)
- [Python typing.Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol) - structural subtyping
- [The Hitchhiker's Guide to Python - Project Structure](https://docs.python-guide.org/writing/structure/) - module organization
- Web search: statistical hypothesis testing patterns in Python 2026

### Tertiary (LOW confidence)
- Web search results on quantum-inspired computing validation methodology (general patterns, not specific to this project)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - scipy/pingouin well-established, Protocol pattern standard
- Architecture: HIGH - dependency inversion is well-understood pattern
- Pitfalls: HIGH - observed in codebase (TYPE_CHECKING issue documented)
- Hypothesis testing methodology: MEDIUM - standard stats but domain-specific application

**Research date:** 2026-02-04
**Valid until:** 30 days (stable domain, patterns unlikely to change)
