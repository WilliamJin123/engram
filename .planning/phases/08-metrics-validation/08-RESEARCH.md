# Phase 8: Metrics & Validation - Research

**Researched:** 2026-02-04
**Domain:** Retrieval system evaluation, statistical validation, visualization
**Confidence:** HIGH

## Summary

Phase 8 focuses on quantifying how interference retrieval compares to baselines (cosine similarity, random, recency) and documenting degradation curves. This requires:

1. **Metrics design**: Using standard Information Retrieval metrics (MRR, Recall@K, target rank) to capture retrieval quality
2. **Statistical validation**: Using scipy.stats for significance testing (Mann-Whitney U or t-test) to demonstrate p < 0.05 across multiple trials
3. **Visualization**: Using matplotlib for degradation curves with error bands (fill_between + errorbar)
4. **pytest infrastructure**: Custom `--update-report` flag to generate reports on-demand

The existing codebase has baseline implementations in `tests/hypothesis_validation/baselines.py` (Jaccard, cosine sparse, random), noise generation in `tests/noise_generators.py`, and stress test infrastructure in `tests/stress/`. Phase 7 established the degradation patterns (linear: rank 1->6->9 for 1->3->5 near-misses).

**Primary recommendation:** Use MRR as the primary metric (single correct answer per query), supplement with Recall@3 for relaxed criterion, use Mann-Whitney U test for significance (non-parametric, handles non-normal distributions), and generate matplotlib plots with shaded confidence bands.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy | >=1.10.0 | Statistical tests (mannwhitneyu, ttest_ind) | Standard for scientific Python, already in pyproject.toml |
| matplotlib | >=3.5.0 | Degradation curve plotting | Industry standard for publication-quality figures |
| numpy | >=1.20.0 | Numerical operations for metrics | Already in pyproject.toml |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | - | Raw data export | Store trial results for reproducibility |
| pytest | >=7.0.0 | Test framework with custom flags | Already in pyproject.toml |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib | seaborn | Seaborn prettier defaults but matplotlib sufficient for this use case |
| scipy.stats | pingouin | pingouin is higher-level but scipy already available and well-documented |
| Mann-Whitney | Welch t-test | t-test if normality can be assumed; Mann-Whitney safer for unknown distributions |

**Installation:**
```bash
pip install matplotlib>=3.5.0
# scipy and numpy already in pyproject.toml dev dependencies
```

Add to pyproject.toml:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "scipy>=1.10.0",
    "pingouin>=0.5.0",
    "numpy>=1.20.0",
    "matplotlib>=3.5.0",  # Add this
]
```

## Architecture Patterns

### Recommended Project Structure
```
tests/
├── stress/
│   ├── conftest.py              # Existing: StressMetrics, fixtures
│   ├── test_interference.py     # Existing Phase 7 tests
│   ├── test_tunneling.py        # Existing Phase 7 tests
│   ├── test_coherence.py        # Existing Phase 7 tests
│   ├── metrics/                  # NEW: Phase 8 metrics module
│   │   ├── __init__.py
│   │   ├── retrieval_metrics.py  # MRR, Recall@K, rank extraction
│   │   ├── statistical.py        # p-value calculation, significance tests
│   │   └── visualization.py      # Degradation curve plotting
│   ├── test_baseline_comparison.py   # NEW: METR-01, METR-02 tests
│   ├── test_degradation_curves.py    # NEW: METR-03 tests
│   └── reports/                  # NEW: Generated reports directory
│       └── .gitkeep
├── conftest.py                   # Existing: NoiseConfig, inject_noise
└── hypothesis_validation/
    └── baselines.py              # Existing: cosine_retrieval, random_retrieval
```

### Pattern 1: Multi-Trial Test Structure
**What:** Run multiple trials per test, aggregate results, compute significance
**When to use:** For statistical validation requiring p < 0.05
**Example:**
```python
# Source: scipy.stats official docs
from scipy.stats import mannwhitneyu
import numpy as np

def test_interference_vs_cosine_baseline(noisy_memory, noise_level):
    """Test that interference retrieval outperforms cosine baseline."""
    n_trials = 30  # Sufficient for Mann-Whitney
    interference_ranks = []
    cosine_ranks = []

    for trial in range(n_trials):
        store, target_ids, noise_result = noisy_memory(
            ["target text"],
            level=noise_level,
            seed=trial * 100,  # Different seed per trial
        )
        target_id = target_ids[0]

        # Interference retrieval
        results = store.retrieve("target text", method="interference", top_k=20)
        int_rank = get_target_rank(results, target_id)
        interference_ranks.append(int_rank if int_rank else 21)  # 21 = not found

        # Cosine baseline
        cosine_results = cosine_retrieval(query_bits, store.patterns, top_k=20)
        cos_rank = get_target_rank_from_tuples(cosine_results, target_id)
        cosine_ranks.append(cos_rank if cos_rank else 21)

    # Mann-Whitney U test (alternative='less' means interference ranks lower/better)
    statistic, p_value = mannwhitneyu(
        interference_ranks,
        cosine_ranks,
        alternative='less'
    )

    # Document results (soft assertion per CONTEXT.md)
    metrics = {
        "interference_mean_rank": np.mean(interference_ranks),
        "cosine_mean_rank": np.mean(cosine_ranks),
        "p_value": p_value,
        "statistically_significant": p_value < 0.05,
    }
    print(f"METRICS: {json.dumps(metrics)}")
```

### Pattern 2: Degradation Curve Data Collection
**What:** Collect performance at multiple noise levels for curve plotting
**When to use:** For METR-03 fine-grained degradation analysis
**Example:**
```python
def collect_degradation_data(noise_levels: list[int], n_trials: int = 10):
    """Collect target rank across noise levels for degradation curve."""
    data = {
        "noise_level": [],
        "mean_rank": [],
        "std_rank": [],
        "recall_at_3": [],
    }

    for near_miss_count in noise_levels:
        ranks = []
        for trial in range(n_trials):
            config = NoiseConfig(
                near_miss_ratio=float(near_miss_count),
                clutter_ratio=3.0,
                seed=trial * 100,
            )
            # ... run retrieval, capture rank
            ranks.append(rank)

        data["noise_level"].append(near_miss_count)
        data["mean_rank"].append(np.mean(ranks))
        data["std_rank"].append(np.std(ranks))
        data["recall_at_3"].append(sum(1 for r in ranks if r and r <= 3) / len(ranks))

    return data
```

### Pattern 3: Matplotlib Degradation Curve
**What:** Plot with shaded confidence bands
**When to use:** For visualizing degradation with uncertainty
**Example:**
```python
# Source: matplotlib official docs - fill_between
import matplotlib.pyplot as plt
import numpy as np

def plot_degradation_curve(data: dict, output_path: str):
    """Plot degradation curve with shaded std band."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.array(data["noise_level"])
    y = np.array(data["mean_rank"])
    yerr = np.array(data["std_rank"])

    # Main line
    ax.plot(x, y, 'o-', label="Interference Retrieval", color='tab:blue')

    # Shaded confidence band (mean +/- std)
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color='tab:blue')

    ax.set_xlabel("Near-Miss Count")
    ax.set_ylabel("Target Rank (lower is better)")
    ax.set_title("Retrieval Degradation Under Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Pattern 4: pytest Custom Flag
**What:** `--update-report` flag for on-demand report generation
**When to use:** Per CONTEXT.md decision - reports not auto-generated every run
**Example:**
```python
# Source: pytest official docs - parser.addoption
# In tests/stress/conftest.py

def pytest_addoption(parser):
    parser.addoption(
        "--update-report",
        action="store_true",
        default=False,
        help="Generate/update metrics report (slow, runs many trials)",
    )

@pytest.fixture
def update_report(request):
    """Fixture to check if --update-report flag is set."""
    return request.config.getoption("--update-report")

# Usage in test:
def test_generate_report(update_report, noisy_memory):
    if not update_report:
        pytest.skip("Report generation requires --update-report flag")
    # ... expensive multi-trial tests
```

### Anti-Patterns to Avoid
- **Hardcoded trial counts without justification:** Document why N trials chosen (e.g., 30 for Mann-Whitney stability)
- **Auto-generating reports on every test run:** Expensive tests slow down development
- **Using t-test without checking normality:** Mann-Whitney is safer for unknown distributions
- **Failing tests on metrics:** Phase 8 is measurement - use soft assertions (document, don't fail)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Statistical significance | Custom p-value calculation | `scipy.stats.mannwhitneyu` | Handles ties, exact method, validated |
| Cosine similarity | Dense vector implementation | Existing `baselines.cosine_similarity_sparse` | Already optimized for sparse patterns |
| Confidence intervals | Manual std calculation | `numpy.std()` + matplotlib `fill_between` | Handles edge cases correctly |
| JSON serialization | Custom serializer | `json.dumps()` with numpy conversion | Use `.tolist()` for numpy arrays |
| Random baseline | Custom shuffle | Existing `baselines.random_retrieval` | Already seeded for reproducibility |

**Key insight:** scipy.stats and the existing baselines.py handle all the statistical and baseline comparison needs. Don't reimplement - integrate and extend.

## Common Pitfalls

### Pitfall 1: Assuming Normal Distribution for t-test
**What goes wrong:** Using t-test on rank data which is ordinal, not normally distributed
**Why it happens:** t-test is familiar and often the first choice
**How to avoid:** Use Mann-Whitney U test by default for rank comparisons
**Warning signs:** Rank data is bounded (1 to N), not continuous

### Pitfall 2: Insufficient Trial Count for Statistical Power
**What goes wrong:** p-value fluctuates wildly between runs with too few trials
**Why it happens:** Small sample sizes have high variance
**How to avoid:** Use at least 20-30 trials per condition for Mann-Whitney
**Warning signs:** Re-running tests gives very different p-values

### Pitfall 3: Not Handling "Not Found" Results
**What goes wrong:** Division by zero or undefined behavior when target not in results
**Why it happens:** High noise can cause target to fall outside top-K
**How to avoid:** Assign sentinel rank (e.g., K+1) for not-found, document the convention
**Warning signs:** NaN in metrics, crashes during aggregation

### Pitfall 4: Comparing Methods at Different Coherence States
**What goes wrong:** Interference retrieval uses coherence weighting; baseline doesn't
**Why it happens:** Methods have different semantics
**How to avoid:** Compare on fresh memory (all coherence=1.0) OR document the comparison context
**Warning signs:** Unfair advantage to one method due to coherence state

### Pitfall 5: Matplotlib Memory Leak in Loops
**What goes wrong:** Memory grows when generating many plots in a loop
**Why it happens:** Figures not properly closed
**How to avoid:** Always call `plt.close(fig)` after saving, or use `plt.close('all')`
**Warning signs:** Memory usage grows over test run

### Pitfall 6: Seed Reuse Across Trials
**What goes wrong:** Trials are not independent, statistics invalid
**Why it happens:** Same seed used for all trials
**How to avoid:** Use `seed = base_seed + trial_index` pattern
**Warning signs:** Identical results across "different" trials

## Code Examples

Verified patterns from official sources:

### Mean Reciprocal Rank (MRR) Calculation
```python
# Source: Information Retrieval textbooks (standard formula)
def compute_mrr(ranks: list[int | None]) -> float:
    """Compute Mean Reciprocal Rank.

    MRR = (1/Q) * sum(1/rank_i) for each query i

    Args:
        ranks: List of target ranks (1-indexed), None if not found

    Returns:
        MRR score in [0, 1]. Higher is better.
    """
    reciprocals = []
    for rank in ranks:
        if rank is not None:
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)  # Not found contributes 0
    return sum(reciprocals) / len(reciprocals) if reciprocals else 0.0
```

### Recall@K Calculation
```python
# Source: Pinecone IR evaluation guide
def compute_recall_at_k(ranks: list[int | None], k: int = 3) -> float:
    """Compute Recall@K (fraction of targets found in top-K).

    Args:
        ranks: List of target ranks (1-indexed), None if not found
        k: Cutoff for top-K

    Returns:
        Recall@K score in [0, 1]. Higher is better.
    """
    found_in_top_k = sum(1 for r in ranks if r is not None and r <= k)
    return found_in_top_k / len(ranks) if ranks else 0.0
```

### Mann-Whitney U Test for Method Comparison
```python
# Source: scipy.stats.mannwhitneyu official docs
from scipy.stats import mannwhitneyu

def compare_methods(
    ranks_a: list[int],
    ranks_b: list[int],
    method_a_name: str = "A",
    method_b_name: str = "B",
) -> dict:
    """Compare two retrieval methods using Mann-Whitney U test.

    Tests whether method A produces significantly lower (better) ranks than method B.

    Args:
        ranks_a: Ranks from method A (lower is better)
        ranks_b: Ranks from method B (lower is better)

    Returns:
        Dict with statistic, p_value, and interpretation
    """
    # alternative='less' tests if ranks_a < ranks_b (A is better)
    statistic, p_value = mannwhitneyu(ranks_a, ranks_b, alternative='less')

    return {
        "test": "Mann-Whitney U",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_0.05": p_value < 0.05,
        "interpretation": (
            f"{method_a_name} significantly better than {method_b_name}"
            if p_value < 0.05 else
            f"No significant difference between {method_a_name} and {method_b_name}"
        ),
    }
```

### Matplotlib Multi-Method Comparison Plot
```python
# Source: matplotlib official docs - fill_between, errorbar
import matplotlib.pyplot as plt
import numpy as np

def plot_method_comparison(
    noise_levels: list[int],
    methods_data: dict[str, dict],  # method_name -> {mean_rank, std_rank}
    output_path: str,
):
    """Plot degradation curves for multiple methods.

    Args:
        noise_levels: X-axis values (e.g., near-miss counts)
        methods_data: Dict mapping method name to {mean_rank: [], std_rank: []}
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    x = np.array(noise_levels)

    for i, (method_name, data) in enumerate(methods_data.items()):
        y = np.array(data["mean_rank"])
        yerr = np.array(data["std_rank"])
        color = colors[i % len(colors)]

        # Line with markers
        ax.plot(x, y, 'o-', label=method_name, color=color)
        # Shaded confidence band
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)

    ax.set_xlabel("Near-Miss Count")
    ax.set_ylabel("Target Rank (lower is better)")
    ax.set_title("Retrieval Method Comparison Under Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Optional: lower rank at top

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Prevent memory leak
```

### Recency Baseline Implementation
```python
# Pattern for recency baseline (per CONTEXT.md: include as third baseline)
def recency_retrieval(
    patterns: dict[str, "EvolvingPattern"],
    current_tick: int,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Rank patterns by recency (most recently accessed first).

    Args:
        patterns: Dict mapping pattern_id to EvolvingPattern
        current_tick: Current system tick for age calculation
        top_k: Number of results to return

    Returns:
        List of (pattern_id, recency_score) tuples, sorted by recency descending.
    """
    results = []
    for pattern_id, pattern in patterns.items():
        age = current_tick - pattern.last_access_tick
        # Recency score: newer patterns have higher scores
        recency_score = 1.0 / (1.0 + age)
        results.append((pattern_id, recency_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Precision/Recall only | MRR + Recall@K + NDCG | 2020+ | Rank-aware metrics standard for IR |
| t-test everywhere | Mann-Whitney for ranks | Always | Non-parametric safer for ordinal data |
| Manual bar plots | fill_between bands | matplotlib 2.0+ | Shows uncertainty more clearly |
| Hardcoded thresholds | Statistical significance (p-value) | Research standard | Reproducible, defensible claims |

**Deprecated/outdated:**
- Using precision/recall without rank consideration for retrieval systems
- Asserting "X is better" without statistical significance testing

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal trial count**
   - What we know: 20-30 trials sufficient for Mann-Whitney stability
   - What's unclear: Exact tradeoff between runtime and statistical power for this specific data
   - Recommendation: Start with 30 trials, adjust based on observed variance

2. **Cosine baseline: sparse vs dense**
   - What we know: CONTEXT.md says "Claude's discretion on sparse vs dense"
   - What's unclear: Which is the "fairer" comparison (both are valid)
   - Recommendation: Use sparse (existing implementation) as it matches pattern representation

3. **Fine-grained curve granularity**
   - What we know: CONTEXT.md wants "10+ data points"
   - What's unclear: Optimal spacing (linear vs logarithmic)
   - Recommendation: Linear spacing [0, 1, 2, 3, 4, 5, 7, 10, 15, 20] near-misses

## Sources

### Primary (HIGH confidence)
- [scipy.stats.mannwhitneyu official docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html) - Function signature, parameters, return values
- [scipy.stats.ttest_ind official docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html) - Welch's t-test alternative
- [matplotlib fill_between official demo](https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html) - Confidence band plotting
- [matplotlib errorbar official docs](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html) - Error bar parameters
- [pytest addoption official docs](https://docs.pytest.org/en/stable/example/simple.html) - Custom command-line flags

### Secondary (MEDIUM confidence)
- [Pinecone IR Evaluation Guide](https://www.pinecone.io/learn/offline-evaluation/) - MRR, Recall@K, Precision@K definitions
- [Weaviate Retrieval Metrics Blog](https://weaviate.io/blog/retrieval-evaluation-metrics) - Metric selection guidance
- [Evidently AI NDCG Explained](https://www.evidentlyai.com/ranking-metrics/ndcg-metric) - When to use NDCG vs MRR

### Tertiary (LOW confidence)
- WebSearch results for matplotlib best practices 2026 - General guidance

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in pyproject.toml or standard Python
- Architecture: HIGH - Builds on existing Phase 7 patterns
- Pitfalls: HIGH - Based on official docs and known statistical requirements
- Code examples: HIGH - Verified against official documentation

**Research date:** 2026-02-04
**Valid until:** 30 days (stable domain, libraries are mature)
