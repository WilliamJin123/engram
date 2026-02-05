# tests/stress/metrics/statistical.py
"""Statistical significance testing for method comparison.

Provides Mann-Whitney U test for comparing retrieval method rankings.
Mann-Whitney is non-parametric (doesn't assume normal distribution),
making it suitable for rank data which is ordinal.
"""

from __future__ import annotations

from scipy.stats import mannwhitneyu


def compare_methods(
    ranks_a: list[int],
    ranks_b: list[int],
    method_a_name: str = "A",
    method_b_name: str = "B",
) -> dict:
    """Compare two retrieval methods using Mann-Whitney U test.

    Tests whether method A produces significantly lower (better) ranks than method B.
    Uses alternative='less' because lower ranks are better in retrieval.

    Args:
        ranks_a: Ranks from method A (lower is better)
        ranks_b: Ranks from method B (lower is better)
        method_a_name: Name for method A (for interpretation string)
        method_b_name: Name for method B (for interpretation string)

    Returns:
        Dict with:
        - test: "Mann-Whitney U"
        - statistic: U statistic
        - p_value: p-value for one-sided test
        - significant_at_0.05: True if p < 0.05
        - interpretation: Human-readable result
    """
    # alternative='less' tests if ranks_a < ranks_b (A is better)
    statistic, p_value = mannwhitneyu(ranks_a, ranks_b, alternative="less")

    return {
        "test": "Mann-Whitney U",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_0.05": p_value < 0.05,
        "interpretation": (
            f"{method_a_name} significantly better than {method_b_name}"
            if p_value < 0.05
            else f"No significant difference between {method_a_name} and {method_b_name}"
        ),
    }
