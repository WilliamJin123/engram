# tests/stress/metrics/visualization.py
"""Visualization utilities for degradation curves.

Provides matplotlib-based plotting for:
- Single method degradation curves with error bands
- Multi-method comparison plots

All plots use fill_between for shaded confidence bands and
properly close figures to prevent memory leaks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_degradation_curve(data: dict, output_path: str) -> None:
    """Plot degradation curve with shaded std band.

    Args:
        data: Dict with keys:
            - noise_level: List of x-axis values (e.g., near-miss counts)
            - mean_rank: List of mean rank values
            - std_rank: List of standard deviation values
        output_path: Where to save the figure (e.g., 'reports/curve.png')
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.array(data["noise_level"])
    y = np.array(data["mean_rank"])
    yerr = np.array(data["std_rank"])

    # Main line
    ax.plot(x, y, "o-", label="Interference Retrieval", color="tab:blue")

    # Shaded confidence band (mean +/- std)
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color="tab:blue")

    ax.set_xlabel("Near-Miss Count")
    ax.set_ylabel("Target Rank (lower is better)")
    ax.set_title("Retrieval Degradation Under Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # Prevent memory leak


def plot_method_comparison(
    noise_levels: list[int],
    methods_data: dict[str, dict],
    output_path: str,
) -> None:
    """Plot degradation curves for multiple methods.

    Args:
        noise_levels: X-axis values (e.g., near-miss counts)
        methods_data: Dict mapping method name to {mean_rank: [], std_rank: []}
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    x = np.array(noise_levels)

    for i, (method_name, data) in enumerate(methods_data.items()):
        y = np.array(data["mean_rank"])
        yerr = np.array(data["std_rank"])
        color = colors[i % len(colors)]

        # Line with markers
        ax.plot(x, y, "o-", label=method_name, color=color)
        # Shaded confidence band
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)

    ax.set_xlabel("Near-Miss Count")
    ax.set_ylabel("Target Rank (lower is better)")
    ax.set_title("Retrieval Method Comparison Under Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # Prevent memory leak
