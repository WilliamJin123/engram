# tests/conftest.py
"""Shared noise generation fixtures for stress-testing retrieval.

This module provides:
- NoiseLevel: Named presets (NONE, LOW, MEDIUM, HIGH)
- NoiseConfig: Configuration for noise generation
- NoiseResult: Result of noise injection with pattern IDs

All noise generation is seeded for reproducibility - seed is REQUIRED.

Usage:
    from tests.conftest import NoiseLevel, NoiseConfig, NoiseResult

    # Using presets
    config = NoiseConfig.from_preset(NoiseLevel.HIGH, seed=42)

    # Custom configuration
    config = NoiseConfig(
        near_miss_ratio=1.5,
        clutter_ratio=2.0,
        seed=42
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NoiseLevel(Enum):
    """Named noise level presets for stress testing.

    Each level defines a ratio of noise patterns to target patterns:
    - NONE: No noise (baseline testing)
    - LOW: Light noise (0.5 near-misses, 1.0 clutter per target)
    - MEDIUM: Moderate noise (1.0 near-misses, 3.0 clutter per target)
    - HIGH: Heavy noise (2.0 near-misses, 5.0 clutter per target)
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class NoiseConfig:
    """Configuration for noise generation.

    All noise generation is seeded for reproducibility.
    The seed parameter is REQUIRED and has no default.

    Attributes:
        near_miss_ratio: Near-miss patterns per target (e.g., 1.0 = 1 per target).
            Fractional parts are treated probabilistically.
        clutter_ratio: Clutter patterns per target.
        near_miss_overlap: Fraction of bits shared with target (0-1).
            Must be < 0.5 to avoid near-misses being too similar to targets.
        near_miss_phase_noise: Phase perturbation in radians (~1.5 = pi/2).
            Higher values create more phase disruption.
        coherence_range: (min, max) coherence for near-miss patterns.
            Simulates aged patterns, not fresh baseline coherence.
        clutter_coherence_range: (min, max) coherence for clutter patterns.
            Broader range than near-misses for realistic variation.
        seed: RNG seed for reproducibility (REQUIRED - no default).
    """

    near_miss_ratio: float = 0.0
    clutter_ratio: float = 0.0
    near_miss_overlap: float = 0.4
    near_miss_phase_noise: float = 1.5
    coherence_range: tuple[float, float] = (0.2, 0.7)
    clutter_coherence_range: tuple[float, float] = (0.1, 0.9)
    seed: int = field(default_factory=lambda: _raise_seed_required())

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.near_miss_overlap >= 0.5:
            raise ValueError(
                f"near_miss_overlap must be < 0.5, got {self.near_miss_overlap}"
            )
        if self.near_miss_ratio < 0:
            raise ValueError(
                f"near_miss_ratio must be >= 0, got {self.near_miss_ratio}"
            )
        if self.clutter_ratio < 0:
            raise ValueError(f"clutter_ratio must be >= 0, got {self.clutter_ratio}")

    @classmethod
    def from_preset(cls, level: NoiseLevel, seed: int) -> NoiseConfig:
        """Create config from named preset.

        Args:
            level: NoiseLevel preset
            seed: RNG seed for reproducibility (REQUIRED)

        Returns:
            NoiseConfig with preset values
        """
        presets = {
            NoiseLevel.NONE: cls(
                near_miss_ratio=0.0,
                clutter_ratio=0.0,
                seed=seed,
            ),
            NoiseLevel.LOW: cls(
                near_miss_ratio=0.5,
                clutter_ratio=1.0,
                seed=seed,
            ),
            NoiseLevel.MEDIUM: cls(
                near_miss_ratio=1.0,
                clutter_ratio=3.0,
                seed=seed,
            ),
            NoiseLevel.HIGH: cls(
                near_miss_ratio=2.0,
                clutter_ratio=5.0,
                seed=seed,
            ),
        }
        return presets[level]


def _raise_seed_required() -> int:
    """Raise error when seed is not provided."""
    raise TypeError(
        "NoiseConfig requires 'seed' parameter. "
        "Use NoiseConfig.from_preset(level, seed=N) or NoiseConfig(..., seed=N)"
    )


@dataclass
class NoiseResult:
    """Result of noise injection - holds pattern IDs and metadata.

    Attributes:
        near_misses: Mapping from target_id to list of generated near_miss_ids.
        clutter_ids: List of all generated clutter pattern IDs.
        config: The NoiseConfig used to generate this noise.
    """

    near_misses: dict[str, list[str]]
    clutter_ids: list[str]
    config: NoiseConfig
