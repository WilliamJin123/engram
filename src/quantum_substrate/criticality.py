# src/quantum_substrate/criticality.py
"""Criticality dynamics for edge-of-chaos self-regulation.

Criticality is a global parameter (0-1) that controls the order/chaos balance
of the system. It self-adjusts based on retrieval quality and surprise frequency.

Key concepts (from COHR-07):
- 0.5 is optimal "edge of chaos" - balanced exploration/exploitation
- 0.0 is maximum order (rigid, problematic)
- 1.0 is maximum chaos (unstable, problematic)
- Criticality affects tunneling probability via tunneling_amplification
- Self-adjustment uses dampening to prevent oscillation
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CriticalityConfig:
    """Configuration for criticality dynamics."""

    initial_value: float = 0.5  # Start at optimal edge-of-chaos
    adjustment_rate: float = 0.01  # How fast to adjust per feedback cycle
    dampening: float = 0.9  # Resist rapid swings (0.9 = 10% dampen)
    min_feedback_samples: int = 5  # Need this many samples before adjusting
    window_size: int = 20  # Rolling window for feedback tracking

    # Target ranges for self-adjustment
    target_surprise_rate_low: float = 0.1  # Below this = too rigid
    target_surprise_rate_high: float = 0.5  # Above this = too chaotic
    quality_threshold: float = 0.5  # Below this = poor retrieval


@dataclass
class CriticalityState:
    """Global criticality parameter with self-adjustment.

    Criticality controls the order/chaos balance:
    - 0.0: Maximum order (rigid, no exploration) - PROBLEMATIC
    - 0.5: Edge of chaos (optimal, balanced) - IDEAL
    - 1.0: Maximum chaos (unstable, too random) - PROBLEMATIC

    Self-adjusts based on:
    - Retrieval quality: poor matches -> increase chaos
    - Surprise frequency: too many -> decrease, too few -> increase
    """

    value: float = 0.5
    config: CriticalityConfig = field(default_factory=CriticalityConfig)

    # Feedback tracking
    _recent_quality: list[float] = field(default_factory=list)
    _recent_surprises: list[bool] = field(default_factory=list)

    def record_feedback(self, retrieval_quality: float, had_surprise: bool) -> None:
        """Record feedback from a retrieval operation.

        Args:
            retrieval_quality: Top retrieval score (0-1). Higher = better match.
            had_surprise: True if surprise was detected in this retrieval.
        """
        self._recent_quality.append(retrieval_quality)
        self._recent_surprises.append(had_surprise)

        # Maintain window size
        if len(self._recent_quality) > self.config.window_size:
            self._recent_quality.pop(0)
        if len(self._recent_surprises) > self.config.window_size:
            self._recent_surprises.pop(0)

    def self_adjust(self) -> float:
        """Self-adjust criticality based on accumulated feedback.

        Call periodically (e.g., every N operations) to update criticality.

        Returns:
            New criticality value.
        """
        if len(self._recent_quality) < self.config.min_feedback_samples:
            return self.value  # Not enough data

        # Compute signals
        avg_quality = sum(self._recent_quality) / len(self._recent_quality)
        surprise_rate = sum(self._recent_surprises) / len(self._recent_surprises)

        adjustment = 0.0

        # Poor retrieval quality -> need more exploration (increase chaos)
        if avg_quality < self.config.quality_threshold:
            adjustment += self.config.adjustment_rate

        # Too many surprises -> too chaotic -> decrease chaos
        if surprise_rate > self.config.target_surprise_rate_high:
            adjustment -= self.config.adjustment_rate
        # Too few surprises -> too rigid -> increase chaos
        elif surprise_rate < self.config.target_surprise_rate_low:
            adjustment += self.config.adjustment_rate

        # Apply dampening to prevent oscillation
        adjustment *= self.config.dampening

        # Update with bounds [0, 1]
        self.value = max(0.0, min(1.0, self.value + adjustment))

        return self.value

    @property
    def tunneling_amplification(self) -> float:
        """Multiplier for tunneling probability based on criticality.

        At criticality 0.5 (optimal), amplification = 1.0
        At criticality 1.0 (chaotic), amplification = 1.5
        At criticality 0.0 (rigid), amplification = 0.5

        This creates a linear relationship: amp = 0.5 + value
        """
        return 0.5 + self.value

    @property
    def is_healthy(self) -> bool:
        """Check if criticality is in healthy range (not at extremes)."""
        return 0.1 <= self.value <= 0.9

    def reset(self) -> None:
        """Reset criticality to initial state."""
        self.value = self.config.initial_value
        self._recent_quality.clear()
        self._recent_surprises.clear()
