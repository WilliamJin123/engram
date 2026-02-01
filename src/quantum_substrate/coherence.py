"""Coherence dynamics for quantum-inspired memory substrate.

Manages the global tick counter and applies decay/refresh to patterns.
Coherence represents the "quantumness" of a pattern:
- High coherence (near 1.0): pattern participates in interference effects
- Low coherence (near 0.01): pattern behaves classically

Time is measured in operations (ticks), not wall-clock time.
Each store/retrieve/coactivate/bind operation advances the tick counter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from agentic.evolving_pattern import EvolvingPattern
    from quantum_substrate.surprise import SurpriseResult


@dataclass
class CoherenceConfig:
    """Configuration for coherence dynamics."""

    decay_rate: float = 0.05  # Base decay rate (half-life ~14 ticks)
    floor: float = 0.01  # Minimum coherence (never fully zero)
    refresh_cap: float = 1.0  # Maximum coherence after refresh


class CoherenceManager:
    """Manages coherence decay and refresh across all patterns.

    Tracks a global tick counter. Each operation (store, retrieve, coactivate, bind)
    advances the tick. Decay is computed lazily on access using exponential decay
    formula from last_access_tick to current_tick.

    Decay formula: coherence *= exp(-decay_rate * dt / embeddedness)
    where dt = current_tick - last_access_tick
    and embeddedness = 1 + 0.1 * connection_count
    """

    def __init__(self, config: CoherenceConfig | None = None):
        self.config = config or CoherenceConfig()
        self._current_tick: int = 0

    @property
    def current_tick(self) -> int:
        """Current global tick counter."""
        return self._current_tick

    def advance_tick(self) -> int:
        """Advance the global tick counter by one.

        Call this once per operation (store, retrieve, coactivate, bind).
        Returns the new tick value.
        """
        self._current_tick += 1
        return self._current_tick

    def compute_decayed_coherence(
        self,
        pattern: "EvolvingPattern",
    ) -> float:
        """Compute what pattern's coherence would be after decay.

        Uses lazy evaluation: computes decay from last_access_tick to current_tick.
        Does NOT modify the pattern — call apply_decay for that.

        Formula: coherence * exp(-decay_rate * dt / embeddedness)
        """
        dt = self._current_tick - pattern.last_access_tick
        if dt <= 0:
            return pattern.coherence

        effective_rate = self.config.decay_rate / pattern.embeddedness
        decayed = pattern.coherence * math.exp(-effective_rate * dt)
        return max(self.config.floor, decayed)

    def apply_decay(
        self,
        pattern: "EvolvingPattern",
    ) -> float:
        """Apply decay to pattern and update its coherence.

        Returns the new coherence value.
        """
        pattern.coherence = self.compute_decayed_coherence(pattern)
        return pattern.coherence

    def apply_refresh(
        self,
        pattern: "EvolvingPattern",
        activation_strength: float = 1.0,
    ) -> float:
        """Refresh pattern's coherence based on activation strength.

        Refresh amount is proportional to activation_strength.
        Updates last_access_tick to current_tick.

        Args:
            pattern: Pattern to refresh.
            activation_strength: How strongly pattern was activated (0-1).
                - Store: 1.0 (explicit creation)
                - Retrieve: similarity score
                - Coactivate: retrieval score that triggered coactivation
                - Bind: 1.0 (explicit binding)

        Returns the new coherence value.
        """
        # Refresh proportional to activation, but diminishing returns near cap
        headroom = self.config.refresh_cap - pattern.coherence
        refresh_amount = headroom * activation_strength * 0.5  # 50% of headroom max

        pattern.coherence = min(
            self.config.refresh_cap,
            pattern.coherence + refresh_amount
        )
        pattern.last_access_tick = self._current_tick

        return pattern.coherence

    def decay_then_refresh(
        self,
        pattern: "EvolvingPattern",
        activation_strength: float = 1.0,
    ) -> float:
        """Apply decay first, then refresh. Standard operation order.

        Per CONTEXT.md: "decay all patterns first, then refresh activated patterns"
        This method applies both in correct order for a single pattern.

        Returns the new coherence value after both operations.
        """
        self.apply_decay(pattern)
        return self.apply_refresh(pattern, activation_strength)

    def decay_all(
        self,
        patterns: Sequence["EvolvingPattern"],
    ) -> None:
        """Apply decay to all patterns.

        Call this at start of each operation before any refresh.
        """
        for pattern in patterns:
            self.apply_decay(pattern)

    def reset(self) -> None:
        """Reset tick counter to 0. Useful for testing."""
        self._current_tick = 0

    def apply_recoherence(
        self,
        patterns: dict[str, "EvolvingPattern"],
        surprise: "SurpriseResult",
        boost_coefficient: float = 0.5,
    ) -> dict[str, float]:
        """Apply re-coherence based on surprise result.

        Surprise triggers re-coherence of involved patterns, allowing decayed
        or even zero-coherence patterns to be resurrected through unexpected
        retrieval results.

        Scaling per CONTEXT.md:
        - Surprising pattern (unexpected result): full proportional boost
        - Wrong prediction (expected that was wrong): medium boost (0.5x)
        - Other participants: boost scaled by contribution strength (0.3x multiplier)

        All boosts are additive and capped at 1.0 coherence.

        Args:
            patterns: All patterns by ID.
            surprise: Detection result from SurpriseDetector.
            boost_coefficient: Convert surprise magnitude to coherence boost.
                Default 0.5 means magnitude=1.0 gives base_boost=0.5.

        Returns:
            Dict of pattern_id -> coherence delta applied.
            Useful for observability and testing.
        """
        # Skip if no meaningful surprise
        if surprise.magnitude < 0.001:
            return {}

        base_boost = surprise.magnitude * boost_coefficient
        deltas: dict[str, float] = {}

        # 1. Surprising pattern gets full boost
        if surprise.surprising_pattern_id and surprise.surprising_pattern_id in patterns:
            pattern = patterns[surprise.surprising_pattern_id]
            headroom = self.config.refresh_cap - pattern.coherence
            delta = min(base_boost, headroom)
            pattern.coherence += delta
            deltas[surprise.surprising_pattern_id] = delta

        # 2. Expected (wrong prediction) gets 0.5x boost
        if surprise.expected_pattern_id and surprise.expected_pattern_id in patterns:
            pattern = patterns[surprise.expected_pattern_id]
            headroom = self.config.refresh_cap - pattern.coherence
            delta = min(base_boost * 0.5, headroom)
            pattern.coherence += delta
            deltas[surprise.expected_pattern_id] = delta

        # 3. Other participants get 0.3x * involvement
        for pattern_id, involvement in surprise.participant_scores.items():
            # Skip if already handled above
            if pattern_id in deltas:
                continue
            if pattern_id not in patterns:
                continue

            pattern = patterns[pattern_id]
            headroom = self.config.refresh_cap - pattern.coherence
            delta = min(base_boost * involvement * 0.3, headroom)

            # Only record if delta is meaningful
            if delta > 0.0001:
                pattern.coherence += delta
                deltas[pattern_id] = delta

        return deltas
