# src/quantum_substrate/tunneling.py
"""Tunneling mechanics for creative/exploratory retrieval.

Tunneling allows high-coherence patterns to probabilistically activate
weakly-related patterns via connection paths. This enables associative
leaps to related but dissimilar patterns.

Key constraints (from COHR-06):
- High-coherence patterns can initiate tunneling; low-coherence cannot
- Tunneling requires a connection path between patterns
- Tunneling strength scales with source pattern coherence
- Creative mode amplifies tunneling probability
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.evolving_pattern import EvolvingPattern


@dataclass
class TunnelingConfig:
    """Configuration for tunneling behavior."""

    baseline_probability: float = 0.1  # Base chance of tunnel activation
    creative_mode_multiplier: float = 3.0  # Amplification in creative mode
    min_source_coherence: float = 0.3  # Source must have this coherence to tunnel
    max_bit_overlap: float = 0.2  # Target must have low overlap (< 20%)
    coherence_exponent: float = 1.0  # Linear scaling by default


@dataclass
class TunnelingResult:
    """Result of tunneling attempt."""

    tunneled: bool
    source_pattern_id: str
    target_pattern_id: str | None
    tunnel_strength: float  # How strong the activation was (0-1)


@dataclass
class CreativeModeTracker:
    """Track conditions for auto-creative mode activation.

    Auto-creative triggers when retrieval repeatedly fails (low scores).
    This helps the system make associative leaps when direct matches
    are not working.
    """

    recent_scores: list[float] = field(default_factory=list)
    failure_threshold: float = 0.3  # Score below this = "failure"
    consecutive_failures_needed: int = 3
    max_history: int = 10

    def record_score(self, score: float) -> None:
        """Record a retrieval score."""
        self.recent_scores.append(score)
        if len(self.recent_scores) > self.max_history:
            self.recent_scores.pop(0)

    def should_activate_creative(self) -> bool:
        """Check if creative mode should auto-activate.

        Returns True if the last N retrieval scores were all below
        the failure threshold, indicating repeated retrieval failures.
        """
        if len(self.recent_scores) < self.consecutive_failures_needed:
            return False
        # Check if last N scores are all below threshold
        recent = self.recent_scores[-self.consecutive_failures_needed:]
        return all(s < self.failure_threshold for s in recent)

    def reset(self) -> None:
        """Reset tracking history."""
        self.recent_scores.clear()


def attempt_tunneling(
    source: "EvolvingPattern",
    source_id: str,
    all_patterns: dict[str, "EvolvingPattern"],
    connections: dict[str, set[str]],  # pattern_id -> connected pattern IDs
    config: TunnelingConfig,
    creative_mode: bool = False,
    rng: random.Random | None = None,  # For deterministic testing
) -> TunnelingResult:
    """Attempt tunneling from source to a weakly-related connected pattern.

    Tunneling enables high-coherence patterns to activate patterns that are:
    1. Connected (via connection_map)
    2. Weakly related (low bit overlap)

    This creates associative leaps - activating dissimilar but connected patterns.

    Args:
        source: The pattern attempting to tunnel.
        source_id: ID of the source pattern.
        all_patterns: Dictionary of all patterns by ID.
        connections: Map of pattern_id -> set of connected pattern IDs.
        config: Tunneling configuration.
        creative_mode: If True, amplify tunneling probability.
        rng: Optional Random instance for deterministic testing.

    Returns:
        TunnelingResult with tunneled=True if successful.
    """
    _rng = rng if rng is not None else random

    # Early exit: source must have sufficient coherence to initiate tunneling
    if source.coherence < config.min_source_coherence:
        return TunnelingResult(
            tunneled=False,
            source_pattern_id=source_id,
            target_pattern_id=None,
            tunnel_strength=0.0,
        )

    # Get connected patterns
    connected_ids = connections.get(source_id, set())
    if not connected_ids:
        return TunnelingResult(
            tunneled=False,
            source_pattern_id=source_id,
            target_pattern_id=None,
            tunnel_strength=0.0,
        )

    # Find candidate targets: connected AND low bit overlap
    source_bits = source.bits
    candidates: list[tuple[str, float]] = []  # (pattern_id, inverse_overlap)

    for target_id in connected_ids:
        if target_id not in all_patterns:
            continue

        target = all_patterns[target_id]
        target_bits = target.bits

        # Calculate Jaccard overlap
        intersection = len(source_bits & target_bits)
        union = len(source_bits | target_bits)
        overlap_ratio = intersection / union if union > 0 else 0.0

        # Include target only if overlap is below threshold
        if overlap_ratio < config.max_bit_overlap:
            # Weight by inverse overlap (more different = more interesting)
            # Use 1 - overlap so higher weight means more different
            inverse_overlap = 1.0 - overlap_ratio
            candidates.append((target_id, inverse_overlap))

    if not candidates:
        return TunnelingResult(
            tunneled=False,
            source_pattern_id=source_id,
            target_pattern_id=None,
            tunnel_strength=0.0,
        )

    # Calculate tunnel probability
    tunnel_strength = source.coherence ** config.coherence_exponent
    probability = config.baseline_probability * tunnel_strength

    if creative_mode:
        probability *= config.creative_mode_multiplier

    # Clamp probability to [0, 1]
    probability = max(0.0, min(1.0, probability))

    # Probabilistic check
    if _rng.random() >= probability:
        return TunnelingResult(
            tunneled=False,
            source_pattern_id=source_id,
            target_pattern_id=None,
            tunnel_strength=tunnel_strength,
        )

    # Select target weighted by inverse overlap (more different = more likely)
    total_weight = sum(w for _, w in candidates)
    if total_weight <= 0:
        # Uniform selection if all weights are zero
        target_id = _rng.choice([c[0] for c in candidates])
    else:
        # Weighted selection
        r = _rng.random() * total_weight
        cumulative = 0.0
        target_id = candidates[0][0]  # Default
        for cid, weight in candidates:
            cumulative += weight
            if r <= cumulative:
                target_id = cid
                break

    return TunnelingResult(
        tunneled=True,
        source_pattern_id=source_id,
        target_pattern_id=target_id,
        tunnel_strength=tunnel_strength,
    )
