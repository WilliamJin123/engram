# src/agentic/coactivation.py
"""Coactivation-based associative learning for evolving patterns.

When patterns are retrieved together (by LLM judgment or substrate retrieval),
they undergo coactivation which causes them to share bits over time.

This creates emergent semantic similarity without pre-baked structure:
- Patterns frequently retrieved together become more similar
- The similarity is in the substrate, not in LLM embeddings
- Retrieval improves over time as patterns cluster

Key design: Only ORIGINAL bits are transferred, not acquired bits.
This prevents transitive pollution (A learns from B, C learns from A's
borrowed bits, leading to everything becoming similar).

LLM integration: The LLM decides WHICH patterns to coactivate (via reranking).
This module provides the learning rule. See docs/KEYCYCLE.md for LLM details.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from agentic.evolving_pattern import EvolvingPattern

if TYPE_CHECKING:
    from quantum_substrate.coherence import CoherenceManager


@dataclass
class CoactivationConfig:
    """Configuration for coactivation learning."""

    strength: float = 0.01  # Base probability of bit transfer
    max_bits: int = 200  # Never exceed this many bits
    obesity_decay: bool = True  # Large patterns learn slower
    bidirectional: bool = True  # Transfer in both directions


def coactivate(
    patterns: Sequence[EvolvingPattern],
    strength: float | None = None,
    config: CoactivationConfig | None = None,
    coherence_manager: "CoherenceManager | None" = None,
    activation_scores: Sequence[float] | None = None,
    rng: random.Random | None = None,
) -> None:
    """Apply coactivation to a group of patterns.

    Patterns that are retrieved together share bits.
    Only ORIGINAL bits are transferred to prevent transitive pollution.

    Args:
        patterns: Patterns to coactivate (typically top-k from retrieval).
        strength: Learning rate (overrides config if provided).
        config: Full configuration (uses defaults if not provided).
        coherence_manager: Optional manager to refresh coherence.
        activation_scores: Optional scores for each pattern (for proportional refresh).
        rng: Optional Random instance for deterministic testing.
    """
    if config is None:
        config = CoactivationConfig()
    if strength is not None:
        config = CoactivationConfig(
            strength=strength,
            max_bits=config.max_bits,
            obesity_decay=config.obesity_decay,
            bidirectional=config.bidirectional,
        )

    # Create local RNG reference (falls back to global random module)
    _rng = rng if rng is not None else random

    # Coactivation logic with connection tracking
    for i, p1 in enumerate(patterns):
        for p2 in patterns[i + 1:]:
            _transfer_bits(p1, p2, config, _rng)
            if config.bidirectional:
                _transfer_bits(p2, p1, config, _rng)

    # Optional coherence refresh for coactivated patterns
    if coherence_manager is not None:
        if activation_scores is None:
            # Default to 1.0 for all patterns
            activation_scores = [1.0] * len(patterns)
        for pattern, score in zip(patterns, activation_scores):
            coherence_manager.apply_refresh(pattern, activation_strength=score)


def _transfer_bits(
    source: EvolvingPattern,
    target: EvolvingPattern,
    config: CoactivationConfig,
    rng: random.Random | None = None,
) -> None:
    """Transfer bits from source to target.

    Only transfers ORIGINAL bits from source (not acquired bits).
    Respects max_bits budget and obesity decay.
    Also increments connection_count for embeddedness tracking.

    Args:
        source: Pattern to transfer bits from.
        target: Pattern to transfer bits to.
        config: Coactivation configuration.
        rng: Optional Random instance for deterministic testing.
    """
    transferable = source.original_bits - target.bits

    if not transferable:
        # Still count the connection attempt for embeddedness
        target.connection_count += 1
        return

    effective_strength = config.strength
    if config.obesity_decay:
        obesity = target.obesity
        effective_strength = config.strength / max(1.0, obesity)

    n_transfer = max(1, int(len(transferable) * effective_strength))

    available_slots = config.max_bits - len(target.bits)
    if available_slots <= 0:
        # Count connection even when at capacity
        target.connection_count += 1
        return
    n_transfer = min(n_transfer, available_slots)

    # Use provided RNG or fall back to global random module
    _rng = rng if rng is not None else random
    bits_to_add = set(_rng.sample(list(transferable), min(n_transfer, len(transferable))))

    for bit in bits_to_add:
        target.bits.add(bit)
        target.acquired_bits.add(bit)
        if bit in source.phases:
            target.phases[bit] = source.phases[bit]

    # Track connection and acquisition
    target.connection_count += 1
    target.acquisition_count += 1
