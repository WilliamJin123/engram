# src/agentic/memory_store.py
"""Memory store with pattern-based retrieval.

LLM integration is handled externally - this module is LLM-agnostic.
See docs/KEYCYCLE.md for LLM integration details.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

import torch

from agentic.evolving_pattern import EvolvingPattern
from quantum_substrate.coherence import CoherenceManager, CoherenceConfig


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    pattern_id: str
    pattern: EvolvingPattern
    score: float

    @property
    def text(self) -> str:
        return self.pattern.text

    @property
    def metadata(self) -> dict[str, Any]:
        return self.pattern.metadata


class MemoryStore:
    """Store and retrieve patterns with similarity search.

    This is the substrate layer - it handles pattern storage and retrieval.
    LLM reranking should be done externally (see docs/KEYCYCLE.md).

    Integrates CoherenceManager for decay/refresh on operations:
    - Each store/retrieve advances the global tick
    - All patterns decay at start of each operation
    - Accessed patterns refresh proportional to activation strength
    """

    def __init__(
        self,
        dim: int = 1024,
        k: int = 50,
        coherence_config: CoherenceConfig | None = None,
    ):
        self.dim = dim
        self.k = k
        self.patterns: dict[str, EvolvingPattern] = {}
        self.coherence_manager = CoherenceManager(coherence_config)

    def store(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        pattern_id: str | None = None,
    ) -> str:
        """Store text as a pattern.

        Operation order:
        1. Advance tick
        2. Decay all existing patterns
        3. Create and store new pattern
        4. Refresh new pattern with activation_strength=1.0
        """
        # Advance tick for this operation
        self.coherence_manager.advance_tick()

        # Decay all existing patterns first
        self.coherence_manager.decay_all(self.patterns.values())

        # Create and store pattern
        if pattern_id is None:
            pattern_id = str(uuid.uuid4())

        pattern = EvolvingPattern.from_text(
            text=text,
            dim=self.dim,
            k=self.k,
            metadata=metadata or {},
        )
        pattern.metadata["id"] = pattern_id

        # Refresh new pattern (store = full activation)
        self.coherence_manager.apply_refresh(pattern, activation_strength=1.0)

        self.patterns[pattern_id] = pattern
        return pattern_id

    def get(self, pattern_id: str) -> EvolvingPattern | None:
        """Get pattern by ID."""
        return self.patterns.get(pattern_id)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        method: Literal["jaccard", "interference"] = "jaccard",
        use_evolved: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve patterns similar to query.

        Operation order:
        1. Advance tick
        2. Decay all patterns
        3. Score patterns against query
        4. Refresh retrieved patterns proportional to score
        """
        # Advance tick for this operation
        self.coherence_manager.advance_tick()

        # Decay all patterns first
        self.coherence_manager.decay_all(self.patterns.values())

        query_pattern = EvolvingPattern.from_text(query, dim=self.dim, k=self.k)

        if method == "jaccard":
            results = self._retrieve_jaccard(query_pattern, top_k, use_evolved)
        else:
            results = self._retrieve_interference(query_pattern, top_k, use_evolved)

        # Refresh retrieved patterns proportional to score
        for result in results:
            self.coherence_manager.apply_refresh(result.pattern, activation_strength=result.score)

        return results

    def _retrieve_jaccard(
        self,
        query: EvolvingPattern,
        top_k: int,
        use_evolved: bool,
    ) -> list[RetrievalResult]:
        """Retrieve using Jaccard similarity on bit overlap."""
        query_bits = query.bits if use_evolved else set(query.original_bits)

        results = []
        for pattern_id, pattern in self.patterns.items():
            pattern_bits = pattern.bits if use_evolved else set(pattern.original_bits)

            intersection = len(query_bits & pattern_bits)
            union = len(query_bits | pattern_bits)
            score = intersection / union if union > 0 else 0.0

            results.append(RetrievalResult(
                pattern_id=pattern_id,
                pattern=pattern,
                score=score,
            ))

        results.sort(key=lambda r: -r.score)
        return results[:top_k]

    def _retrieve_interference(
        self,
        query: EvolvingPattern,
        top_k: int,
        use_evolved: bool,
    ) -> list[RetrievalResult]:
        """Retrieve using phase-aware interference."""
        if use_evolved:
            query_dense = query.to_dense_evolved()
        else:
            query_dense = query.to_dense()

        results = []
        for pattern_id, pattern in self.patterns.items():
            if use_evolved:
                pattern_dense = pattern.to_dense_evolved()
            else:
                pattern_dense = pattern.to_dense()

            combined = query_dense + pattern_dense

            query_active = query_dense.abs() > 1e-6
            pattern_active = pattern_dense.abs() > 1e-6
            overlap = query_active & pattern_active

            if overlap.sum() == 0:
                score = 0.0
            else:
                interference = combined[overlap].abs().sum().item()
                max_possible = (query_dense[overlap].abs() + pattern_dense[overlap].abs()).sum().item()
                score = interference / max_possible if max_possible > 0 else 0.0

            results.append(RetrievalResult(
                pattern_id=pattern_id,
                pattern=pattern,
                score=score,
            ))

        results.sort(key=lambda r: -r.score)
        return results[:top_k]

    def get_effective_coherence(self, pattern_id: str) -> float | None:
        """Get pattern's current coherence after decay.

        Computes what the pattern's coherence would be at current tick
        without modifying the pattern.
        """
        pattern = self.patterns.get(pattern_id)
        if pattern is None:
            return None
        return self.coherence_manager.compute_decayed_coherence(pattern)

    @property
    def current_tick(self) -> int:
        """Current operation tick count."""
        return self.coherence_manager.current_tick

    @property
    def pattern_count(self) -> int:
        """Number of stored patterns."""
        return len(self.patterns)

    def get_all_patterns(self) -> list[EvolvingPattern]:
        """Get all patterns for analysis."""
        return list(self.patterns.values())
