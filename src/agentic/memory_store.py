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
    """

    def __init__(self, dim: int = 1024, k: int = 50):
        self.dim = dim
        self.k = k
        self.patterns: dict[str, EvolvingPattern] = {}

    def store(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        pattern_id: str | None = None,
    ) -> str:
        """Store text as a pattern."""
        if pattern_id is None:
            pattern_id = str(uuid.uuid4())

        pattern = EvolvingPattern.from_text(
            text=text,
            dim=self.dim,
            k=self.k,
            metadata=metadata or {},
        )
        pattern.metadata["id"] = pattern_id

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
        """Retrieve patterns similar to query."""
        query_pattern = EvolvingPattern.from_text(query, dim=self.dim, k=self.k)

        if method == "jaccard":
            return self._retrieve_jaccard(query_pattern, top_k, use_evolved)
        else:
            return self._retrieve_interference(query_pattern, top_k, use_evolved)

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

    @property
    def pattern_count(self) -> int:
        """Number of stored patterns."""
        return len(self.patterns)

    def get_all_patterns(self) -> list[EvolvingPattern]:
        """Get all patterns for analysis."""
        return list(self.patterns.values())
