# src/agentic/llm_interface.py
"""LLM interface for semantic reranking.

This module provides an abstract interface for LLM reranking.
The actual LLM integration should follow docs/KEYCYCLE.md.

Includes a MockLLMReranker for testing without API calls.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.memory_store import RetrievalResult


class BaseLLMReranker(ABC):
    """Abstract base for LLM rerankers.

    Implement this interface to integrate with your LLM of choice.
    See docs/KEYCYCLE.md for integration details.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list["RetrievalResult"],
        top_k: int | None = None,
    ) -> list["RetrievalResult"]:
        """Rerank candidates by semantic relevance to query.

        Args:
            query: The user's query.
            candidates: Candidate results from pattern retrieval.
            top_k: Limit results (None = return all).

        Returns:
            Candidates reordered by semantic relevance.
        """
        pass


class MockLLMReranker(BaseLLMReranker):
    """Mock LLM reranker using keyword overlap.

    For testing without API calls. Simulates semantic judgment
    using simple word overlap between query and candidates.

    This is NOT a real LLM - it's a placeholder for testing.
    For real LLM integration, see docs/KEYCYCLE.md.
    """

    def rerank(
        self,
        query: str,
        candidates: list["RetrievalResult"],
        top_k: int | None = None,
    ) -> list["RetrievalResult"]:
        """Rerank using keyword overlap as proxy for semantics."""
        query_words = set(self._tokenize(query.lower()))

        scored = []
        for candidate in candidates:
            text_words = set(self._tokenize(candidate.text.lower()))

            overlap = len(query_words & text_words)
            if query.lower() in candidate.text.lower():
                overlap += 10

            scored.append((candidate, overlap))

        scored.sort(key=lambda x: -x[1])

        result = [c for c, _ in scored]
        if top_k is not None:
            result = result[:top_k]
        return result

    def _tokenize(self, text: str) -> list[str]:
        """Simple word tokenization."""
        return re.findall(r'\b\w+\b', text)
