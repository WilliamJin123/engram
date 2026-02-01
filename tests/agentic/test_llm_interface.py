# tests/agentic/test_llm_interface.py
"""Tests for LLM reranking interface.

The interface is LLM-agnostic. Real LLM integration should use
the approach described in docs/KEYCYCLE.md.
"""

import pytest
from agentic.llm_interface import BaseLLMReranker, MockLLMReranker
from agentic.memory_store import MemoryStore, RetrievalResult
from agentic.evolving_pattern import EvolvingPattern


class TestMockLLMReranker:
    """Test mock LLM for offline testing."""

    def test_mock_reranker_returns_same_order_for_exact_match(self):
        """Mock reranker keeps exact matches at top."""
        reranker = MockLLMReranker()
        store = MemoryStore(dim=1024, k=50)

        store.store("dogs are great pets")
        store.store("cats are independent")

        candidates = store.retrieve("dogs are great pets", top_k=2)
        reranked = reranker.rerank("dogs are great pets", candidates)

        assert reranked[0].text == "dogs are great pets"

    def test_mock_reranker_uses_keyword_overlap(self):
        """Mock reranker uses simple keyword matching."""
        reranker = MockLLMReranker()
        store = MemoryStore(dim=1024, k=50)

        store.store("dogs bark loudly")
        store.store("cats meow quietly")
        store.store("birds sing songs")

        candidates = store.retrieve("dogs", top_k=3, method="jaccard")
        reranked = reranker.rerank("tell me about dogs", candidates)

        assert "dogs" in reranked[0].text


class TestLLMRerankerInterface:
    """Test the LLM reranker interface structure."""

    def test_reranker_accepts_candidates_and_query(self):
        """Reranker interface takes query and candidates."""
        reranker = MockLLMReranker()

        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)
        candidates = [
            RetrievalResult(pattern_id="1", pattern=pattern, score=0.5),
            RetrievalResult(pattern_id="2", pattern=pattern, score=0.3),
        ]

        result = reranker.rerank("query", candidates)

        assert isinstance(result, list)
        assert all(isinstance(r, RetrievalResult) for r in result)

    def test_custom_reranker_can_be_implemented(self):
        """Custom rerankers can implement the interface."""

        class ReverseReranker(BaseLLMReranker):
            """A reranker that reverses the order (for testing)."""

            def rerank(self, query, candidates, top_k=None):
                result = list(reversed(candidates))
                if top_k:
                    result = result[:top_k]
                return result

        reranker = ReverseReranker()

        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)
        candidates = [
            RetrievalResult(pattern_id="1", pattern=pattern, score=0.9),
            RetrievalResult(pattern_id="2", pattern=pattern, score=0.5),
            RetrievalResult(pattern_id="3", pattern=pattern, score=0.1),
        ]

        result = reranker.rerank("query", candidates)

        assert result[0].pattern_id == "3"
        assert result[-1].pattern_id == "1"
