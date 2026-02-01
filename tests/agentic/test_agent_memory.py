# tests/agentic/test_agent_memory.py
"""Tests for AgentMemory high-level API.

AgentMemory integrates:
- MemoryStore (pattern storage and retrieval)
- Coactivation (learning rule)
- LLM interface (semantic reranking)
"""

import pytest
from agentic.agent_memory import AgentMemory, AgentMemoryConfig
from agentic.llm_interface import BaseLLMReranker, MockLLMReranker
from agentic.memory_store import RetrievalResult
from agentic.coactivation import CoactivationConfig


class TestAgentMemoryStore:
    """Test storing memories."""

    def test_store_text_returns_id(self):
        """Storing text returns a memory ID."""
        memory = AgentMemory(dim=1024, k=50)

        memory_id = memory.store("Hello world")

        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

    def test_store_with_metadata(self):
        """Can store text with metadata."""
        memory = AgentMemory(dim=1024, k=50)

        memory_id = memory.store(
            "Alice said hello",
            metadata={"speaker": "alice", "turn": 1},
        )

        pattern = memory.get(memory_id)
        assert pattern is not None
        assert pattern.metadata["speaker"] == "alice"
        assert pattern.metadata["turn"] == 1

    def test_store_multiple_memories(self):
        """Can store multiple memories."""
        memory = AgentMemory(dim=1024, k=50)

        id1 = memory.store("First memory")
        id2 = memory.store("Second memory")
        id3 = memory.store("Third memory")

        assert id1 != id2 != id3
        assert memory.memory_count == 3


class TestAgentMemoryRecall:
    """Test recalling memories."""

    def test_recall_returns_results(self):
        """Recall returns list of results."""
        memory = AgentMemory(dim=1024, k=50)

        memory.store("dogs are great pets")
        memory.store("cats are independent")
        memory.store("birds can fly")

        results = memory.recall("dogs are great pets", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_recall_finds_exact_match(self):
        """Recall finds exact match at top."""
        memory = AgentMemory(dim=1024, k=50)

        memory.store("dogs are great pets")
        memory.store("cats are independent")

        results = memory.recall("dogs are great pets", top_k=2)

        assert results[0].text == "dogs are great pets"

    def test_recall_with_reranking(self):
        """Recall can use LLM reranking."""
        memory = AgentMemory(
            dim=1024,
            k=50,
            reranker=MockLLMReranker(),
        )

        memory.store("dogs bark loudly")
        memory.store("cats meow quietly")
        memory.store("birds sing songs")

        results = memory.recall(
            "tell me about dogs",
            top_k=3,
            use_reranking=True,
        )

        assert "dogs" in results[0].text

    def test_recall_without_reranking(self):
        """Recall can skip reranking."""
        memory = AgentMemory(
            dim=1024,
            k=50,
            reranker=MockLLMReranker(),
        )

        memory.store("test memory")

        results = memory.recall("test memory", top_k=1, use_reranking=False)

        assert len(results) == 1

    def test_recall_empty_store(self):
        """Recall on empty store returns empty list."""
        memory = AgentMemory(dim=1024, k=50)

        results = memory.recall("anything", top_k=5)

        assert len(results) == 0


class TestAgentMemoryLearn:
    """Test learning via coactivation."""

    def test_learn_applies_coactivation(self):
        """Learn applies coactivation to retrieved patterns."""
        memory = AgentMemory(dim=1024, k=50)

        id1 = memory.store("dogs are animals")
        id2 = memory.store("wolves are animals")

        p1_before = set(memory.get(id1).bits)
        p2_before = set(memory.get(id2).bits)
        initial_overlap = len(p1_before & p2_before)

        for _ in range(10):
            results = memory.recall("animals", top_k=2)
            memory.learn(results, strength=0.05)

        p1_after = memory.get(id1).bits
        p2_after = memory.get(id2).bits
        final_overlap = len(p1_after & p2_after)

        assert final_overlap > initial_overlap

    def test_learn_respects_config(self):
        """Learn respects coactivation config."""
        config = AgentMemoryConfig(
            coactivation=CoactivationConfig(max_bits=60, strength=0.2),
        )
        memory = AgentMemory(dim=1024, k=50, config=config)

        id1 = memory.store("one")
        id2 = memory.store("two")

        for _ in range(100):
            results = memory.recall("one two", top_k=2)
            memory.learn(results)

        assert len(memory.get(id1).bits) <= 60
        assert len(memory.get(id2).bits) <= 60

    def test_learn_tracks_acquisition_count(self):
        """Learn increments acquisition count."""
        memory = AgentMemory(dim=1024, k=50)

        id1 = memory.store("alpha")
        id2 = memory.store("beta")

        assert memory.get(id1).acquisition_count == 0

        for _ in range(5):
            results = memory.recall("alpha beta", top_k=2)
            memory.learn(results, strength=0.1)

        assert memory.get(id1).acquisition_count > 0
        assert memory.get(id2).acquisition_count > 0


class TestAgentMemoryWorkflow:
    """Test full workflow: store -> recall -> learn."""

    def test_full_workflow(self):
        """Full workflow improves retrieval over time."""
        memory = AgentMemory(
            dim=1024,
            k=50,
            reranker=MockLLMReranker(),
        )

        # Store memories about dogs and cats
        memory.store("dogs are loyal pets")
        memory.store("dogs love to play fetch")
        memory.store("cats are independent")
        memory.store("cats like to sleep")

        # Simulate multiple interactions about dogs
        for _ in range(5):
            results = memory.recall("tell me about dogs", top_k=2, use_reranking=True)
            memory.learn(results, strength=0.1)

        # Get all patterns
        all_patterns = memory.get_all_patterns()

        # Check we have the right count
        assert len(all_patterns) == 4

    def test_workflow_with_metadata(self):
        """Workflow preserves metadata through operations."""
        memory = AgentMemory(dim=1024, k=50)

        id1 = memory.store(
            "important fact",
            metadata={"source": "user", "priority": "high"},
        )

        results = memory.recall("important fact", top_k=1)
        memory.learn(results)

        pattern = memory.get(id1)
        assert pattern.metadata["source"] == "user"
        assert pattern.metadata["priority"] == "high"


class TestAgentMemoryConfig:
    """Test configuration options."""

    def test_default_config(self):
        """Default config uses sensible defaults."""
        memory = AgentMemory(dim=1024, k=50)

        assert memory.dim == 1024
        assert memory.k == 50
        assert memory.memory_count == 0

    def test_custom_config(self):
        """Can use custom config."""
        config = AgentMemoryConfig(
            retrieval_method="interference",
            use_evolved_bits=True,
            coactivation=CoactivationConfig(max_bits=100),
        )
        memory = AgentMemory(dim=2048, k=100, config=config)

        assert memory.dim == 2048
        assert memory.k == 100

    def test_set_reranker(self):
        """Can set reranker after creation."""
        memory = AgentMemory(dim=1024, k=50)

        assert memory.reranker is None

        memory.set_reranker(MockLLMReranker())

        assert memory.reranker is not None

    def test_custom_reranker(self):
        """Can use custom reranker implementation."""

        class TopicReranker(BaseLLMReranker):
            """Reranker that boosts specific topic."""

            def __init__(self, boost_topic: str):
                self.boost_topic = boost_topic

            def rerank(self, query, candidates, top_k=None):
                scored = []
                for c in candidates:
                    score = 10 if self.boost_topic in c.text.lower() else 0
                    scored.append((c, score))
                scored.sort(key=lambda x: -x[1])
                result = [c for c, _ in scored]
                return result[:top_k] if top_k else result

        memory = AgentMemory(
            dim=1024,
            k=50,
            reranker=TopicReranker("dogs"),
        )

        memory.store("dogs are great")
        memory.store("cats are great")

        results = memory.recall("pets", top_k=2, use_reranking=True)

        assert "dogs" in results[0].text


class TestAgentMemoryMetrics:
    """Test metrics and analysis helpers."""

    def test_memory_count(self):
        """Memory count tracks stored items."""
        memory = AgentMemory(dim=1024, k=50)

        assert memory.memory_count == 0
        memory.store("one")
        assert memory.memory_count == 1
        memory.store("two")
        assert memory.memory_count == 2

    def test_get_all_patterns(self):
        """Can get all patterns for analysis."""
        memory = AgentMemory(dim=1024, k=50)

        memory.store("one")
        memory.store("two")
        memory.store("three")

        patterns = memory.get_all_patterns()

        assert len(patterns) == 3

    def test_average_obesity(self):
        """Can compute average pattern obesity."""
        memory = AgentMemory(dim=1024, k=50)

        memory.store("one")
        memory.store("two")

        # Before learning, obesity should be 1.0
        assert memory.average_obesity == 1.0

        # After learning, obesity may increase
        for _ in range(10):
            results = memory.recall("one two", top_k=2)
            memory.learn(results, strength=0.1)

        assert memory.average_obesity >= 1.0
