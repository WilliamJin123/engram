# tests/agentic/test_memory_store.py
"""Tests for MemoryStore with pattern-based retrieval."""

import pytest
from agentic.memory_store import MemoryStore
from agentic.evolving_pattern import EvolvingPattern


class TestMemoryStoreBasics:
    """Test basic store and retrieve operations."""

    def test_store_and_retrieve_by_id(self):
        """Can store pattern and retrieve by ID."""
        store = MemoryStore(dim=1024, k=50)

        pattern_id = store.store("Hello world", metadata={"speaker": "alice"})

        retrieved = store.get(pattern_id)
        assert retrieved is not None
        assert retrieved.text == "Hello world"
        assert retrieved.metadata["speaker"] == "alice"

    def test_retrieve_by_pattern_similarity(self):
        """Can retrieve similar patterns."""
        store = MemoryStore(dim=1024, k=50)

        store.store("dogs are great pets")
        store.store("cats are independent")
        store.store("the weather is nice today")

        results = store.retrieve("dogs are great pets", top_k=3)

        assert len(results) == 3
        assert results[0].text == "dogs are great pets"
        assert results[0].score == 1.0

    def test_jaccard_retrieval(self):
        """Jaccard similarity finds overlapping patterns."""
        store = MemoryStore(dim=1024, k=50)

        store.store("alpha beta gamma")
        store.store("alpha beta delta")
        store.store("epsilon zeta eta")

        results = store.retrieve("alpha beta gamma", top_k=3, method="jaccard")

        texts = [r.text for r in results]
        assert "alpha beta gamma" in texts[:2]

    def test_interference_retrieval(self):
        """Interference retrieval uses phase information."""
        store = MemoryStore(dim=1024, k=50)

        store.store("meeting at noon")
        store.store("lunch at noon")
        store.store("random other thing")

        results = store.retrieve("noon appointment", top_k=3, method="interference")

        assert len(results) == 3


class TestMemoryStoreMetrics:
    """Test metric tracking."""

    def test_tracks_pattern_count(self):
        """Store tracks number of patterns."""
        store = MemoryStore(dim=1024, k=50)

        assert store.pattern_count == 0
        store.store("one")
        assert store.pattern_count == 1
        store.store("two")
        assert store.pattern_count == 2

    def test_get_all_patterns(self):
        """Can retrieve all patterns for analysis."""
        store = MemoryStore(dim=1024, k=50)

        store.store("one")
        store.store("two")
        store.store("three")

        all_patterns = store.get_all_patterns()
        assert len(all_patterns) == 3
