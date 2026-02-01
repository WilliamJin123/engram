# tests/agentic/test_evaluation.py
"""Tests for evaluation metrics measuring emergent semantic clustering.

This module tests metrics for:
1. semantic_separation() - measure within-group vs cross-group similarity
2. pattern_obesity_stats() - track pattern growth metrics
3. retrieval_precision() - measure if coactivation improves retrieval
"""

import pytest
from agentic.evolving_pattern import EvolvingPattern
from agentic.memory_store import MemoryStore
from agentic.coactivation import coactivate, CoactivationConfig
from agentic.evaluation import (
    semantic_separation,
    pattern_obesity_stats,
    retrieval_precision,
    SemanticSeparationResult,
    ObesityStats,
    RetrievalPrecisionResult,
)


# Test data: grouped by semantic topic
ANIMAL_TEXTS = [
    "dogs are loyal pets that love their owners",
    "wolves hunt in packs at night",
    "cats are independent and self-sufficient",
    "horses run fast across open fields",
]

TECH_TEXTS = [
    "machine learning requires lots of training data",
    "neural networks can recognize images accurately",
    "databases store information efficiently",
    "algorithms process data in sorted order",
]

FOOD_TEXTS = [
    "pizza is delicious with extra cheese",
    "sushi requires fresh fish and rice",
    "bread needs time to rise properly",
    "soup is best served hot in winter",
]


class TestSemanticSeparation:
    """Test semantic separation metric."""

    def test_semantic_separation_returns_result_object(self):
        """semantic_separation returns a SemanticSeparationResult."""
        patterns_by_group = {
            "animals": [EvolvingPattern.from_text(t, dim=1024, k=50) for t in ANIMAL_TEXTS],
            "tech": [EvolvingPattern.from_text(t, dim=1024, k=50) for t in TECH_TEXTS],
        }

        result = semantic_separation(patterns_by_group)

        assert isinstance(result, SemanticSeparationResult)
        assert hasattr(result, "within_group_similarity")
        assert hasattr(result, "cross_group_similarity")
        assert hasattr(result, "separation_ratio")

    def test_semantic_separation_values_are_valid(self):
        """Similarity values should be between 0 and 1."""
        patterns_by_group = {
            "animals": [EvolvingPattern.from_text(t, dim=1024, k=50) for t in ANIMAL_TEXTS],
            "tech": [EvolvingPattern.from_text(t, dim=1024, k=50) for t in TECH_TEXTS],
        }

        result = semantic_separation(patterns_by_group)

        assert 0 <= result.within_group_similarity <= 1
        assert 0 <= result.cross_group_similarity <= 1
        assert result.separation_ratio >= 0

    def test_semantic_separation_with_single_group(self):
        """Single group should have valid within-group similarity."""
        patterns_by_group = {
            "animals": [EvolvingPattern.from_text(t, dim=1024, k=50) for t in ANIMAL_TEXTS],
        }

        result = semantic_separation(patterns_by_group)

        assert result.within_group_similarity >= 0
        # Cross-group is 0 when only one group
        assert result.cross_group_similarity == 0

    def test_coactivation_increases_within_group_similarity(self):
        """Coactivating patterns within a group should increase their similarity."""
        patterns_by_group = {
            "animals": [EvolvingPattern.from_text(t, dim=1024, k=50) for t in ANIMAL_TEXTS[:2]],
            "tech": [EvolvingPattern.from_text(t, dim=1024, k=50) for t in TECH_TEXTS[:2]],
        }

        # Measure before coactivation
        result_before = semantic_separation(patterns_by_group)

        # Coactivate only within groups
        for _ in range(10):
            coactivate(patterns_by_group["animals"], strength=0.1)
            coactivate(patterns_by_group["tech"], strength=0.1)

        # Measure after coactivation
        result_after = semantic_separation(patterns_by_group)

        # Within-group similarity should increase
        assert result_after.within_group_similarity >= result_before.within_group_similarity

    def test_semantic_separation_with_empty_group(self):
        """Empty groups should be handled gracefully."""
        patterns_by_group = {
            "animals": [EvolvingPattern.from_text(t, dim=1024, k=50) for t in ANIMAL_TEXTS],
            "empty": [],
        }

        result = semantic_separation(patterns_by_group)

        # Should compute without error, using non-empty groups
        assert result.within_group_similarity >= 0


class TestPatternObesityStats:
    """Test pattern obesity statistics."""

    def test_obesity_stats_returns_result_object(self):
        """pattern_obesity_stats returns an ObesityStats object."""
        patterns = [EvolvingPattern.from_text(t, dim=1024, k=50) for t in ANIMAL_TEXTS]

        stats = pattern_obesity_stats(patterns)

        assert isinstance(stats, ObesityStats)
        assert hasattr(stats, "mean_obesity")
        assert hasattr(stats, "max_obesity")
        assert hasattr(stats, "min_obesity")
        assert hasattr(stats, "mean_acquired_ratio")
        assert hasattr(stats, "total_patterns")

    def test_fresh_patterns_have_obesity_1(self):
        """Fresh patterns (no coactivation) should have obesity 1.0."""
        patterns = [EvolvingPattern.from_text(t, dim=1024, k=50) for t in ANIMAL_TEXTS]

        stats = pattern_obesity_stats(patterns)

        assert stats.mean_obesity == 1.0
        assert stats.max_obesity == 1.0
        assert stats.min_obesity == 1.0
        assert stats.mean_acquired_ratio == 0.0

    def test_coactivation_increases_obesity(self):
        """Coactivation should increase pattern obesity."""
        patterns = [EvolvingPattern.from_text(t, dim=1024, k=50) for t in ANIMAL_TEXTS[:2]]

        stats_before = pattern_obesity_stats(patterns)

        # Coactivate patterns
        for _ in range(10):
            coactivate(patterns, strength=0.1)

        stats_after = pattern_obesity_stats(patterns)

        assert stats_after.mean_obesity > stats_before.mean_obesity
        assert stats_after.mean_acquired_ratio > 0

    def test_obesity_stats_with_empty_list(self):
        """Empty pattern list should return zero values."""
        stats = pattern_obesity_stats([])

        assert stats.mean_obesity == 0
        assert stats.max_obesity == 0
        assert stats.min_obesity == 0
        assert stats.total_patterns == 0

    def test_obesity_stats_count_patterns(self):
        """total_patterns should match input count."""
        patterns = [EvolvingPattern.from_text(t, dim=1024, k=50) for t in ANIMAL_TEXTS]

        stats = pattern_obesity_stats(patterns)

        assert stats.total_patterns == len(ANIMAL_TEXTS)


class TestRetrievalPrecision:
    """Test retrieval precision metric."""

    def test_retrieval_precision_returns_result_object(self):
        """retrieval_precision returns a RetrievalPrecisionResult."""
        store = MemoryStore(dim=1024, k=50)

        # Store patterns with known group labels
        for text in ANIMAL_TEXTS:
            store.store(text, metadata={"group": "animals"})
        for text in TECH_TEXTS:
            store.store(text, metadata={"group": "tech"})

        # Query with expected group
        queries = [
            ("dogs are great", "animals"),
            ("machine learning", "tech"),
        ]

        result = retrieval_precision(store, queries, top_k=3)

        assert isinstance(result, RetrievalPrecisionResult)
        assert hasattr(result, "precision_at_k")
        assert hasattr(result, "mean_reciprocal_rank")
        assert hasattr(result, "total_queries")

    def test_retrieval_precision_exact_match_high(self):
        """Exact matches should have high precision."""
        store = MemoryStore(dim=1024, k=50)

        store.store("dogs are loyal pets", metadata={"group": "animals"})
        store.store("cats are independent", metadata={"group": "animals"})
        store.store("computers process data", metadata={"group": "tech"})

        queries = [
            ("dogs are loyal pets", "animals"),
        ]

        result = retrieval_precision(store, queries, top_k=1)

        # Exact match should be top result
        assert result.precision_at_k >= 0.9

    def test_retrieval_precision_values_bounded(self):
        """Precision values should be between 0 and 1."""
        store = MemoryStore(dim=1024, k=50)

        for text in ANIMAL_TEXTS:
            store.store(text, metadata={"group": "animals"})
        for text in TECH_TEXTS:
            store.store(text, metadata={"group": "tech"})

        queries = [
            ("dogs", "animals"),
            ("computers", "tech"),
        ]

        result = retrieval_precision(store, queries, top_k=3)

        assert 0 <= result.precision_at_k <= 1
        assert 0 <= result.mean_reciprocal_rank <= 1

    def test_coactivation_can_improve_retrieval(self):
        """Coactivation within groups should maintain or improve retrieval precision."""
        store = MemoryStore(dim=1024, k=50)

        # Store patterns with known group labels
        animal_ids = [store.store(text, metadata={"group": "animals"}) for text in ANIMAL_TEXTS]
        tech_ids = [store.store(text, metadata={"group": "tech"}) for text in TECH_TEXTS]

        queries = [
            ("loyal pets dogs", "animals"),
            ("neural networks AI", "tech"),
        ]

        # Measure baseline precision
        result_before = retrieval_precision(store, queries, top_k=2)

        # Get actual patterns and coactivate within groups
        animal_patterns = [store.get(pid) for pid in animal_ids]
        tech_patterns = [store.get(pid) for pid in tech_ids]

        for _ in range(5):
            coactivate(animal_patterns, strength=0.1)
            coactivate(tech_patterns, strength=0.1)

        # Measure precision after coactivation
        result_after = retrieval_precision(store, queries, top_k=2)

        # Precision should not decrease significantly (may increase)
        # Allow small decrease due to noise
        assert result_after.precision_at_k >= result_before.precision_at_k - 0.2

    def test_retrieval_precision_with_no_queries(self):
        """Empty query list should return zero values."""
        store = MemoryStore(dim=1024, k=50)
        store.store("test", metadata={"group": "test"})

        result = retrieval_precision(store, [], top_k=3)

        assert result.total_queries == 0
        assert result.precision_at_k == 0
        assert result.mean_reciprocal_rank == 0


class TestEvaluationIntegration:
    """Integration tests combining multiple metrics."""

    def test_full_evaluation_workflow(self):
        """Test a complete evaluation workflow."""
        # Create patterns organized by group
        patterns_by_group = {
            "animals": [EvolvingPattern.from_text(t, dim=1024, k=50) for t in ANIMAL_TEXTS],
            "tech": [EvolvingPattern.from_text(t, dim=1024, k=50) for t in TECH_TEXTS],
            "food": [EvolvingPattern.from_text(t, dim=1024, k=50) for t in FOOD_TEXTS],
        }

        # Initial metrics
        sep_initial = semantic_separation(patterns_by_group)
        all_patterns = [p for group in patterns_by_group.values() for p in group]
        obesity_initial = pattern_obesity_stats(all_patterns)

        # Simulate learning: coactivate within groups
        for _ in range(5):
            for group_patterns in patterns_by_group.values():
                coactivate(group_patterns, strength=0.05)

        # Final metrics
        sep_final = semantic_separation(patterns_by_group)
        obesity_final = pattern_obesity_stats(all_patterns)

        # Verify metrics are computed correctly
        assert sep_initial.separation_ratio >= 0
        assert sep_final.separation_ratio >= 0
        assert obesity_initial.mean_obesity == 1.0
        assert obesity_final.mean_obesity >= 1.0

        # Print summary for manual inspection
        print(f"\nEvaluation Summary:")
        print(f"  Initial separation ratio: {sep_initial.separation_ratio:.4f}")
        print(f"  Final separation ratio: {sep_final.separation_ratio:.4f}")
        print(f"  Initial mean obesity: {obesity_initial.mean_obesity:.4f}")
        print(f"  Final mean obesity: {obesity_final.mean_obesity:.4f}")
        print(f"  Final max obesity: {obesity_final.max_obesity:.4f}")
