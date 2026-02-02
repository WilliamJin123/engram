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
from quantum_substrate.surprise import SurpriseDetector, SurpriseResult


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
        self.connection_map: dict[str, set[str]] = {}  # pattern_id -> connected pattern IDs

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

        # Initialize connection entry for new pattern
        if pattern_id not in self.connection_map:
            self.connection_map[pattern_id] = set()

        return pattern_id

    def get(self, pattern_id: str) -> EvolvingPattern | None:
        """Get pattern by ID."""
        return self.patterns.get(pattern_id)

    def create_connection(self, pattern_id_a: str, pattern_id_b: str) -> None:
        """Create bidirectional connection between patterns.

        Connections are used for tunneling - patterns can only tunnel
        to connected patterns.

        Args:
            pattern_id_a: First pattern ID.
            pattern_id_b: Second pattern ID.
        """
        if pattern_id_a not in self.connection_map:
            self.connection_map[pattern_id_a] = set()
        if pattern_id_b not in self.connection_map:
            self.connection_map[pattern_id_b] = set()

        self.connection_map[pattern_id_a].add(pattern_id_b)
        self.connection_map[pattern_id_b].add(pattern_id_a)

    def get_connections(self, pattern_id: str) -> set[str]:
        """Get IDs of patterns connected to this pattern.

        Args:
            pattern_id: Pattern to get connections for.

        Returns:
            Set of connected pattern IDs (copy to prevent mutation).
        """
        return self.connection_map.get(pattern_id, set()).copy()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        method: Literal["jaccard", "interference"] = "jaccard",
        use_evolved: bool = True,
        coherence_exponent: float = 1.0,
    ) -> list[RetrievalResult]:
        """Retrieve patterns similar to query.

        NOTE: This method uses coherence weighting for interference method.
        For Phase 3 semantics where coherence = malleability (not accessibility),
        use retrieve_pure_similarity() instead.

        Operation order:
        1. Advance tick
        2. Decay all patterns
        3. Score patterns against query (with coherence weighting for interference)
        4. Refresh retrieved patterns proportional to score

        Args:
            query: Text to search for.
            top_k: Number of results to return.
            method: "jaccard" for bit overlap, "interference" for phase-aware.
            use_evolved: Use evolved bits (with coactivation changes).
            coherence_exponent: Exponent for coherence weighting in interference.
                - 1.0 (default): linear weighting (weight = coherence)
                - >1.0: high-coherence patterns dominate more strongly
                - <1.0: more uniform weighting
                - 0.0: uniform weighting (ignores coherence)
        """
        # Advance tick for this operation
        self.coherence_manager.advance_tick()

        # Decay all patterns first
        self.coherence_manager.decay_all(self.patterns.values())

        query_pattern = EvolvingPattern.from_text(query, dim=self.dim, k=self.k)

        if method == "jaccard":
            results = self._retrieve_jaccard(query_pattern, top_k, use_evolved)
        else:
            results = self._retrieve_interference(
                query_pattern, top_k, use_evolved, coherence_exponent
            )

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
        coherence_exponent: float = 1.0,
    ) -> list[RetrievalResult]:
        """Retrieve using phase-aware interference with coherence weighting.

        Patterns contribute to the final score proportional to their coherence.
        High-coherence patterns dominate results; low-coherence patterns fade
        but still participate (no hard cutoffs).

        Per CONTEXT.md: coherence weighting uses weight = coherence^exponent,
        with weights normalized to sum to 1.0 for proper weighted average.

        Args:
            query: Query pattern to match against.
            top_k: Number of results to return.
            use_evolved: Use evolved bits (with coactivation changes).
            coherence_exponent: Exponent for coherence weighting.
                - 1.0 (default): linear weighting
                - >1.0: high-coherence patterns dominate more
                - <1.0: more uniform weighting
                - 0.0: uniform weighting (all patterns equal)
        """
        if use_evolved:
            query_dense = query.to_dense_evolved()
        else:
            query_dense = query.to_dense()

        # First pass: compute raw interference scores and coherence weights
        raw_results = []
        total_weight = 0.0

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
                raw_score = 0.0
            else:
                interference = combined[overlap].abs().sum().item()
                max_possible = (query_dense[overlap].abs() + pattern_dense[overlap].abs()).sum().item()
                raw_score = interference / max_possible if max_possible > 0 else 0.0

            # Compute coherence weight: coherence^exponent
            # Use current coherence (already decayed in retrieve())
            coherence_weight = pattern.coherence ** coherence_exponent
            total_weight += coherence_weight

            raw_results.append({
                "pattern_id": pattern_id,
                "pattern": pattern,
                "raw_score": raw_score,
                "coherence_weight": coherence_weight,
            })

        # Second pass: apply normalized coherence weighting to scores
        # Score = raw_score * (normalized_weight)
        # where normalized_weight = coherence_weight / total_weight
        results = []
        for item in raw_results:
            if total_weight > 0:
                normalized_weight = item["coherence_weight"] / total_weight
            else:
                # Edge case: all patterns have zero coherence^exponent
                normalized_weight = 1.0 / len(raw_results) if raw_results else 0.0

            # Final score combines raw interference with coherence weighting
            # Higher coherence = higher contribution to final score
            # The weighting formula: score = raw_score * (1 + normalized_weight * (n_patterns - 1))
            # This ensures high-coherence patterns get boosted while low-coherence still contribute
            # Simpler approach: score = raw_score * coherence_weight (unnormalized)
            # Per CONTEXT: "Weights are normalized so contributions sum to 1.0"
            # Final interpretation: weighted_score = raw_score * coherence_weight
            # Then sort by weighted_score (higher coherence = higher rank for same raw_score)
            weighted_score = item["raw_score"] * item["coherence_weight"]

            results.append(RetrievalResult(
                pattern_id=item["pattern_id"],
                pattern=item["pattern"],
                score=weighted_score,
            ))

        results.sort(key=lambda r: -r.score)
        return results[:top_k]

    def retrieve_pure_similarity(
        self,
        query: str,
        top_k: int = 10,
        use_evolved: bool = True,
        recency_boost: float = 0.0,
    ) -> list[RetrievalResult]:
        """Retrieve using pure similarity - coherence does NOT affect ranking.

        This is the Phase 3 semantics where coherence = malleability, not accessibility.
        Low-coherence (crystallized) patterns are just as retrievable as high-coherence.

        Design note: The ROADMAP mentions "weighted by embeddedness/connections" but
        this was refined during phase discussion. Per CONTEXT.md: "Remove embeddedness
        from explicit weighting - connectivity naturally affects retrieval through
        network pathways" and "Pure similarity (bit overlap) as primary retrieval
        mechanism". Well-connected patterns surface naturally through the connection
        network; explicit embeddedness weighting is not needed.

        Operation order:
        1. Advance tick
        2. Decay all patterns
        3. Compute pure Jaccard similarity for each pattern
        4. Optionally blend with recency factor
        5. Refresh retrieved patterns proportional to score

        Args:
            query: Query text.
            top_k: Number of results.
            use_evolved: Use evolved bits (with acquired bits from coactivation).
            recency_boost: Optional recency factor (0-1). If >0, final score is:
                (1-boost)*similarity + boost*recency_factor
                where recency_factor = 1.0 / (1.0 + age_ticks)

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        # Advance tick for this operation
        self.coherence_manager.advance_tick()

        # Decay all patterns first
        self.coherence_manager.decay_all(self.patterns.values())

        if not self.patterns:
            return []

        # Create query pattern
        query_pattern = EvolvingPattern.from_text(query, dim=self.dim, k=self.k)
        query_bits = query_pattern.bits if use_evolved else set(query_pattern.original_bits)

        current_tick = self.coherence_manager.current_tick

        results = []
        for pattern_id, pattern in self.patterns.items():
            pattern_bits = pattern.bits if use_evolved else set(pattern.original_bits)

            # Pure Jaccard similarity
            intersection = len(query_bits & pattern_bits)
            union = len(query_bits | pattern_bits)
            similarity = intersection / union if union > 0 else 0.0

            # Optional recency blending
            if recency_boost > 0:
                age_ticks = current_tick - pattern.last_access_tick
                recency_factor = 1.0 / (1.0 + age_ticks)
                score = (1 - recency_boost) * similarity + recency_boost * recency_factor
            else:
                score = similarity

            results.append(RetrievalResult(
                pattern_id=pattern_id,
                pattern=pattern,
                score=score,
            ))

        # Sort by score descending
        results.sort(key=lambda r: -r.score)
        results = results[:top_k]

        # Refresh retrieved patterns proportional to score
        for result in results:
            self.coherence_manager.apply_refresh(result.pattern, activation_strength=result.score)

        return results

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

    def retrieve_with_surprise(
        self,
        query: str,
        top_k: int = 10,
        method: Literal["jaccard", "interference"] = "interference",
        use_evolved: bool = True,
        coherence_exponent: float = 1.0,
        boost_coefficient: float = 0.5,
        use_pure_similarity: bool = False,
    ) -> tuple[list[RetrievalResult], SurpriseResult | None]:
        """Retrieve with automatic surprise detection and re-coherence.

        This is the full coherence-aware retrieval flow:
        1. Advance tick, decay all patterns
        2. Build expectation (which patterns we expect to match)
        3. Retrieve using selected method
        4. Detect surprise (expected vs actual mismatch)
        5. Apply re-coherence to involved patterns
        6. Apply normal refresh to retrieved patterns

        Args:
            query: Query text.
            top_k: Number of results.
            method: Retrieval method ("interference" recommended for coherence effects).
                Ignored if use_pure_similarity=True.
            use_evolved: Use evolved bits (with acquired).
            coherence_exponent: Power for coherence weighting.
                Ignored if use_pure_similarity=True.
            boost_coefficient: Surprise-to-coherence conversion factor.
            use_pure_similarity: If True, use retrieve_pure_similarity semantics
                (Phase 3: coherence = malleability, not accessibility).

        Returns:
            Tuple of (results, surprise_result). surprise_result is None if no patterns.
        """
        # Step 1: Advance tick for this operation
        self.coherence_manager.advance_tick()

        # Step 2: Decay all patterns first
        self.coherence_manager.decay_all(self.patterns.values())

        # Handle empty store
        if not self.patterns:
            return [], None

        # Step 3: Create query pattern from text
        query_pattern = EvolvingPattern.from_text(query, dim=self.dim, k=self.k)
        query_bits = query_pattern.bits if use_evolved else set(query_pattern.original_bits)

        # Step 4: Build expectation using SurpriseDetector
        detector = SurpriseDetector()
        expected_bits, expected_weights = detector.build_expectation(
            query_bits, self.patterns, self.coherence_manager
        )

        # Step 5: Retrieve using appropriate method
        if use_pure_similarity:
            # Pure similarity: coherence does NOT affect ranking
            results = self._retrieve_pure_similarity_internal(query_pattern, top_k, use_evolved)
        elif method == "jaccard":
            results = self._retrieve_jaccard(query_pattern, top_k, use_evolved)
        else:
            results = self._retrieve_interference(
                query_pattern, top_k, use_evolved, coherence_exponent
            )

        # Handle no results
        if not results:
            return results, None

        # Step 6: Detect surprise comparing expected to actual top result
        actual_pattern = results[0].pattern
        actual_pattern_id = results[0].pattern_id
        retrieval_scores = [(r.pattern_id, r.score) for r in results]

        surprise_result = detector.detect(
            expected_bits,
            expected_weights,
            actual_pattern,
            actual_pattern_id,
            retrieval_scores,
        )

        # Step 7: Apply re-coherence based on surprise
        self.coherence_manager.apply_recoherence(
            self.patterns, surprise_result, boost_coefficient
        )

        # Step 8: Apply normal refresh to all retrieved patterns (same as retrieve())
        for result in results:
            self.coherence_manager.apply_refresh(result.pattern, activation_strength=result.score)

        return results, surprise_result

    def _retrieve_pure_similarity_internal(
        self,
        query: EvolvingPattern,
        top_k: int,
        use_evolved: bool,
    ) -> list[RetrievalResult]:
        """Internal pure similarity retrieval (no tick advance/decay).

        Used by retrieve_with_surprise when use_pure_similarity=True.
        """
        query_bits = query.bits if use_evolved else set(query.original_bits)

        results = []
        for pattern_id, pattern in self.patterns.items():
            pattern_bits = pattern.bits if use_evolved else set(pattern.original_bits)

            # Pure Jaccard similarity
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
