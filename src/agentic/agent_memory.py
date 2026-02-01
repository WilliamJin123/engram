# src/agentic/agent_memory.py
"""High-level AgentMemory API integrating all components.

AgentMemory provides a unified interface for:
- Storing memories (text -> patterns)
- Recalling memories (retrieval with optional LLM reranking)
- Learning from interactions (coactivation)

This is the primary interface for applications using the agentic memory system.

Example usage:
    memory = AgentMemory(dim=1024, k=50)

    # Store memories
    memory.store("dogs are loyal pets")
    memory.store("cats are independent")

    # Recall and learn
    results = memory.recall("tell me about pets", top_k=5)
    memory.learn(results)  # Coactivate retrieved patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from agentic.memory_store import MemoryStore, RetrievalResult
from agentic.coactivation import coactivate, CoactivationConfig
from agentic.evolving_pattern import EvolvingPattern

if TYPE_CHECKING:
    from agentic.llm_interface import BaseLLMReranker


@dataclass
class AgentMemoryConfig:
    """Configuration for AgentMemory.

    Attributes:
        retrieval_method: Method for initial retrieval ("jaccard" or "interference").
        use_evolved_bits: Whether to use evolved bits for retrieval.
        coactivation: Configuration for coactivation learning.
    """

    retrieval_method: Literal["jaccard", "interference"] = "jaccard"
    use_evolved_bits: bool = True
    coactivation: CoactivationConfig = field(default_factory=CoactivationConfig)


class AgentMemory:
    """High-level memory interface integrating store, recall, and learn.

    This class provides a unified API for the agentic memory system,
    combining MemoryStore (storage/retrieval), coactivation (learning),
    and optional LLM reranking (semantic understanding).

    Attributes:
        dim: Dimensionality of pattern space.
        k: Number of active bits per pattern (sparsity).
        config: Configuration options.
        reranker: Optional LLM reranker for semantic recall.
    """

    def __init__(
        self,
        dim: int = 1024,
        k: int = 50,
        config: AgentMemoryConfig | None = None,
        reranker: "BaseLLMReranker | None" = None,
    ):
        """Initialize AgentMemory.

        Args:
            dim: Dimensionality of pattern space.
            k: Number of active bits per pattern.
            config: Configuration options.
            reranker: Optional LLM reranker for semantic recall.
        """
        self.dim = dim
        self.k = k
        self.config = config or AgentMemoryConfig()
        self.reranker = reranker

        self._store = MemoryStore(dim=dim, k=k)

    def store(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> str:
        """Store a memory.

        Converts text to a pattern and stores it in the memory store.

        Args:
            text: Text content to store.
            metadata: Optional metadata to attach.
            memory_id: Optional ID (auto-generated if not provided).

        Returns:
            The memory ID.
        """
        return self._store.store(
            text=text,
            metadata=metadata,
            pattern_id=memory_id,
        )

    def get(self, memory_id: str) -> EvolvingPattern | None:
        """Get a memory by ID.

        Args:
            memory_id: The memory ID.

        Returns:
            The EvolvingPattern if found, None otherwise.
        """
        return self._store.get(memory_id)

    def recall(
        self,
        query: str,
        top_k: int = 10,
        use_reranking: bool = False,
        retrieval_top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Recall memories relevant to query.

        First retrieves candidates using pattern similarity,
        then optionally reranks using LLM for semantic relevance.

        Args:
            query: Query text.
            top_k: Number of results to return.
            use_reranking: Whether to use LLM reranking.
            retrieval_top_k: Number of candidates for reranking
                (defaults to 2 * top_k if reranking, else top_k).

        Returns:
            List of RetrievalResult sorted by relevance.
        """
        # Determine how many candidates to retrieve
        if retrieval_top_k is None:
            retrieval_top_k = top_k * 2 if use_reranking else top_k

        # Initial retrieval using pattern similarity
        candidates = self._store.retrieve(
            query=query,
            top_k=retrieval_top_k,
            method=self.config.retrieval_method,
            use_evolved=self.config.use_evolved_bits,
        )

        # Optionally rerank using LLM
        if use_reranking and self.reranker is not None:
            candidates = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        else:
            candidates = candidates[:top_k]

        return candidates

    def learn(
        self,
        results: list[RetrievalResult],
        strength: float | None = None,
    ) -> None:
        """Apply coactivation learning to retrieved patterns.

        Patterns that are retrieved together share bits over time,
        creating emergent semantic similarity.

        Args:
            results: Retrieval results (typically from recall()).
            strength: Learning rate (overrides config if provided).
        """
        patterns = [r.pattern for r in results]

        if not patterns:
            return

        effective_strength = strength
        if effective_strength is None:
            effective_strength = self.config.coactivation.strength

        coactivate(
            patterns=patterns,
            strength=effective_strength,
            config=self.config.coactivation,
        )

    def set_reranker(self, reranker: "BaseLLMReranker") -> None:
        """Set the LLM reranker.

        Args:
            reranker: The reranker to use.
        """
        self.reranker = reranker

    @property
    def memory_count(self) -> int:
        """Number of stored memories."""
        return self._store.pattern_count

    def get_all_patterns(self) -> list[EvolvingPattern]:
        """Get all patterns for analysis.

        Returns:
            List of all stored EvolvingPatterns.
        """
        return self._store.get_all_patterns()

    @property
    def average_obesity(self) -> float:
        """Average pattern obesity across all memories.

        Obesity is the ratio of current bits to original bits.
        A value of 1.0 means no learning has occurred.
        Higher values indicate patterns have acquired bits.

        Returns:
            Average obesity, or 1.0 if no patterns stored.
        """
        patterns = self.get_all_patterns()
        if not patterns:
            return 1.0

        total_obesity = sum(p.obesity for p in patterns)
        return total_obesity / len(patterns)
