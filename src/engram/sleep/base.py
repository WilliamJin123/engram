"""Base class for sleep agents."""

from abc import ABC, abstractmethod
from engram.graph import KnowledgeGraph


class SleepAgent(ABC):
    """Base class for hardcoded sleep processes.

    Sleep agents operate on the graph but are not nodes themselves.
    They are part of the system's "DNA" - fundamental processes that
    exist before the graph has any content.
    """

    @abstractmethod
    def run(self, graph: KnowledgeGraph) -> dict:
        """Execute this sleep agent on the graph.

        Args:
            graph: The knowledge graph to process.

        Returns:
            Dictionary with results/statistics of the operation.
        """
        pass
