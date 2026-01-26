"""Strategy matching for uncertainties."""

from engram.graph import Node, KnowledgeGraph
from engram.hdv import distributional_similarity
from .base import is_strategy_node, get_strategy_type


class StrategyMatcher:
    """Matches uncertainties to appropriate seeking strategies.

    Uses HDV similarity to find strategies that apply to the
    same kind of uncertainty, then ranks by mass (trustworthiness).
    """

    def __init__(
        self,
        min_similarity: float = 0.3,
        min_energy: float = 0.1,
    ):
        """Initialize matcher.

        Args:
            min_similarity: Minimum HDV similarity to consider a match.
            min_energy: Minimum strategy energy to consider.
        """
        self.min_similarity = min_similarity
        self.min_energy = min_energy

    def find_strategies(
        self,
        graph: KnowledgeGraph,
        uncertainty: Node,
        top_k: int = 5,
    ) -> list[dict]:
        """Find strategies that match an uncertainty.

        Args:
            graph: Knowledge graph containing strategies.
            uncertainty: The uncertainty node to find strategies for.
            top_k: Maximum number of strategies to return.

        Returns:
            List of matches, sorted by score (highest first).
            Each match is dict with strategy_id, similarity, score, strategy_type.
        """
        matches = []

        for node_id in graph._nodes.keys():
            node = graph.get_node(node_id)
            if not node or not is_strategy_node(node):
                continue

            if node.energy < self.min_energy:
                continue

            # Compute similarity
            sim, _ = distributional_similarity(uncertainty.hdv, node.hdv)

            if sim < self.min_similarity:
                continue

            # Score combines similarity and mass
            score = sim * node.mass

            matches.append({
                "strategy_id": node_id,
                "similarity": sim,
                "mass": node.mass,
                "score": score,
                "strategy_type": get_strategy_type(node),
            })

        # Sort by score descending
        matches.sort(key=lambda x: x["score"], reverse=True)

        return matches[:top_k]
