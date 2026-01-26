"""Clusterer sleep agent for finding similar node groups."""

from engram.graph import KnowledgeGraph
from engram.hdv import distributional_similarity
from .base import SleepAgent


class Clusterer(SleepAgent):
    """Finds clusters of similar nodes in the graph.

    Uses simple greedy clustering based on HDV similarity.
    This is a hardcoded process - part of the system's DNA.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_strength: float = 0.0,
    ):
        """Initialize clusterer.

        Args:
            similarity_threshold: Minimum similarity to be in same cluster.
            min_strength: Minimum strength for nodes to be considered.
        """
        self.similarity_threshold = similarity_threshold
        self.min_strength = min_strength

    def run(self, graph: KnowledgeGraph) -> dict:
        """Run clustering and return results."""
        clusters = self.find_clusters(graph)
        return {
            "clusters": clusters,
            "cluster_count": len(clusters),
        }

    def find_clusters(self, graph: KnowledgeGraph) -> list[set[str]]:
        """Find clusters of similar nodes.

        Uses greedy clustering: pick a node, find all similar nodes,
        form a cluster, repeat with remaining nodes.

        Args:
            graph: Knowledge graph to cluster.

        Returns:
            List of clusters, each cluster is a set of node IDs.
        """
        # Get eligible nodes
        eligible = []
        for node_id in self._get_all_node_ids(graph):
            node = graph.get_node(node_id)
            if node and node.strength >= self.min_strength:
                eligible.append(node_id)

        if not eligible:
            return []

        clustered = set()
        clusters = []

        for node_id in eligible:
            if node_id in clustered:
                continue

            # Start new cluster
            cluster = {node_id}
            node = graph.get_node(node_id)

            # Find similar nodes
            for other_id in eligible:
                if other_id in clustered or other_id == node_id:
                    continue

                other = graph.get_node(other_id)
                sim, _ = distributional_similarity(node.hdv, other.hdv)

                if sim >= self.similarity_threshold:
                    cluster.add(other_id)

            # Only add clusters with multiple members
            if len(cluster) > 1:
                clusters.append(cluster)
                clustered.update(cluster)
            else:
                # Single-node cluster
                clusters.append(cluster)
                clustered.add(node_id)

        return clusters

    def _get_all_node_ids(self, graph: KnowledgeGraph) -> list[str]:
        """Get all node IDs from graph."""
        # Access internal state - could add proper iteration to KnowledgeGraph
        return list(graph._nodes.keys())
