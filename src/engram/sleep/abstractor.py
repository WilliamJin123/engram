"""Abstractor sleep agent for creating concept nodes from clusters."""

import torch
from engram.graph import Node, Edge, KnowledgeGraph
from engram.hdv import DistributionalHDV
from .base import SleepAgent


class Abstractor(SleepAgent):
    """Creates abstract concept nodes from clusters of similar nodes.

    When given a cluster of similar instances, creates a parent concept
    node with HDV that is the bundle of all instances, and links
    instances to the concept.
    """

    def __init__(
        self,
        initial_concept_strength: float = 0.5,
    ):
        """Initialize abstractor.

        Args:
            initial_concept_strength: Starting strength for new concepts.
        """
        self.initial_concept_strength = initial_concept_strength

    def run(self, graph: KnowledgeGraph) -> dict:
        """Run abstractor - requires clusters to be provided externally."""
        return {"message": "Use abstract_cluster() with specific clusters"}

    def abstract_cluster(
        self,
        graph: KnowledgeGraph,
        cluster: set[str],
        concept_id: str,
        concept_content: str = None,
    ) -> dict:
        """Create a concept node from a cluster of instances.

        Args:
            graph: Knowledge graph to modify.
            cluster: Set of node IDs that form the cluster.
            concept_id: ID for the new concept node.
            concept_content: Optional human-readable content.

        Returns:
            Dictionary with creation results.
        """
        if not cluster:
            return {"error": "Empty cluster"}

        # Gather instance HDVs
        instance_hdvs = []
        for node_id in cluster:
            node = graph.get_node(node_id)
            if node:
                instance_hdvs.append(node.hdv)

        if not instance_hdvs:
            return {"error": "No valid instances"}

        # Create concept HDV by bundling instance HDVs
        concept_hdv = self._bundle_hdvs(instance_hdvs)

        # Create concept node
        concept_node = Node(
            id=concept_id,
            hdv=concept_hdv,
            strength=self.initial_concept_strength,
            last_accessed=0.0,
            content=concept_content,
        )

        graph.add_node(concept_node)

        # Create edges from instances to concept
        for node_id in cluster:
            if graph.get_node(node_id):
                graph.add_edge(Edge(source=node_id, target=concept_id))

        return {
            "concept_id": concept_id,
            "instance_count": len(cluster),
        }

    def _bundle_hdvs(self, hdvs: list[DistributionalHDV]) -> DistributionalHDV:
        """Bundle multiple HDVs into one concept HDV.

        Uses precision-weighted averaging for the mean and
        computes combined variance.
        """
        if len(hdvs) == 1:
            return hdvs[0]

        # Stack means and variances
        means = torch.stack([h.mean for h in hdvs])
        variances = torch.stack([h.variance for h in hdvs])

        # Precision-weighted mean
        eps = 1e-10
        precisions = 1.0 / (variances + eps)
        total_precision = precisions.sum(dim=0)

        weighted_mean = (precisions * means).sum(dim=0) / total_precision

        # Combined variance (inverse of total precision)
        combined_variance = 1.0 / total_precision

        # Get latest timestamps
        last_accessed = max(h.last_accessed for h in hdvs)
        last_updated = max(h.last_updated for h in hdvs)

        return DistributionalHDV(
            mean=weighted_mean,
            variance=combined_variance,
            last_accessed=last_accessed,
            last_updated=last_updated,
        )
