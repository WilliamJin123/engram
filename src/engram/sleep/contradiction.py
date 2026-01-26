"""ContradictionDetector sleep agent for finding conflicting beliefs."""

import uuid
import torch
from engram.graph import Node, Edge, KnowledgeGraph
from engram.hdv import distributional_similarity, random_distributional
from .base import SleepAgent


class ContradictionDetector(SleepAgent):
    """Detects contradictory beliefs and creates uncertainty nodes.

    Contradiction is defined as: nodes with similar topic (high structural
    similarity considering variance) but opposing content (negative mean
    correlation).

    Creates uncertainty nodes that:
    - Have high energy (demand attention)
    - Link to contradicting nodes
    - Represent "I need to resolve this"
    """

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        opposition_threshold: float = -0.3,
        min_mass: float = 0.1,
    ):
        """Initialize detector.

        Args:
            similarity_threshold: Min distributional similarity to compare.
            opposition_threshold: Max mean correlation to count as opposing.
            min_mass: Minimum mass for nodes to be considered.
        """
        self.similarity_threshold = similarity_threshold
        self.opposition_threshold = opposition_threshold
        self.min_mass = min_mass

    def run(self, graph: KnowledgeGraph) -> dict:
        """Find contradictions and create uncertainty nodes."""
        contradictions = self.find_contradictions(graph)

        uncertainty_nodes = []
        for contradiction in contradictions:
            node_id = self._create_uncertainty_node(graph, contradiction)
            if node_id:
                uncertainty_nodes.append(node_id)

        return {
            "contradictions": contradictions,
            "uncertainty_nodes": uncertainty_nodes,
        }

    def find_contradictions(self, graph: KnowledgeGraph) -> list[dict]:
        """Find pairs of contradicting nodes.

        Returns:
            List of contradiction dicts with 'nodes' and 'severity'.
        """
        contradictions = []
        node_ids = list(graph._nodes.keys())

        # Compare all pairs
        for i, id_a in enumerate(node_ids):
            node_a = graph.get_node(id_a)
            if not node_a or node_a.mass < self.min_mass:
                continue

            for id_b in node_ids[i + 1:]:
                node_b = graph.get_node(id_b)
                if not node_b or node_b.mass < self.min_mass:
                    continue

                # Check distributional similarity (are they about same topic?)
                sim, _ = distributional_similarity(node_a.hdv, node_b.hdv)

                if sim < self.similarity_threshold:
                    continue  # Different topics

                # Check mean correlation (are they opposing?)
                mean_corr = self._mean_correlation(node_a.hdv.mean, node_b.hdv.mean)

                if mean_corr <= self.opposition_threshold:
                    # Found contradiction
                    severity = abs(mean_corr) * sim  # Worse if similar AND opposing
                    contradictions.append({
                        "nodes": {id_a, id_b},
                        "severity": severity,
                        "similarity": sim,
                        "opposition": mean_corr,
                    })

        return contradictions

    def _mean_correlation(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute correlation between two mean vectors."""
        norm_a = torch.norm(a)
        norm_b = torch.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return (torch.dot(a, b) / (norm_a * norm_b)).item()

    def _create_uncertainty_node(
        self,
        graph: KnowledgeGraph,
        contradiction: dict
    ) -> str:
        """Create an uncertainty node for a contradiction."""
        node_ids = list(contradiction["nodes"])

        # Generate unique ID
        uncertainty_id = f"uncertainty_{uuid.uuid4().hex[:8]}"

        # Get first node's HDV dimension
        first_node = graph.get_node(node_ids[0])
        dim = first_node.hdv.dim

        # Create HDV with high variance (uncertain!)
        hdv = random_distributional(
            dim,
            initial_variance=1.5,  # High uncertainty
            current_time=0.0,
        )

        # Create node with high energy (demands attention)
        uncertainty_node = Node(
            id=uncertainty_id,
            hdv=hdv,
            energy=1.0,  # Maximum energy
            mass=0.1,  # Low mass (not established yet)
            content=f"Contradiction between {node_ids}",
        )

        graph.add_node(uncertainty_node)

        # Link contradicting nodes to uncertainty node
        for node_id in node_ids:
            graph.add_edge(Edge(source=node_id, target=uncertainty_id))

        return uncertainty_id
