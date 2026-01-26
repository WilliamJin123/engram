"""Curiosity drive - hardcoded motivation to resolve uncertainty."""

from engram.graph import KnowledgeGraph


class CuriosityDrive:
    """Hardcoded drive that identifies what needs resolution.

    The curiosity drive is DNA-level - it's not a node, can't be
    modified, and always operates. It:

    1. Monitors for high-energy uncertainty nodes
    2. Prioritizes by importance (mass of related nodes)
    3. Signals when active seeking should occur

    The STRATEGIES for resolving uncertainty are emergent (learned
    nodes), but the DRIVE itself is hardcoded.
    """

    def __init__(
        self,
        energy_threshold: float = 0.5,
        variance_threshold: float = 1.0,
        seek_threshold: float = 5.0,
    ):
        """Initialize curiosity drive.

        Args:
            energy_threshold: Min energy for uncertainty to surface.
            variance_threshold: Min variance to count as uncertain.
            seek_threshold: Priority score above which active seeking triggers.
        """
        self.energy_threshold = energy_threshold
        self.variance_threshold = variance_threshold
        self.seek_threshold = seek_threshold

    def get_pending_uncertainties(
        self,
        graph: KnowledgeGraph,
        ranked: bool = False,
    ) -> list[str]:
        """Get uncertainty nodes that need resolution.

        Args:
            graph: Knowledge graph to scan.
            ranked: If True, return sorted by priority (highest first).

        Returns:
            List of uncertainty node IDs.
        """
        uncertainties = []

        for node_id in graph._nodes.keys():
            node = graph.get_node(node_id)
            if not node:
                continue

            # Check if this is an uncertainty node
            if not self._is_uncertainty_node(node):
                continue

            # Check energy threshold
            if node.energy < self.energy_threshold:
                continue

            priority = self._compute_priority(node, graph)
            uncertainties.append((node_id, priority))

        if ranked:
            uncertainties.sort(key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in uncertainties]

        return [node_id for node_id, _ in uncertainties]

    def should_seek(self, graph: KnowledgeGraph, uncertainty_id: str) -> bool:
        """Determine if active seeking should be triggered.

        Active seeking means the system should autonomously try to
        resolve this uncertainty (vs passive waiting for info).

        Args:
            graph: Knowledge graph.
            uncertainty_id: ID of uncertainty node to evaluate.

        Returns:
            True if active seeking is warranted.
        """
        node = graph.get_node(uncertainty_id)
        if not node:
            return False

        priority = self._compute_priority(node, graph)
        return priority >= self.seek_threshold

    def _is_uncertainty_node(self, node) -> bool:
        """Check if a node represents uncertainty.

        Heuristics:
        - ID starts with 'uncertainty_'
        - OR has high average variance
        """
        if node.id.startswith("uncertainty_"):
            return True

        avg_variance = node.hdv.variance.mean().item()
        return avg_variance >= self.variance_threshold

    def _compute_priority(self, node, graph: KnowledgeGraph) -> float:
        """Compute priority score for an uncertainty.

        Priority = energy * mass * (1 + connected_mass)

        Higher energy = more urgent
        Higher mass = more important
        Connected to high-mass nodes = more important
        """
        # Base priority
        priority = node.energy * node.mass

        # Add mass of connected nodes
        connected_mass = 0.0
        sources = graph.get_sources(node.id)
        for source_id in sources:
            source = graph.get_node(source_id)
            if source:
                connected_mass += source.mass

        return priority * (1 + connected_mass)
