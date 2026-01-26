"""Curiosity drive - hardcoded motivation to resolve uncertainty."""

from engram.graph import KnowledgeGraph


class CuriosityDrive:
    """Hardcoded drive that identifies what needs resolution.

    The curiosity drive is DNA-level - it's not a node, can't be
    modified, and always operates. It:

    1. Monitors for recently-accessed uncertainty nodes (via effective strength)
    2. Prioritizes by importance (strength of related nodes)
    3. Signals when active seeking should occur

    The STRATEGIES for resolving uncertainty are emergent (learned
    nodes), but the DRIVE itself is hardcoded.
    """

    def __init__(
        self,
        effective_strength_threshold: float = 0.5,
        variance_threshold: float = 1.0,
        seek_threshold: float = 5.0,
        current_time: float = 0.0,
    ):
        """Initialize curiosity drive.

        Args:
            effective_strength_threshold: Min effective strength for uncertainty to surface.
            variance_threshold: Min variance to count as uncertain.
            seek_threshold: Priority score above which active seeking triggers.
            current_time: Current timestamp for recency calculations.
        """
        self.effective_strength_threshold = effective_strength_threshold
        self.variance_threshold = variance_threshold
        self.seek_threshold = seek_threshold
        self.current_time = current_time

    def get_pending_uncertainties(
        self,
        graph: KnowledgeGraph,
        ranked: bool = False,
        current_time: float = None,
    ) -> list[str]:
        """Get uncertainty nodes that need resolution.

        Args:
            graph: Knowledge graph to scan.
            ranked: If True, return sorted by priority (highest first).
            current_time: Current timestamp (uses self.current_time if None).

        Returns:
            List of uncertainty node IDs.
        """
        if current_time is None:
            current_time = self.current_time

        uncertainties = []

        for node_id in graph._nodes.keys():
            node = graph.get_node(node_id)
            if not node:
                continue

            # Check if this is an uncertainty node
            if not self._is_uncertainty_node(node):
                continue

            # Check effective strength threshold (includes recency)
            eff_strength = node.get_effective_strength(current_time)
            if eff_strength < self.effective_strength_threshold:
                continue

            priority = self._compute_priority(node, graph, current_time)
            uncertainties.append((node_id, priority))

        if ranked:
            uncertainties.sort(key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in uncertainties]

        return [node_id for node_id, _ in uncertainties]

    def should_seek(
        self,
        graph: KnowledgeGraph,
        uncertainty_id: str,
        current_time: float = None,
    ) -> bool:
        """Determine if active seeking should be triggered.

        Active seeking means the system should autonomously try to
        resolve this uncertainty (vs passive waiting for info).

        Args:
            graph: Knowledge graph.
            uncertainty_id: ID of uncertainty node to evaluate.
            current_time: Current timestamp (uses self.current_time if None).

        Returns:
            True if active seeking is warranted.
        """
        if current_time is None:
            current_time = self.current_time

        node = graph.get_node(uncertainty_id)
        if not node:
            return False

        priority = self._compute_priority(node, graph, current_time)
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

    def _compute_priority(
        self,
        node,
        graph: KnowledgeGraph,
        current_time: float = None,
    ) -> float:
        """Compute priority score for an uncertainty.

        Priority = effective_strength * (1 + connected_strength)

        Higher effective strength (includes recency) = more urgent
        Connected to high-strength nodes = more important

        Args:
            node: The uncertainty node.
            graph: Knowledge graph.
            current_time: Current timestamp (uses self.current_time if None).

        Returns:
            Priority score.
        """
        if current_time is None:
            current_time = self.current_time

        # Base priority from effective strength
        eff_strength = node.get_effective_strength(current_time)
        priority = eff_strength

        # Add strength of connected nodes
        connected_strength = 0.0
        sources = graph.get_sources(node.id)
        for source_id in sources:
            source = graph.get_node(source_id)
            if source:
                connected_strength += source.strength

        return priority * (1 + connected_strength)
