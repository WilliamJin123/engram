"""Knowledge graph container for nodes and edges."""

from typing import Optional
from .node import Node
from .edge import Edge
from engram.hdv import distributional_similarity


class KnowledgeGraph:
    """Container for nodes and edges with query operations.

    Provides:
    - Node storage and retrieval by ID
    - Edge storage with source/target queries
    - HDV-based similarity search
    """

    def __init__(self):
        """Initialize empty graph."""
        self._nodes: dict[str, Node] = {}
        self._edges: set[Edge] = set()
        # Indexes for fast edge queries
        self._outgoing: dict[str, set[str]] = {}  # source -> {targets}
        self._incoming: dict[str, set[str]] = {}  # target -> {sources}

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self._nodes[node.id] = node
        if node.id not in self._outgoing:
            self._outgoing[node.id] = set()
        if node.id not in self._incoming:
            self._incoming[node.id] = set()

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID."""
        return self._nodes.get(node_id)

    def update_node(self, node: Node) -> None:
        """Update an existing node."""
        if node.id not in self._nodes:
            raise KeyError(f"Node {node.id} not in graph")
        self._nodes[node.id] = node

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            return

        # Remove edges
        for target in list(self._outgoing.get(node_id, [])):
            self._edges.discard(Edge(source=node_id, target=target))
            self._incoming[target].discard(node_id)

        for source in list(self._incoming.get(node_id, [])):
            self._edges.discard(Edge(source=source, target=node_id))
            self._outgoing[source].discard(node_id)

        # Remove from indexes
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)

        # Remove node
        del self._nodes[node_id]

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise KeyError(f"Both nodes must exist: {edge.source}, {edge.target}")

        self._edges.add(edge)
        self._outgoing[edge.source].add(edge.target)
        self._incoming[edge.target].add(edge.source)

    def get_targets(self, source_id: str) -> set[str]:
        """Get all nodes that source points to."""
        return self._outgoing.get(source_id, set()).copy()

    def get_sources(self, target_id: str) -> set[str]:
        """Get all nodes that point to target."""
        return self._incoming.get(target_id, set()).copy()

    def find_similar(
        self,
        query_hdv,
        top_k: int = 10,
        min_energy: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Find nodes most similar to query HDV.

        Args:
            query_hdv: DistributionalHDV to compare against.
            top_k: Maximum number of results.
            min_energy: Minimum energy threshold for results.

        Returns:
            List of (node_id, similarity) tuples, sorted by similarity descending.
        """
        results = []

        for node_id, node in self._nodes.items():
            if node.energy < min_energy:
                continue

            sim, _ = distributional_similarity(query_hdv, node.hdv)
            results.append((node_id, sim))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return len(self._edges)
