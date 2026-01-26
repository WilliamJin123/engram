"""Edge with confidence for uncertainty-aware knowledge graphs."""

from dataclasses import dataclass


@dataclass
class Edge:
    """An edge connecting two nodes with confidence.

    Attributes:
        source: ID of the source node.
        target: ID of the target node.
        edge_type: Type of relationship (e.g., "IS_A", "HAS").
        confidence: Certainty of this relationship (0=uncertain, 1=certain).
        weight: Strength/importance of the connection.
    """
    source: str
    target: str
    edge_type: str
    confidence: float = 1.0
    weight: float = 1.0

    def __post_init__(self):
        """Clamp confidence to valid range."""
        self.confidence = max(0.0, min(1.0, self.confidence))

    @property
    def uncertainty(self) -> float:
        """Return uncertainty (1 - confidence)."""
        return 1.0 - self.confidence
