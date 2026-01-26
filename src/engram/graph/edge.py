"""Edge as unlabeled directional arrow.

In the node-centric architecture, edges are pure structural primitives.
All semantic meaning lives in nodes. An edge simply says "source points to target".
Relationships, confidence, metadata are all expressed as nodes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    """An unlabeled directional arrow connecting two nodes.

    Attributes:
        source: ID of the source node.
        target: ID of the target node.
    """
    source: str
    target: str
