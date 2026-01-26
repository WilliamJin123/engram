"""Node as the universal primitive for knowledge representation.

In the node-centric architecture, everything is a node:
- Facts, concepts, entities
- Relationships (connecting other nodes)
- Types (connected to by instances)
- Agents, workflows, the graph itself
- Meta-relationships, conditionals, provenance

All semantic meaning lives in nodes. Edges are just unlabeled arrows.

v0.5.0: Unified strength model
- Synaptic weights (strength) are stored long-term
- Activation is computed transiently via spreading activation
- Recency effects come from last_accessed timestamp, not stored "energy"
"""

import math
from dataclasses import dataclass
from typing import Any

from engram.hdv import DistributionalHDV


@dataclass
class Node:
    """A node in the knowledge graph.

    Attributes:
        id: Unique identifier for this node.
        hdv: Distributional HDV encoding this node's identity/meaning.
             Source of truth for similarity and composition.
        strength: Importance/confidence score (>0). Decays slowly, grows with reinforcement.
        last_accessed: Timestamp of last access (for recency computation).
        content: Optional human-readable payload. LLM encodes this to HDV.
    """
    id: str
    hdv: DistributionalHDV
    strength: float = 1.0
    last_accessed: float = 0.0
    content: Any = None

    MIN_STRENGTH: float = 0.01

    def __post_init__(self):
        """Clamp strength to valid range."""
        self.strength = max(self.MIN_STRENGTH, self.strength)

    def get_effective_strength(
        self,
        current_time: float,
        recency_tau: float = 1.0,
    ) -> float:
        """Get strength with short-term potentiation from recent access.

        Args:
            current_time: Current timestamp.
            recency_tau: Time constant for recency decay. Larger = slower decay.

        Returns:
            Effective strength including recency boost.
        """
        time_since_access = current_time - self.last_accessed
        if time_since_access < 0:
            time_since_access = 0
        recency_boost = math.exp(-time_since_access / recency_tau)
        return self.strength * (1 + recency_boost)

    def decay(self, dt: float, decay_rate: float = 0.001) -> "Node":
        """Apply temporal decay to strength.

        Args:
            dt: Time elapsed since last decay.
            decay_rate: Strength loss per time unit (very slow).

        Returns:
            New Node with decayed strength.
        """
        new_strength = self.strength * (1.0 - decay_rate * dt)
        new_strength = max(self.MIN_STRENGTH, new_strength)

        return Node(
            id=self.id,
            hdv=self.hdv,
            strength=new_strength,
            last_accessed=self.last_accessed,
            content=self.content,
        )

    def access(self, current_time: float) -> "Node":
        """Mark node as accessed (updates timestamp).

        Args:
            current_time: Current timestamp.

        Returns:
            New Node with updated last_accessed.
        """
        return Node(
            id=self.id,
            hdv=self.hdv,
            strength=self.strength,
            last_accessed=current_time,
            content=self.content,
        )

    def reinforce(self, boost: float = 0.1) -> "Node":
        """Increase strength when node is successfully used.

        Args:
            boost: Amount of strength to add.

        Returns:
            New Node with increased strength.
        """
        return Node(
            id=self.id,
            hdv=self.hdv,
            strength=self.strength + boost,
            last_accessed=self.last_accessed,
            content=self.content,
        )

    def __repr__(self) -> str:
        content_preview = (
            f"'{self.content[:20]}...'" if isinstance(self.content, str) and len(self.content) > 20
            else repr(self.content)
        )
        return (
            f"Node(id='{self.id}', strength={self.strength:.2f}, "
            f"last_accessed={self.last_accessed:.2f}, content={content_preview})"
        )
