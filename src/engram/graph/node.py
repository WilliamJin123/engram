"""Node as the universal primitive for knowledge representation.

In the node-centric architecture, everything is a node:
- Facts, concepts, entities
- Relationships (connecting other nodes)
- Types (connected to by instances)
- Agents, workflows, the graph itself
- Meta-relationships, conditionals, provenance

All semantic meaning lives in nodes. Edges are just unlabeled arrows.
"""

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
        energy: Recency/relevance score (0-1). Decays fast without access.
        mass: Importance/confidence score (>0). Decays slowly, grows with use.
        content: Optional human-readable payload. LLM encodes this to HDV.
    """
    id: str
    hdv: DistributionalHDV
    energy: float = 1.0
    mass: float = 1.0
    content: Any = None

    # Minimum mass to prevent complete decay
    MIN_MASS: float = 0.01

    def __post_init__(self):
        """Clamp energy and mass to valid ranges."""
        self.energy = max(0.0, min(1.0, self.energy))
        self.mass = max(self.MIN_MASS, self.mass)

    def decay(
        self,
        dt: float,
        energy_decay_rate: float = 0.1,
        mass_decay_rate: float = 0.001,
    ) -> "Node":
        """Apply temporal decay to energy and mass.

        Args:
            dt: Time elapsed since last decay.
            energy_decay_rate: Energy loss per time unit.
            mass_decay_rate: Mass loss per time unit (much slower).

        Returns:
            New Node with decayed energy and mass.
        """
        new_energy = self.energy * (1.0 - energy_decay_rate * dt)
        new_energy = max(0.0, new_energy)

        new_mass = self.mass * (1.0 - mass_decay_rate * dt)
        new_mass = max(self.MIN_MASS, new_mass)

        return Node(
            id=self.id,
            hdv=self.hdv,
            energy=new_energy,
            mass=new_mass,
            content=self.content,
        )

    def access(self, energy_boost: float = 0.3) -> "Node":
        """Restore energy when node is accessed.

        Args:
            energy_boost: Amount of energy to restore.

        Returns:
            New Node with boosted energy.
        """
        new_energy = min(1.0, self.energy + energy_boost)

        return Node(
            id=self.id,
            hdv=self.hdv,
            energy=new_energy,
            mass=self.mass,
            content=self.content,
        )

    def reinforce(self, mass_boost: float = 0.1) -> "Node":
        """Add mass when node is successfully used.

        Args:
            mass_boost: Amount of mass to add.

        Returns:
            New Node with increased mass.
        """
        return Node(
            id=self.id,
            hdv=self.hdv,
            energy=self.energy,
            mass=self.mass + mass_boost,
            content=self.content,
        )

    def __repr__(self) -> str:
        content_preview = (
            f"'{self.content[:20]}...'" if isinstance(self.content, str) and len(self.content) > 20
            else repr(self.content)
        )
        return (
            f"Node(id='{self.id}', energy={self.energy:.2f}, "
            f"mass={self.mass:.2f}, content={content_preview})"
        )
