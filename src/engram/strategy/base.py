"""Base utilities for strategy nodes."""

from engram.graph import Node
from engram.hdv import DistributionalHDV


def create_strategy_node(
    node_id: str,
    hdv: DistributionalHDV,
    strategy_type: str,
    description: str = None,
    mass: float = 1.0,
    energy: float = 1.0,
) -> Node:
    """Create a strategy node.

    Strategy nodes are regular nodes with specific content structure.
    They follow normal node physics (decay, reinforce).

    Args:
        node_id: Unique ID for the strategy.
        hdv: HDV representing what kinds of uncertainties this applies to.
        strategy_type: Type of strategy (e.g., 'web_search', 'ask_human').
        description: Human-readable description.
        mass: Initial mass (trustworthiness).
        energy: Initial energy.

    Returns:
        A Node configured as a strategy.
    """
    content = {
        "type": "strategy",
        "strategy_type": strategy_type,
        "description": description,
    }

    return Node(
        id=node_id,
        hdv=hdv,
        mass=mass,
        energy=energy,
        content=content,
    )


def is_strategy_node(node: Node) -> bool:
    """Check if a node is a strategy node."""
    if not isinstance(node.content, dict):
        return False
    return node.content.get("type") == "strategy"


def get_strategy_type(node: Node) -> str:
    """Get the strategy type from a strategy node."""
    if not is_strategy_node(node):
        return None
    return node.content.get("strategy_type")
