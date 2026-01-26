"""Emergent seeking strategies.

Strategies are NODES in the graph - they follow node physics
(decay, reinforce, merge). Unlike sleep agents, strategies are
learned from experience and can be abstracted.
"""

from .base import create_strategy_node, is_strategy_node
from .matcher import StrategyMatcher

__all__ = ["create_strategy_node", "is_strategy_node", "StrategyMatcher"]
