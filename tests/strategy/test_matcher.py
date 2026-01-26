"""Tests for strategy matching."""

import torch
import pytest
from engram.graph import Node, Edge, KnowledgeGraph
from engram.hdv import random_distributional
from engram.strategy import StrategyMatcher, create_strategy_node


class TestStrategyMatcher:
    """Verify strategy matching finds appropriate strategies."""

    def test_matches_strategy_by_hdv_similarity(self, dim):
        """Finds strategies with similar HDV to uncertainty.

        Scenario: Strategy for 'factual questions' should match
        an uncertainty about a factual matter.
        """
        graph = KnowledgeGraph()

        # Create a strategy for factual questions
        fact_hdv = random_distributional(dim, seed=42)
        strategy = create_strategy_node(
            node_id="strategy_web_search",
            hdv=fact_hdv,
            strategy_type="web_search",
            description="Search web for factual answers",
        )
        graph.add_node(strategy)

        # Create uncertainty with similar HDV (same seed = similar topic)
        uncertainty_hdv = random_distributional(dim, seed=42)
        uncertainty = Node(
            id="uncertainty_fact",
            hdv=uncertainty_hdv,
            content="What is the population of Tokyo?"
        )
        graph.add_node(uncertainty)

        # Match
        matcher = StrategyMatcher()
        matches = matcher.find_strategies(graph, uncertainty)

        assert len(matches) > 0
        assert matches[0]["strategy_id"] == "strategy_web_search"

    def test_prefers_high_mass_strategies(self, dim):
        """Higher mass strategies rank higher (more trusted)."""
        graph = KnowledgeGraph()

        # Two strategies for same topic, different mass
        hdv = random_distributional(dim, seed=1)

        low_mass = create_strategy_node(
            node_id="strategy_low",
            hdv=hdv,
            strategy_type="ask_random",
            mass=0.5,
        )
        high_mass = create_strategy_node(
            node_id="strategy_high",
            hdv=hdv,
            strategy_type="ask_expert",
            mass=5.0,
        )

        graph.add_node(low_mass)
        graph.add_node(high_mass)

        # Uncertainty
        uncertainty = Node(id="uncertainty", hdv=hdv)
        graph.add_node(uncertainty)

        matcher = StrategyMatcher()
        matches = matcher.find_strategies(graph, uncertainty)

        # High mass should rank first
        assert matches[0]["strategy_id"] == "strategy_high"

    def test_no_match_for_dissimilar(self, dim):
        """No strategies returned if none match."""
        graph = KnowledgeGraph()

        # Strategy for one topic
        hdv1 = random_distributional(dim, seed=1)
        strategy = create_strategy_node(
            node_id="strategy_cooking",
            hdv=hdv1,
            strategy_type="check_recipe",
        )
        graph.add_node(strategy)

        # Uncertainty about completely different topic
        hdv2 = random_distributional(dim, seed=999)
        uncertainty = Node(id="uncertainty_math", hdv=hdv2)
        graph.add_node(uncertainty)

        matcher = StrategyMatcher(min_similarity=0.5)
        matches = matcher.find_strategies(graph, uncertainty)

        assert len(matches) == 0
