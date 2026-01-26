"""Tests for Clusterer sleep agent."""

import torch
import pytest
from engram.graph import Node, KnowledgeGraph
from engram.hdv import random_distributional, bundle
from engram.sleep import Clusterer


class TestClusterer:
    """Verify clusterer finds similar node groups."""

    def test_finds_obvious_clusters(self, dim):
        """Clusterer identifies groups of similar nodes.

        Scenario: Create two distinct clusters of nodes, verify
        clusterer finds them.
        """
        graph = KnowledgeGraph()

        # Cluster 1: Animals (same seed = identical HDVs, will cluster)
        for name in ["dog", "cat", "horse"]:
            hdv = random_distributional(dim, seed=1)  # Same seed = same HDV
            graph.add_node(Node(id=name, hdv=hdv))

        # Cluster 2: Vehicles (different seed from animals, same within group)
        for name in ["car", "truck", "bike"]:
            hdv = random_distributional(dim, seed=100)  # Different base
            graph.add_node(Node(id=name, hdv=hdv))

        # Run clusterer - use high threshold since HDVs are identical within groups
        clusterer = Clusterer(similarity_threshold=0.99)
        clusters = clusterer.find_clusters(graph)

        # Should find exactly 2 clusters
        assert len(clusters) == 2

        # Animals should be in same cluster
        animal_cluster = None
        for cluster in clusters:
            if "dog" in cluster:
                animal_cluster = cluster
                break

        assert animal_cluster is not None
        assert "cat" in animal_cluster and "horse" in animal_cluster

        # Vehicles should be in same cluster
        vehicle_cluster = None
        for cluster in clusters:
            if "car" in cluster:
                vehicle_cluster = cluster
                break

        assert vehicle_cluster is not None
        assert "truck" in vehicle_cluster and "bike" in vehicle_cluster

    def test_no_clusters_when_dissimilar(self, dim):
        """No clusters found when all nodes are dissimilar."""
        graph = KnowledgeGraph()

        # All different seeds = random, dissimilar HDVs
        for i in range(5):
            hdv = random_distributional(dim, seed=i * 100)
            graph.add_node(Node(id=f"node_{i}", hdv=hdv))

        clusterer = Clusterer(similarity_threshold=0.9)
        clusters = clusterer.find_clusters(graph)

        # Each node in its own cluster or no clusters
        for cluster in clusters:
            assert len(cluster) == 1

    def test_respects_energy_threshold(self, dim):
        """Low-energy nodes can be excluded from clustering."""
        graph = KnowledgeGraph()

        hdv = random_distributional(dim, seed=1)
        graph.add_node(Node(id="active", hdv=hdv, energy=1.0))
        graph.add_node(Node(id="dormant", hdv=hdv, energy=0.1))

        clusterer = Clusterer(similarity_threshold=0.9, min_energy=0.5)
        clusters = clusterer.find_clusters(graph)

        # Dormant should be excluded
        all_nodes = set()
        for cluster in clusters:
            all_nodes.update(cluster)

        assert "active" in all_nodes
        assert "dormant" not in all_nodes
