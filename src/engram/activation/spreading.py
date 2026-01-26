"""Spreading activation for knowledge graph queries.

Activation is TRANSIENT - computed during queries, not stored on nodes.
This matches biological neural networks where activation is computed
each forward pass, while weights (strength) are stored.
"""

from engram.graph import KnowledgeGraph, Node


def spread_activation(
    graph: KnowledgeGraph,
    seed_ids: list[str] | str,
    current_time: float,
    max_depth: int = 3,
    spread_decay: float = 0.5,
    min_activation: float = 0.05,
    recency_tau: float = 1.0,
    mark_accessed: bool = True,
) -> dict[str, float]:
    """Spread activation from seed node(s) through the graph.

    Activation is transient - returned as a dict, not stored on nodes.
    Effective strength (including recency boost) determines how much
    activation each node contributes to its neighbors.

    Args:
        graph: Knowledge graph to spread through.
        seed_ids: Starting node ID(s). Can be single ID or list.
        current_time: Current timestamp for recency computation.
        max_depth: Maximum hops from seed.
        spread_decay: Activation multiplier per hop (0-1).
        min_activation: Stop spreading below this threshold.
        recency_tau: Time constant for recency boost.
        mark_accessed: If True, update last_accessed on visited nodes.

    Returns:
        Dict mapping node_id -> activation strength (transient).
    """
    if isinstance(seed_ids, str):
        seed_ids = [seed_ids]

    activations: dict[str, float] = {}
    frontier: list[tuple[str, float, int]] = []  # (node_id, activation, depth)

    # Initialize seeds
    for seed_id in seed_ids:
        seed = graph.get_node(seed_id)
        if seed:
            activations[seed_id] = 1.0
            frontier.append((seed_id, 1.0, 0))
            if mark_accessed:
                graph.update_node(seed_id, last_accessed=current_time)

    while frontier:
        node_id, activation, depth = frontier.pop(0)

        if depth >= max_depth:
            continue

        # Get all neighbors (both directions)
        outgoing = graph.get_outgoing(node_id)
        incoming = graph.get_incoming(node_id)
        neighbors = outgoing | incoming

        for neighbor_id in neighbors:
            neighbor = graph.get_node(neighbor_id)
            if not neighbor:
                continue

            # Compute spread based on neighbor's effective strength
            eff_strength = neighbor.get_effective_strength(current_time, recency_tau)

            # Strength factor: high strength = more activation received
            strength_factor = eff_strength / (eff_strength + 1)

            # Spread activation
            spread = activation * spread_decay * strength_factor

            if spread < min_activation:
                continue

            # Accumulate (take max if already activated)
            old_activation = activations.get(neighbor_id, 0)
            if spread > old_activation:
                activations[neighbor_id] = spread
                frontier.append((neighbor_id, spread, depth + 1))

                if mark_accessed:
                    graph.update_node(neighbor_id, last_accessed=current_time)

    return activations


def query(
    graph: KnowledgeGraph,
    query_id: str,
    current_time: float,
    top_k: int = 10,
    **spread_kwargs,
) -> list[tuple[Node, float]]:
    """Query the graph by spreading activation from a node.

    Args:
        graph: Knowledge graph.
        query_id: Node to start from.
        current_time: Current timestamp.
        top_k: Number of results to return.
        **spread_kwargs: Additional args for spread_activation.

    Returns:
        List of (node, activation) tuples, sorted by activation descending.
    """
    activations = spread_activation(
        graph, query_id, current_time, **spread_kwargs
    )

    # Sort by activation
    sorted_items = sorted(activations.items(), key=lambda x: -x[1])

    # Return top_k as (Node, activation) tuples
    results = []
    for node_id, activation in sorted_items[:top_k]:
        node = graph.get_node(node_id)
        if node:
            results.append((node, activation))

    return results


def multi_query(
    graph: KnowledgeGraph,
    query_ids: list[str],
    current_time: float,
    top_k: int = 10,
    **spread_kwargs,
) -> list[tuple[Node, float]]:
    """Query from multiple seed nodes simultaneously.

    Useful for queries like "what connects A and B?"

    Args:
        graph: Knowledge graph.
        query_ids: List of starting node IDs.
        current_time: Current timestamp.
        top_k: Number of results to return.
        **spread_kwargs: Additional args for spread_activation.

    Returns:
        List of (node, activation) tuples, sorted by activation descending.
    """
    activations = spread_activation(
        graph, query_ids, current_time, **spread_kwargs
    )

    sorted_items = sorted(activations.items(), key=lambda x: -x[1])

    results = []
    for node_id, activation in sorted_items[:top_k]:
        node = graph.get_node(node_id)
        if node:
            results.append((node, activation))

    return results
