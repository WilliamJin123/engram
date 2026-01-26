"""Poincare ball model for hyperbolic space operations.

The Poincare ball is a model of hyperbolic geometry where the entire hyperbolic
space is mapped to an open unit ball. Points near the boundary represent points
"far away" in hyperbolic space, and distances grow exponentially near the boundary.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from engram.graph import Node, Edge
    from engram.hdv import DistributionalHDV


def project_to_poincare(v: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Project a vector into the Poincare ball (norm < 1 - eps).

    Args:
        v: Vector to project.
        eps: Margin from boundary to maintain numerical stability.

    Returns:
        Vector inside the Poincare ball.
    """
    norm = torch.norm(v)
    if norm == 0:
        return v
    max_norm = 1.0 - eps
    if norm >= max_norm:
        return v * (max_norm / norm)
    return v


def poincare_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute hyperbolic distance between two points in the Poincare ball.

    Uses the formula: d(a,b) = 2 * arctanh(||(-a) + b|| / sqrt((1-||a||²)(1-||b||²) + <a,b>² - 2<a,b> + ||a||²||b||²))

    Simplified form using Mobius addition:
    d(a,b) = 2 * arctanh(||-a ⊕ b||)

    where ⊕ is Mobius addition.

    Args:
        a: First point in Poincare ball.
        b: Second point in Poincare ball.

    Returns:
        Hyperbolic distance (non-negative float).
    """
    # Use the standard formula for Poincare distance
    # d(a,b) = acosh(1 + 2 * ||a-b||² / ((1-||a||²)(1-||b||²)))

    diff = a - b
    norm_diff_sq = torch.dot(diff, diff)
    norm_a_sq = torch.dot(a, a)
    norm_b_sq = torch.dot(b, b)

    # Clamp to avoid numerical issues
    denom = (1 - norm_a_sq) * (1 - norm_b_sq)
    denom = torch.clamp(denom, min=1e-10)

    # acosh(1 + 2x) where x = ||a-b||² / ((1-||a||²)(1-||b||²))
    x = norm_diff_sq / denom
    # acosh(1 + 2x) = 2 * asinh(sqrt(x)) for x >= 0
    # Or we can use: acosh(y) = log(y + sqrt(y²-1))
    y = 1 + 2 * x
    # Clamp y to be at least 1 for numerical stability
    y = torch.clamp(y, min=1.0)
    dist = torch.acosh(y)

    return dist.item()


def is_valid_poincare_point(p: torch.Tensor) -> bool:
    """Check if a point is inside the Poincare ball (norm < 1).

    Args:
        p: Point to check.

    Returns:
        True if norm(p) < 1, False otherwise.
    """
    return torch.norm(p).item() < 1.0


def embed_tree(tree: dict, dim: int, seed: int | None = None) -> dict[str, torch.Tensor]:
    """Embed a tree structure into the Poincare ball.

    Places the root at/near origin and children progressively further out.
    Siblings are placed in different angular directions using orthogonal perturbations.

    Args:
        tree: Dictionary representing tree structure.
              Format: {"name": str, "children": [tree, ...]} or just "name" for leaves.
        dim: Dimension of the embedding space.
        seed: Optional random seed for reproducibility.

    Returns:
        Dictionary mapping node names to their Poincare embeddings.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    embeddings: dict[str, torch.Tensor] = {}

    def _get_random_direction() -> torch.Tensor:
        """Generate a random unit vector for angular direction."""
        v = torch.randn(dim, generator=gen)
        return v / torch.norm(v)

    def _get_orthogonal_directions(base: torch.Tensor, n: int) -> list[torch.Tensor]:
        """Generate n directions that are spread out from base direction.

        Uses orthogonal perturbations to ensure siblings are well-separated.
        """
        if n == 0:
            return []
        if n == 1:
            # Single child stays close to parent direction
            perturb = _get_random_direction() * 0.15
            d = base + perturb
            return [d / torch.norm(d)]

        # For multiple children, create maximally spread perturbations
        directions = []

        # Generate n orthogonal random vectors in the subspace orthogonal to base
        orthogonal_vecs = []
        for _ in range(n):
            v = _get_random_direction()
            # Project out the base direction
            v = v - torch.dot(v, base) * base
            # Project out previous orthogonal vectors (Gram-Schmidt)
            for prev in orthogonal_vecs:
                v = v - torch.dot(v, prev) * prev
            norm_v = torch.norm(v)
            if norm_v > 1e-6:
                v = v / norm_v
                orthogonal_vecs.append(v)
            else:
                # Fallback: use random direction if we run out of orthogonal space
                orthogonal_vecs.append(_get_random_direction())

        # Create child directions by adding orthogonal perturbations to base
        for i, orth in enumerate(orthogonal_vecs):
            # Very large spread to ensure siblings are in different angular regions
            # spread=1.5 means orthogonal component dominates, creating ~45° angles
            spread = 1.5
            d = base + spread * orth
            d = d / torch.norm(d)
            directions.append(d)

        return directions

    def _embed_node(
        node: dict | str,
        depth: int,
        direction: torch.Tensor
    ) -> None:
        """Recursively embed a node and its children."""
        # Handle string leaf format
        if isinstance(node, str):
            node = {"name": node}

        name = node["name"]
        children = node.get("children", [])

        # Compute radial distance based on depth
        # Linear growth for clearer hierarchy
        radius = 0.1 + 0.15 * depth
        radius = min(radius, 0.9)  # Stay inside ball

        # Create embedding
        embeddings[name] = direction * radius

        # Embed children with spread-out directions
        child_directions = _get_orthogonal_directions(direction, len(children))
        for child, child_dir in zip(children, child_directions):
            _embed_node(child, depth + 1, child_dir)

    # Start with root at origin direction
    root_direction = _get_random_direction()
    _embed_node(tree, 0, root_direction)
    return embeddings


def cone_query(point: torch.Tensor, embeddings: dict[str, torch.Tensor]) -> list[str]:
    """Find ancestors of a point using cone containment.

    In hyperbolic space, ancestors are points that are:
    1. Closer to the origin
    2. In a similar angular direction

    Args:
        point: Query point in Poincare ball.
        embeddings: Dictionary of name -> embedding.

    Returns:
        List of names of ancestor points.
    """
    ancestors = []
    for name, emb in embeddings.items():
        if is_ancestor(emb, point):
            ancestors.append(name)
    return ancestors


def is_ancestor(ancestor: torch.Tensor, descendant: torch.Tensor) -> bool:
    """Check if one point is an ancestor of another in hyperbolic space.

    A point A is considered an ancestor of D if:
    1. A is significantly closer to the origin than D (at least one level up)
    2. A and D are in a similar angular direction (cone containment)

    Args:
        ancestor: Potential ancestor point.
        descendant: Potential descendant point.

    Returns:
        True if ancestor is likely an ancestor of descendant.
    """
    # Get norms (distance from origin in Euclidean sense)
    ancestor_norm = torch.norm(ancestor).item()
    descendant_norm = torch.norm(descendant).item()

    # Condition 1: Ancestor must be significantly closer to origin
    # Require at least 0.1 difference to ensure we're at different tree levels
    # This prevents siblings (same level) from being considered ancestors
    min_depth_diff = 0.08
    if ancestor_norm >= descendant_norm - min_depth_diff:
        return False

    # Handle edge cases
    if ancestor_norm < 1e-6:
        # Ancestor at/near origin is ancestor of everything non-origin
        return descendant_norm >= min_depth_diff
    if descendant_norm < 1e-6:
        # Nothing can be ancestor of origin
        return False

    # Condition 2: Angular similarity (cone containment)
    # Compute cosine of angle between vectors
    cos_angle = (torch.dot(ancestor, descendant) / (ancestor_norm * descendant_norm)).item()

    # Use adaptive cone threshold based on ancestor's depth (radius)
    # Nodes closer to origin (shallow depth) have wider cones
    # Nodes farther from origin (deep) have narrower cones
    # This matches tree semantics: root is ancestor of many, but deeper parents
    # need to be in similar direction as their children
    #
    # ancestor_norm ranges from ~0.1 (root) to ~0.9 (deep leaves)
    # At norm=0.1 (root): threshold = -0.1 (very wide, accepts everything)
    # At norm=0.25 (level 1): threshold = 0.35 (moderate)
    # At norm=0.4 (level 2): threshold = 0.8 (strict)
    # This ensures siblings at same level don't get confused as ancestors
    threshold = -0.1 + 1.8 * ancestor_norm

    return cos_angle > threshold


def compute_hybrid_position(
    node: "Node",
    nodes: dict[str, "Node"],
    edges: list["Edge"],
    dim: int = 2,
    seed: int | None = None,
) -> torch.Tensor:
    """Compute hybrid hyperbolic position for a node.

    Hybrid positioning:
    - Radius: From graph structure (depth from roots) and mass (high mass = closer to origin)
    - Angle: From HDV similarity to other nodes

    Args:
        node: The node to position.
        nodes: Dictionary of all nodes by ID.
        edges: List of all edges in the graph.
        dim: Dimension of the Poincare ball (default 2 for visualization).
        seed: Optional random seed for reproducibility.

    Returns:
        Position in the Poincare ball.
    """
    from engram.graph import Node, Edge
    from engram.hdv import distributional_similarity

    gen = torch.Generator().manual_seed(seed) if seed is not None else None

    # Build adjacency: which nodes does this node point to?
    outgoing = {e.target for e in edges if e.source == node.id}
    incoming = {e.source for e in edges if e.target == node.id}

    # Compute depth: how many hops to a root (node with no outgoing edges)?
    # Use BFS from this node following outgoing edges
    depth = _compute_depth(node.id, edges, nodes)

    # Compute radius from depth and strength
    # - Deeper nodes -> larger radius
    # - Higher strength -> smaller radius (more abstract/important)
    base_radius = 0.1 + 0.15 * depth
    strength_factor = 1.0 / (1.0 + 0.1 * node.strength)  # High strength pulls toward origin
    radius = min(0.9, base_radius * strength_factor)

    # Compute angle from HDV similarity to neighbors
    # If no neighbors, use random direction based on HDV
    if outgoing or incoming:
        # Average direction toward similar neighbors
        neighbor_ids = outgoing | incoming
        direction = _compute_direction_from_neighbors(
            node, neighbor_ids, nodes, dim, gen
        )
    else:
        # Use HDV-derived direction for isolated nodes
        direction = _hdv_to_direction(node.hdv, dim, gen)

    # Combine radius and direction
    position = direction * radius

    return project_to_poincare(position)


def _compute_depth(node_id: str, edges: list, nodes: dict) -> int:
    """Compute depth of a node (hops to nearest root via outgoing edges)."""
    visited = set()
    queue = [(node_id, 0)]
    min_depth = float('inf')

    # Build outgoing adjacency
    outgoing = {}
    for e in edges:
        if e.source not in outgoing:
            outgoing[e.source] = []
        outgoing[e.source].append(e.target)

    while queue:
        current, depth = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        # If no outgoing edges, this is a root
        if current not in outgoing or not outgoing[current]:
            min_depth = min(min_depth, depth)
            continue

        for target in outgoing[current]:
            if target not in visited and target in nodes:
                queue.append((target, depth + 1))

    return min_depth if min_depth != float('inf') else 0


def _compute_direction_from_neighbors(
    node: "Node",
    neighbor_ids: set[str],
    nodes: dict[str, "Node"],
    dim: int,
    gen: torch.Generator | None,
) -> torch.Tensor:
    """Compute direction based on HDV similarity to neighbors."""
    from engram.hdv import distributional_similarity

    # Start with random base direction
    direction = torch.randn(dim, generator=gen)

    # Weight by similarity to each neighbor
    total_weight = 0.0
    weighted_direction = torch.zeros(dim)

    for nid in neighbor_ids:
        if nid not in nodes:
            continue
        neighbor = nodes[nid]
        sim, _ = distributional_similarity(node.hdv, neighbor.hdv)

        # Use neighbor's HDV to derive a direction contribution
        neighbor_dir = _hdv_to_direction(neighbor.hdv, dim, gen)
        weighted_direction += sim * neighbor_dir
        total_weight += sim

    if total_weight > 0:
        direction = weighted_direction / total_weight
    else:
        direction = _hdv_to_direction(node.hdv, dim, gen)

    # Normalize to unit vector
    norm = torch.norm(direction)
    if norm > 1e-6:
        direction = direction / norm
    else:
        direction = torch.randn(dim, generator=gen)
        direction = direction / torch.norm(direction)

    return direction


def _hdv_to_direction(hdv: "DistributionalHDV", dim: int, gen: torch.Generator | None) -> torch.Tensor:
    """Derive a direction from an HDV by projecting to lower dimension.

    This function is deterministic: same HDV always produces same direction.
    """
    # Use first `dim` components of the HDV mean as direction seed
    # This ensures same HDV -> same direction (deterministic mapping)
    if hdv.mean.shape[0] >= dim:
        raw = hdv.mean[:dim].clone()
    else:
        # Pad with zeros if HDV is too small
        raw = torch.zeros(dim)
        raw[:hdv.mean.shape[0]] = hdv.mean

    norm = torch.norm(raw)
    if norm > 1e-6:
        return raw / norm
    else:
        # Fallback: use a deterministic direction based on HDV sum
        fallback = torch.zeros(dim)
        fallback[0] = 1.0  # Default to x-axis if HDV is all zeros
        return fallback
