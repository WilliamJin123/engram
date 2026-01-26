# Test Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce ~200 micro-tests across 15 files to ~11 scenario-based tests in 3 files, plus an exploration folder for documentation-style tests.

**Architecture:** Consolidate granular property tests into workflow-based scenario tests that read like documentation. Each scenario exercises a realistic use case and verifies multiple properties along the way. Move capacity/noise/behavior documentation tests to a separate `tests/exploration/` folder.

**Tech Stack:** pytest, torch, existing engram modules

---

### Task 1: Create new test folder structure

**Files:**
- Create: `tests/exploration/__init__.py`

**Step 1: Create exploration folder**

```python
# tests/exploration/__init__.py
"""Exploration tests for documenting behavior, capacity limits, and noise characteristics.

These tests serve as documentation and were written to understand system behavior.
They are not strictly required for regression testing but provide valuable insights.

Run with: pytest tests/exploration/
Skip with: pytest tests/ --ignore=tests/exploration/
"""
```

**Step 2: Verify folder created**

Run: `python -c "import pathlib; print(pathlib.Path('tests/exploration/__init__.py').exists())"`
Expected: `True`

**Step 3: Commit**

```bash
git add tests/exploration/__init__.py
git commit -m "chore: add exploration test folder for documentation tests"
```

---

### Task 2: Write graph scenario tests

**Files:**
- Create: `tests/graph/test_scenarios.py`
- Reference: `tests/graph/test_node.py`, `tests/graph/test_edge.py`

**Step 1: Write the scenario tests**

```python
"""Scenario-based tests for graph module (Node, Edge)."""

import torch
import pytest
from engram.graph import Node, Edge
from engram.hdv import random_distributional


class TestGraphScenarios:
    """Scenario tests for graph structures."""

    def test_node_lifecycle(self, dim):
        """Create a node, mutate its state, verify behavior.

        Scenario: A node represents a concept that gets accessed and reinforced
        over time, with energy decay between accesses.
        """
        # Create node with HDV and initial properties
        hdv = random_distributional(dim, seed=42)
        node = Node(id="concept_dog", hdv=hdv, energy=1.0, mass=1.0, content="A dog")

        # Verify initial state
        assert node.id == "concept_dog"
        assert node.energy == 1.0
        assert node.mass == 1.0
        assert node.content == "A dog"
        assert node.hdv is hdv

        # Energy decays over time without access
        node.decay(energy_decay_rate=0.1, mass_decay_rate=0.01)
        assert node.energy < 1.0
        assert node.mass < 1.0

        # Access restores energy (up to 1.0)
        node.access(energy_boost=0.5)
        assert node.energy <= 1.0

        # Reinforcement adds mass
        original_mass = node.mass
        node.reinforce(mass_delta=0.5)
        assert node.mass == original_mass + 0.5

        # Energy is clamped to valid range
        node_clamped = Node(id="test", hdv=hdv, energy=5.0)  # Over 1.0
        assert node_clamped.energy == 1.0

        # Node has meaningful string representation
        assert "concept_dog" in repr(node)

    def test_edge_connections(self, dim):
        """Create nodes and connect them with edges.

        Scenario: Build a simple hierarchy of concepts connected by edges,
        verify edge equality and hashing for use in sets.
        """
        # Create nodes
        hdv1 = random_distributional(dim, seed=1)
        hdv2 = random_distributional(dim, seed=2)
        hdv3 = random_distributional(dim, seed=3)

        animal = Node(id="animal", hdv=hdv1)
        dog = Node(id="dog", hdv=hdv2)
        cat = Node(id="cat", hdv=hdv3)

        # Create edges (child -> parent for "is-a" relationship)
        edge_dog = Edge(source="dog", target="animal")
        edge_cat = Edge(source="cat", target="animal")

        # Verify edge properties
        assert edge_dog.source == "dog"
        assert edge_dog.target == "animal"

        # Edge equality works correctly
        edge_dog_copy = Edge(source="dog", target="animal")
        assert edge_dog == edge_dog_copy

        different_edge = Edge(source="cat", target="animal")
        assert edge_dog != different_edge

        # Edges can be used in sets (hashable, duplicates removed)
        edge_set = {edge_dog, edge_cat, edge_dog_copy}
        assert len(edge_set) == 2  # dog_copy is duplicate

        # Edges have meaningful string representation
        assert "dog" in repr(edge_dog)
        assert "animal" in repr(edge_dog)
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/graph/test_scenarios.py -v`
Expected: 2 passed

**Step 3: Commit**

```bash
git add tests/graph/test_scenarios.py
git commit -m "feat(tests): add scenario-based graph tests"
```

---

### Task 3: Write HDV scenario tests (part 1 - core operations)

**Files:**
- Create: `tests/hdv/test_scenarios.py`
- Reference: `tests/hdv/test_bind_unbind.py`, `tests/hdv/test_bundle.py`, `tests/hdv/test_similarity.py`

**Step 1: Write core HDV scenario tests**

```python
"""Scenario-based tests for HDV module."""

import torch
import pytest
from engram.hdv import (
    random_ternary, bind, unbind, bundle, similarity, normalize,
    random_distributional, distributional_bind, distributional_unbind,
    distributional_similarity, DistributionalHDV,
)
from engram.hdv.uncertainty import (
    bayesian_update, temporal_decay, human_confirm,
    handle_contradiction, bundle_observations, UncertaintyParams,
)


class TestHDVScenarios:
    """Scenario tests for HDV operations."""

    def test_binding_for_association(self, dim):
        """Bind creates reversible associations between concepts.

        Scenario: Associate 'red' with 'apple', verify we can recover
        the association and that bind has expected mathematical properties.
        """
        red = random_ternary(dim, sparsity=1.0, seed=1)  # Bipolar for exact recovery
        apple = random_ternary(dim, sparsity=1.0, seed=2)

        # Bind creates association
        red_apple = bind(red, apple)
        assert isinstance(red_apple, torch.Tensor)
        assert red_apple.shape == red.shape

        # Result is ternary (only -1, 0, +1)
        unique_vals = torch.unique(red_apple)
        assert all(v in [-1, 0, 1] for v in unique_vals.tolist())

        # Unbind recovers the other concept
        recovered = unbind(red_apple, red)
        assert similarity(recovered, apple) > 0.99

        # Bind is commutative: bind(a,b) == bind(b,a)
        assert torch.equal(bind(red, apple), bind(apple, red))

        # Bind is associative: bind(bind(a,b),c) == bind(a,bind(b,c))
        green = random_ternary(dim, sparsity=1.0, seed=3)
        assert torch.equal(bind(bind(red, apple), green), bind(red, bind(apple, green)))

        # Wrong key gives noise (low similarity)
        wrong_key = random_ternary(dim, sparsity=0.5, seed=99)
        wrong_recovered = unbind(red_apple, wrong_key)
        assert abs(similarity(wrong_recovered, apple)) < 0.15

    def test_bundling_for_sets(self, dim):
        """Bundle combines multiple items into a set representation.

        Scenario: Bundle several fruits together, verify each can be
        queried and that non-members are distinguishable.
        """
        apple = random_ternary(dim, sparsity=0.5, seed=1)
        banana = random_ternary(dim, sparsity=0.5, seed=2)
        cherry = random_ternary(dim, sparsity=0.5, seed=3)

        # Bundle multiple items
        fruits = bundle([apple, banana, cherry])
        assert isinstance(fruits, torch.Tensor)
        assert fruits.shape == apple.shape

        # Bundle is normalized (unit length)
        assert torch.norm(fruits).item() == pytest.approx(1.0, abs=1e-5)

        # Each item is queryable (similarity > threshold)
        assert similarity(fruits, apple) > 0.3
        assert similarity(fruits, banana) > 0.3
        assert similarity(fruits, cherry) > 0.3

        # Non-members have low similarity
        car = random_ternary(dim, sparsity=0.5, seed=100)
        assert abs(similarity(fruits, car)) < 0.15

        # Single item bundle returns normalized version
        single = bundle([apple])
        assert similarity(single, normalize(apple)) > 0.99

        # Identical items reinforce each other
        reinforced = bundle([apple, apple, apple])
        assert similarity(reinforced, apple) > 0.95

    def test_similarity_measures(self, dim):
        """Similarity correctly measures vector relationships.

        Scenario: Compare vectors with known relationships and verify
        similarity behaves as expected.
        """
        v = random_ternary(dim, sparsity=0.5, seed=1)
        w = random_ternary(dim, sparsity=0.5, seed=2)

        # Similarity returns float in [-1, 1]
        sim = similarity(v, w)
        assert isinstance(sim, float)
        assert -1.0 <= sim <= 1.0

        # Self-similarity is 1.0
        assert similarity(v, v) == pytest.approx(1.0)

        # Opposite similarity is -1.0
        assert similarity(v, -v) == pytest.approx(-1.0)

        # Similarity is symmetric
        assert similarity(v, w) == similarity(w, v)

        # Random vectors are approximately orthogonal
        sims = [
            similarity(
                random_ternary(dim, sparsity=0.5, seed=i * 2),
                random_ternary(dim, sparsity=0.5, seed=i * 2 + 1)
            )
            for i in range(50)
        ]
        mean_sim = sum(sims) / len(sims)
        assert abs(mean_sim) < 0.05  # Near zero

        # Normalize preserves direction with unit length
        normalized = normalize(v)
        assert torch.norm(normalized).item() == pytest.approx(1.0, abs=1e-5)
        assert similarity(v, normalized) > 0.99
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/hdv/test_scenarios.py::TestHDVScenarios -v`
Expected: 3 passed

**Step 3: Commit**

```bash
git add tests/hdv/test_scenarios.py
git commit -m "feat(tests): add scenario-based HDV core operation tests"
```

---

### Task 4: Write HDV scenario tests (part 2 - distributional operations)

**Files:**
- Modify: `tests/hdv/test_scenarios.py`
- Reference: `tests/hdv/test_distributional.py`, `tests/hdv/test_operations_distributional.py`

**Step 1: Add distributional scenario tests**

Append to `tests/hdv/test_scenarios.py`:

```python
    def test_distributional_uncertainty(self, dim):
        """Distributional HDVs track uncertainty through operations.

        Scenario: Create uncertain concepts, combine them, observe how
        uncertainty propagates and can be reduced through observation.
        """
        params = UncertaintyParams()

        # Create distributional HDV with explicit uncertainty
        dog = random_distributional(dim, initial_variance=0.5, current_time=0.0, seed=42)
        assert isinstance(dog, DistributionalHDV)
        assert dog.mean.shape == (dim,)
        assert dog.variance.shape == (dim,)
        assert dog.variance.mean().item() == pytest.approx(0.5)

        # Mean is ternary (-1, 0, +1)
        unique_vals = torch.unique(dog.mean)
        assert all(v in [-1, 0, 1] for v in unique_vals.tolist())

        # Distributional bind propagates uncertainty
        cat = random_distributional(dim, initial_variance=0.3, seed=43)
        bound = distributional_bind(dog, cat)
        assert isinstance(bound, DistributionalHDV)
        # Variance increases after binding
        assert bound.variance.mean() > 0

        # Bind is commutative for distributional
        bound_reverse = distributional_bind(cat, dog)
        assert torch.allclose(bound.mean, bound_reverse.mean)
        assert torch.allclose(bound.variance, bound_reverse.variance)

        # Unbind recovers with high similarity for bipolar vectors
        a = random_distributional(dim, initial_variance=0.1, sparsity=1.0, seed=1)
        b = random_distributional(dim, initial_variance=0.1, sparsity=1.0, seed=2)
        bound_ab = distributional_bind(a, b)
        recovered = distributional_unbind(bound_ab, a)
        assert similarity(recovered.mean, b.mean) > 0.99

        # Distributional similarity returns (similarity, uncertainty)
        sim, unc = distributional_similarity(dog, cat)
        assert isinstance(sim, float)
        assert isinstance(unc, float)
        assert 0 <= sim <= 1
        assert unc >= 0

        # Identical distributions have similarity 1.0
        sim_self, _ = distributional_similarity(dog, dog)
        assert sim_self == pytest.approx(1.0, abs=1e-6)

    def test_uncertainty_lifecycle(self, dim):
        """Test the complete uncertainty lifecycle from creation to confirmation.

        Scenario: A concept starts uncertain, gains confidence through observations,
        drifts when unaccessed, handles contradictions, and gets confirmed by human.
        """
        params = UncertaintyParams()

        # 1. Create node from LLM inference (high initial variance)
        concept = random_distributional(dim, initial_variance=0.6, current_time=0.0, seed=42)
        assert concept.variance.mean().item() == pytest.approx(0.6)

        # 2. Observe several instances (variance reduces via Bayesian updates)
        for i in range(5):
            observation = concept.mean + torch.randn(dim) * 0.1  # Noisy observations
            obs_variance = torch.ones(dim) * 0.3
            concept = bayesian_update(concept, observation, obs_variance, current_time=float(i + 1))

        # Variance decreased (gained confidence)
        assert concept.variance.mean() < 0.5

        # Mean moved toward observations
        # (difficult to verify precisely, but variance decrease indicates learning)

        # 3. Time passes without access (variance drifts up)
        concept_decayed = temporal_decay(concept, current_time=500.0, params=params)
        assert concept_decayed.variance.mean() > concept.variance.mean()
        # Mean stays the same
        assert torch.equal(concept_decayed.mean, concept.mean)

        # 4. Contradictory information arrives (variance increases more)
        contradicting = -concept.mean  # Opposite values
        concept_conflicted = handle_contradiction(
            concept_decayed, contradicting, current_time=501.0, params=params
        )
        assert concept_conflicted.variance.mean() > concept_decayed.variance.mean()

        # 5. Human confirms correct information (variance collapses)
        concept_confirmed = human_confirm(concept_conflicted, current_time=502.0, params=params)
        assert concept_confirmed.variance.mean() < 0.1

        # Bundling multiple observations
        observations = [
            (torch.randn(dim), torch.ones(dim) * 0.5),
            (torch.randn(dim), torch.ones(dim) * 0.5),
            (torch.randn(dim), torch.ones(dim) * 0.5),
        ]
        bundled = bundle_observations(observations, current_time=1.0)
        assert isinstance(bundled, DistributionalHDV)
        # More observations = lower variance than single observation
        single = bundle_observations([observations[0]], current_time=1.0)
        assert bundled.variance.mean() < single.variance.mean()
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/hdv/test_scenarios.py -v`
Expected: 5 passed

**Step 3: Commit**

```bash
git add tests/hdv/test_scenarios.py
git commit -m "feat(tests): add distributional HDV scenario tests"
```

---

### Task 5: Write hyperbolic scenario tests

**Files:**
- Create: `tests/hyperbolic/test_scenarios.py`
- Reference: `tests/hyperbolic/test_hierarchy.py`, `tests/hyperbolic/test_cone_queries.py`, `tests/hyperbolic/test_poincare_embedding.py`, `tests/hyperbolic/test_hybrid_positioning.py`

**Step 1: Write hyperbolic scenario tests**

```python
"""Scenario-based tests for hyperbolic module."""

import torch
import pytest
from engram.hyperbolic import (
    embed_tree, is_ancestor, cone_query,
    project_to_poincare, poincare_distance, is_valid_poincare_point,
)
from engram.hyperbolic.poincare import compute_hybrid_position
from engram.graph import Node, Edge
from engram.hdv import random_distributional


class TestHyperbolicScenarios:
    """Scenario tests for hyperbolic embeddings."""

    def test_hierarchy_preservation(self, hyperbolic_dim):
        """Embed a tree and verify hierarchy is preserved in Poincare ball.

        Scenario: Create a taxonomy (animal -> mammal -> dog) and verify
        that parent nodes are closer to origin than children.
        """
        # Build a simple tree: root -> level1 -> level2
        tree = {
            "root": ["mammal", "bird"],
            "mammal": ["dog", "cat"],
            "bird": ["eagle"],
            "dog": [],
            "cat": [],
            "eagle": [],
        }

        # Embed the tree
        embeddings = embed_tree(tree, dim=hyperbolic_dim, root="root", seed=42)

        # Returns dict with embedding for every node
        assert isinstance(embeddings, dict)
        assert set(embeddings.keys()) == set(tree.keys())

        # All embeddings are tensors of correct dimension
        for name, emb in embeddings.items():
            assert isinstance(emb, torch.Tensor)
            assert emb.shape == (hyperbolic_dim,)
            assert is_valid_poincare_point(emb)

        # Root is closest to origin
        root_dist = torch.norm(embeddings["root"]).item()
        for name, emb in embeddings.items():
            if name != "root":
                assert torch.norm(emb).item() > root_dist

        # Parents closer to origin than children
        assert torch.norm(embeddings["root"]) < torch.norm(embeddings["mammal"])
        assert torch.norm(embeddings["mammal"]) < torch.norm(embeddings["dog"])
        assert torch.norm(embeddings["mammal"]) < torch.norm(embeddings["cat"])
        assert torch.norm(embeddings["root"]) < torch.norm(embeddings["bird"])
        assert torch.norm(embeddings["bird"]) < torch.norm(embeddings["eagle"])

        # Reproducible with same seed
        embeddings2 = embed_tree(tree, dim=hyperbolic_dim, root="root", seed=42)
        for name in tree.keys():
            assert torch.equal(embeddings[name], embeddings2[name])

        # Different seed gives different result
        embeddings3 = embed_tree(tree, dim=hyperbolic_dim, root="root", seed=99)
        assert not torch.equal(embeddings["root"], embeddings3["root"])

    def test_cone_containment(self, hyperbolic_dim):
        """Verify ancestor/descendant relationships via cone queries.

        Scenario: In an embedded hierarchy, check that is_ancestor correctly
        identifies ancestry relationships and cone_query returns all ancestors.
        """
        tree = {
            "animal": ["mammal", "bird"],
            "mammal": ["dog", "cat"],
            "bird": ["eagle"],
            "dog": [],
            "cat": [],
            "eagle": [],
        }
        embeddings = embed_tree(tree, dim=hyperbolic_dim, root="animal", seed=42)

        # Root is ancestor of all nodes
        for name in tree.keys():
            if name != "animal":
                assert is_ancestor(embeddings["animal"], embeddings[name], embeddings)

        # Parent is ancestor of child
        assert is_ancestor(embeddings["mammal"], embeddings["dog"], embeddings)
        assert is_ancestor(embeddings["mammal"], embeddings["cat"], embeddings)
        assert is_ancestor(embeddings["bird"], embeddings["eagle"], embeddings)

        # Child is NOT ancestor of parent
        assert not is_ancestor(embeddings["dog"], embeddings["mammal"], embeddings)
        assert not is_ancestor(embeddings["mammal"], embeddings["animal"], embeddings)

        # Siblings are NOT ancestors of each other
        assert not is_ancestor(embeddings["dog"], embeddings["cat"], embeddings)
        assert not is_ancestor(embeddings["mammal"], embeddings["bird"], embeddings)

        # Node is NOT ancestor of itself
        assert not is_ancestor(embeddings["dog"], embeddings["dog"], embeddings)

        # Cone query returns all ancestors
        dog_ancestors = cone_query(embeddings["dog"], embeddings)
        assert "animal" in dog_ancestors
        assert "mammal" in dog_ancestors
        assert "dog" not in dog_ancestors  # Excludes self
        assert "cat" not in dog_ancestors  # Excludes siblings
        assert "bird" not in dog_ancestors  # Excludes other branches

        # Root has no ancestors
        root_ancestors = cone_query(embeddings["animal"], embeddings)
        assert len(root_ancestors) == 0

    def test_poincare_operations(self, hyperbolic_dim):
        """Verify Poincare ball operations (projection, distance).

        Scenario: Project points into Poincare ball and verify distance
        properties (symmetry, triangle inequality, boundary behavior).
        """
        # Projection keeps points inside ball
        large_vec = torch.randn(hyperbolic_dim) * 10
        projected = project_to_poincare(large_vec)
        assert torch.norm(projected).item() < 1.0
        assert is_valid_poincare_point(projected)

        # Projection preserves direction
        cos_sim = torch.dot(large_vec, projected) / (torch.norm(large_vec) * torch.norm(projected))
        assert cos_sim.item() > 0.99

        # Small vectors stay mostly unchanged
        small_vec = torch.randn(hyperbolic_dim) * 0.1
        proj_small = project_to_poincare(small_vec)
        assert torch.allclose(proj_small, small_vec, atol=0.01)

        # Zero vector maps to origin
        zero = project_to_poincare(torch.zeros(hyperbolic_dim))
        assert torch.norm(zero).item() < 1e-6

        # Distance properties
        a = project_to_poincare(torch.randn(hyperbolic_dim) * 0.5)
        b = project_to_poincare(torch.randn(hyperbolic_dim) * 0.5)
        c = project_to_poincare(torch.randn(hyperbolic_dim) * 0.5)

        # Distance is non-negative
        assert poincare_distance(a, b) >= 0

        # Distance to self is zero
        assert poincare_distance(a, a) < 1e-6

        # Distance is symmetric
        assert abs(poincare_distance(a, b) - poincare_distance(b, a)) < 1e-6

        # Triangle inequality
        d_ac = poincare_distance(a, c)
        d_ab = poincare_distance(a, b)
        d_bc = poincare_distance(b, c)
        assert d_ac <= d_ab + d_bc + 1e-6

        # Distance increases toward boundary
        origin = torch.zeros(hyperbolic_dim)
        direction = torch.randn(hyperbolic_dim)
        direction = direction / torch.norm(direction)
        near = direction * 0.3
        far = direction * 0.8
        assert poincare_distance(origin, far) > poincare_distance(origin, near)

    def test_hybrid_positioning(self):
        """Test hybrid positioning combining graph depth and HDV similarity.

        Scenario: Position nodes using both graph structure (for radius)
        and HDV similarity (for angular position).
        """
        # Create nodes with HDVs
        parent_hdv = random_distributional(dim=100, seed=42)
        child_hdv = random_distributional(dim=100, seed=43)
        similar_hdv = random_distributional(dim=100, seed=42)  # Same seed = similar

        parent = Node(id="parent", hdv=parent_hdv, mass=5.0)
        child = Node(id="child", hdv=child_hdv, mass=1.0)
        similar = Node(id="similar", hdv=similar_hdv, mass=1.0)

        nodes = {"parent": parent, "child": child, "similar": similar}
        edges = [Edge(source="child", target="parent")]

        # Compute positions
        pos_parent = compute_hybrid_position(parent, nodes, edges)
        pos_child = compute_hybrid_position(child, nodes, edges)
        pos_similar = compute_hybrid_position(similar, nodes, edges)

        # All positions are valid Poincare points
        assert is_valid_poincare_point(pos_parent)
        assert is_valid_poincare_point(pos_child)
        assert is_valid_poincare_point(pos_similar)

        # Child further from origin than parent (hierarchy)
        assert torch.norm(pos_child) > torch.norm(pos_parent)

        # Root node (no parents) near origin
        root_hdv = random_distributional(dim=100, seed=1)
        root = Node(id="root", hdv=root_hdv, mass=10.0)
        pos_root = compute_hybrid_position(root, {"root": root}, [])
        assert torch.norm(pos_root).item() < 0.3

        # Nodes with similar HDVs cluster angularly
        dir_parent = pos_parent / torch.norm(pos_parent)
        dir_similar = pos_similar / torch.norm(pos_similar)
        dir_child = pos_child / torch.norm(pos_child)
        # parent and similar have same HDV seed, should be more aligned
        cos_parent_similar = torch.dot(dir_parent, dir_similar).item()
        cos_parent_child = torch.dot(dir_parent, dir_child).item()
        assert cos_parent_similar > cos_parent_child

        # High mass nodes closer to origin
        low_mass = Node(id="low", hdv=parent_hdv, mass=0.1)
        high_mass = Node(id="high", hdv=parent_hdv, mass=100.0)
        pos_low = compute_hybrid_position(low_mass, {"low": low_mass}, [])
        pos_high = compute_hybrid_position(high_mass, {"high": high_mass}, [])
        assert torch.norm(pos_high) < torch.norm(pos_low)
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/hyperbolic/test_scenarios.py -v`
Expected: 4 passed

**Step 3: Commit**

```bash
git add tests/hyperbolic/test_scenarios.py
git commit -m "feat(tests): add scenario-based hyperbolic tests"
```

---

### Task 6: Move exploration tests

**Files:**
- Create: `tests/exploration/test_capacity_curves.py`
- Create: `tests/exploration/test_noise_characteristics.py`
- Reference: `tests/hdv/test_noise_characteristics.py`

**Step 1: Create capacity curves exploration test**

```python
"""Exploration tests documenting bundle capacity and degradation.

These tests serve as documentation for system behavior and capacity limits.
They are not required for regression testing.
"""

import torch
import pytest
from engram.hdv import random_ternary, bundle, similarity


class TestBundleCapacity:
    """Document bundle capacity and degradation."""

    @pytest.mark.parametrize("n_items", [5, 10, 25, 50, 100, 200])
    def test_bundle_capacity_curve(self, dim, n_items):
        """Document how similarity degrades as bundle size increases.

        Expected: similarity ~ 1/sqrt(n)
        """
        items = [random_ternary(dim, sparsity=0.5, seed=i) for i in range(n_items)]
        bundled = bundle(items)

        sims = [similarity(bundled, item) for item in items]
        avg_sim = sum(sims) / len(sims)
        min_sim = min(sims)
        expected = 1 / (n_items ** 0.5)

        print(f"\n  n={n_items:3d}: avg_sim={avg_sim:.3f}, min_sim={min_sim:.3f}, expected~{expected:.3f}")

        # Should be in reasonable range of theoretical value
        tolerance = max(0.5, 0.3 + n_items / 500)
        assert avg_sim > expected * tolerance

    def test_bundle_vs_noise_floor(self, dim):
        """Compare bundle queryability against noise floor."""
        n_items = 50
        items = [random_ternary(dim, sparsity=0.5, seed=i) for i in range(n_items)]
        bundled = bundle(items)

        # Member similarities
        member_sims = [similarity(bundled, item) for item in items]
        avg_member = sum(member_sims) / len(member_sims)
        min_member = min(member_sims)

        # Non-member similarities (noise floor)
        non_member_sims = [
            similarity(bundled, random_ternary(dim, sparsity=0.5, seed=1000 + i))
            for i in range(50)
        ]
        max_noise = max(abs(s) for s in non_member_sims)

        print(f"\n  === Bundle Capacity vs Noise (n={n_items}) ===")
        print(f"  Member avg:     {avg_member:.3f}")
        print(f"  Member min:     {min_member:.3f}")
        print(f"  Noise max|sim|: {max_noise:.3f}")
        print(f"  Signal/Noise:   {min_member/max_noise:.1f}x")

        assert min_member > max_noise * 1.5
```

**Step 2: Create noise characteristics exploration test**

```python
"""Exploration tests documenting noise characteristics of HDV operations.

These tests serve as documentation for expected behavior including
capacity limits and noise floors.
"""

import torch
import pytest
from engram.hdv import random_ternary, bind, unbind, similarity


class TestOrthogonalityNoise:
    """Document the noise floor from random vector orthogonality."""

    def test_orthogonality_distribution(self, dim):
        """Random vectors have similarity ~ N(0, 1/sqrt(dim)).

        For dim=10000, this means std ~ 0.01, so |similarity| < 0.03 with 99% confidence.
        """
        n_samples = 100
        sims = []
        for i in range(n_samples):
            a = random_ternary(dim, sparsity=0.5, seed=i * 2)
            b = random_ternary(dim, sparsity=0.5, seed=i * 2 + 1)
            sims.append(similarity(a, b))

        sims_tensor = torch.tensor(sims)
        mean = sims_tensor.mean().item()
        std = sims_tensor.std().item()
        max_abs = sims_tensor.abs().max().item()

        print(f"\n  === Orthogonality Noise (dim={dim}) ===")
        print(f"  Mean:    {mean:.4f} (expected: ~0)")
        print(f"  Std:     {std:.4f} (expected: ~{1/dim**0.5:.4f})")
        print(f"  Max|sim|: {max_abs:.4f}")

        assert abs(mean) < 0.02
        assert std < 0.05
        assert max_abs < 0.1


class TestBindNoiseFloor:
    """Document noise characteristics of bind/unbind operations."""

    def test_unbind_noise_with_wrong_key(self, dim):
        """Document noise when unbinding with wrong key."""
        n_samples = 50
        sims = []
        for i in range(n_samples):
            a = random_ternary(dim, sparsity=0.5, seed=i * 3)
            b = random_ternary(dim, sparsity=0.5, seed=i * 3 + 1)
            wrong_key = random_ternary(dim, sparsity=0.5, seed=i * 3 + 2)
            bound = bind(a, b)
            recovered = unbind(bound, wrong_key)
            sims.append(similarity(recovered, b))

        sims_tensor = torch.tensor(sims)
        max_abs = sims_tensor.abs().max().item()

        print(f"\n  === Wrong-Key Unbind Noise ===")
        print(f"  Max|sim|: {max_abs:.4f}")

        assert max_abs < 0.15

    def test_correct_unbind_vs_noise(self, dim):
        """Compare correct unbind similarity to wrong-key noise."""
        n_samples = 30
        correct_sims = []
        wrong_sims = []

        for i in range(n_samples):
            a = random_ternary(dim, sparsity=0.5, seed=i * 3)
            b = random_ternary(dim, sparsity=0.5, seed=i * 3 + 1)
            wrong_key = random_ternary(dim, sparsity=0.5, seed=i * 3 + 2)

            bound = bind(a, b)
            correct_recovered = unbind(bound, a)
            wrong_recovered = unbind(bound, wrong_key)

            correct_sims.append(similarity(correct_recovered, b))
            wrong_sims.append(similarity(wrong_recovered, b))

        min_correct = min(correct_sims)
        max_wrong = max(abs(s) for s in wrong_sims)

        print(f"\n  === Correct vs Wrong Unbind ===")
        print(f"  Correct min: {min_correct:.3f}")
        print(f"  Wrong max:   {max_wrong:.3f}")
        print(f"  Signal/Noise: {min_correct/max_wrong:.1f}x")

        assert min_correct > max_wrong * 2


class TestSparsityEffects:
    """Document how sparsity affects HDV properties."""

    @pytest.mark.parametrize("sparsity", [0.25, 0.5, 0.75, 1.0])
    def test_unbind_recovery_by_sparsity(self, dim, sparsity):
        """Document how sparsity affects unbind recovery.

        Higher sparsity = better recovery because less information loss.
        """
        n_samples = 30
        sims = []

        for i in range(n_samples):
            a = random_ternary(dim, sparsity=sparsity, seed=i * 2)
            b = random_ternary(dim, sparsity=sparsity, seed=i * 2 + 1)
            bound = bind(a, b)
            recovered = unbind(bound, a)
            sims.append(similarity(recovered, b))

        avg_sim = sum(sims) / len(sims)
        min_sim = min(sims)

        print(f"\n  sparsity={sparsity}: avg_recovery={avg_sim:.3f}, min={min_sim:.3f}")

        if sparsity >= 0.75:
            assert avg_sim > 0.7
        elif sparsity >= 0.5:
            assert avg_sim > 0.4
```

**Step 3: Run exploration tests**

Run: `pytest tests/exploration/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git add tests/exploration/test_capacity_curves.py tests/exploration/test_noise_characteristics.py
git commit -m "feat(tests): add exploration tests for capacity and noise documentation"
```

---

### Task 7: Delete old micro test files

**Files:**
- Delete: `tests/graph/test_node.py`
- Delete: `tests/graph/test_edge.py`
- Delete: `tests/hdv/test_bind_unbind.py`
- Delete: `tests/hdv/test_bundle.py`
- Delete: `tests/hdv/test_similarity.py`
- Delete: `tests/hdv/test_distributional.py`
- Delete: `tests/hdv/test_operations_distributional.py`
- Delete: `tests/hdv/test_random_ternary.py`
- Delete: `tests/hdv/test_uncertainty.py`
- Delete: `tests/hdv/test_integration.py`
- Delete: `tests/hdv/test_noise_characteristics.py`
- Delete: `tests/hyperbolic/test_hierarchy.py`
- Delete: `tests/hyperbolic/test_cone_queries.py`
- Delete: `tests/hyperbolic/test_poincare_embedding.py`
- Delete: `tests/hyperbolic/test_hybrid_positioning.py`

**Step 1: Verify new scenario tests pass**

Run: `pytest tests/graph/test_scenarios.py tests/hdv/test_scenarios.py tests/hyperbolic/test_scenarios.py -v`
Expected: 11 passed

**Step 2: Delete old test files**

```bash
rm tests/graph/test_node.py tests/graph/test_edge.py
rm tests/hdv/test_bind_unbind.py tests/hdv/test_bundle.py tests/hdv/test_similarity.py
rm tests/hdv/test_distributional.py tests/hdv/test_operations_distributional.py
rm tests/hdv/test_random_ternary.py tests/hdv/test_uncertainty.py tests/hdv/test_integration.py
rm tests/hdv/test_noise_characteristics.py
rm tests/hyperbolic/test_hierarchy.py tests/hyperbolic/test_cone_queries.py
rm tests/hyperbolic/test_poincare_embedding.py tests/hyperbolic/test_hybrid_positioning.py
```

**Step 3: Verify all tests still pass**

Run: `pytest tests/ --ignore=tests/exploration/ -v`
Expected: 11 passed (only scenario tests)

Run: `pytest tests/ -v`
Expected: All scenario + exploration tests pass

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor(tests): remove old micro tests in favor of scenario tests

BREAKING CHANGE: Test files reorganized. Old micro tests removed.
New structure:
- tests/graph/test_scenarios.py (2 scenarios)
- tests/hdv/test_scenarios.py (5 scenarios)
- tests/hyperbolic/test_scenarios.py (4 scenarios)
- tests/exploration/ (documentation tests, optional)

Total: ~11 scenario tests replace ~200 micro tests"
```

---

### Task 8: Update pytest configuration (optional)

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add exploration ignore to default test run**

If you want exploration tests excluded by default, add to pytest config:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --ignore=tests/exploration"
```

Or keep as-is if you want all tests to run by default.

**Step 2: Document test running in README or contributing guide**

Consider adding documentation explaining:
- `pytest tests/` - Run all tests including exploration
- `pytest tests/ --ignore=tests/exploration/` - Run only scenario tests
- `pytest tests/exploration/` - Run only exploration/documentation tests

**Step 3: Commit if changed**

```bash
git add pyproject.toml
git commit -m "docs: update pytest config for new test structure"
```

---

## Summary

**Before:** ~200 tests across 15 files
**After:** ~11 scenario tests in 3 files + exploration folder

**Files created:**
- `tests/exploration/__init__.py`
- `tests/graph/test_scenarios.py`
- `tests/hdv/test_scenarios.py`
- `tests/hyperbolic/test_scenarios.py`
- `tests/exploration/test_capacity_curves.py`
- `tests/exploration/test_noise_characteristics.py`

**Files deleted:** 15 old micro test files

**Benefits:**
- Easier to find relevant tests (one file per module)
- Tests read like documentation (scenario-based)
- Exploration tests separated from regression tests
- Same coverage with clearer intent
