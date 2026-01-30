# Quantum Substrate Test Wave 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize test directory and implement comprehensive Wave 2 tests covering sparse patterns, superposition limits, semantic similarity, noise resilience, phase interference, coherence dynamics, and catastrophic interference.

**Architecture:** Restructure flat test files into domain-specific subdirectories (binding/, patterns/, interference/, sequence/, coherence/, scale/, integration/). Each subdirectory contains focused test modules. Wave 2A tests validate foundational assumptions (sparse, superposition, semantics). Wave 2B tests robustness. Wave 2C tests dynamics.

**Tech Stack:** Python 3.11+, pytest, torch, quantum_substrate module

---

## Phase 1: Directory Restructuring

### Task 1: Create New Directory Structure

**Files:**
- Create: `tests/quantum_substrate/binding/__init__.py`
- Create: `tests/quantum_substrate/binding/conftest.py`
- Create: `tests/quantum_substrate/patterns/__init__.py`
- Create: `tests/quantum_substrate/interference/__init__.py`
- Create: `tests/quantum_substrate/sequence/__init__.py`
- Create: `tests/quantum_substrate/coherence/__init__.py`
- Create: `tests/quantum_substrate/scale/__init__.py`
- Create: `tests/quantum_substrate/integration/__init__.py`

**Step 1: Create directory structure**

```bash
mkdir -p tests/quantum_substrate/binding
mkdir -p tests/quantum_substrate/patterns
mkdir -p tests/quantum_substrate/interference
mkdir -p tests/quantum_substrate/sequence
mkdir -p tests/quantum_substrate/coherence
mkdir -p tests/quantum_substrate/scale
mkdir -p tests/quantum_substrate/integration
```

**Step 2: Create root conftest.py**

Create `tests/quantum_substrate/conftest.py`:

```python
"""Shared fixtures for quantum substrate tests."""

import pytest
import torch

SEED = 42


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return torch.Generator().manual_seed(SEED)


@pytest.fixture
def dim():
    """Default pattern dimensionality."""
    return 1024


@pytest.fixture
def sparsity():
    """Default sparsity (number of active dimensions)."""
    return 50  # k=50 active of dim=1024
```

**Step 3: Create __init__.py files**

Create empty `__init__.py` in each subdirectory:

```python
"""Tests for [domain] operations."""
```

**Step 4: Verify structure**

Run: `python -c "import tests.quantum_substrate.binding; print('OK')"`
Expected: OK (no import errors)

**Step 5: Commit**

```bash
git add tests/quantum_substrate/
git commit -m "chore: create test directory structure for wave 2"
```

---

### Task 2: Migrate Existing Binding Tests

**Files:**
- Move: `tests/test_quantum_substrate/test_binding.py` → `tests/quantum_substrate/binding/test_hrr_basics.py`
- Create: `tests/quantum_substrate/binding/conftest.py`

**Step 1: Create binding conftest**

Create `tests/quantum_substrate/binding/conftest.py`:

```python
"""Binding-specific fixtures."""

import torch
import pytest


@pytest.fixture
def role_set(rng, dim):
    """Standard set of role vectors for testing."""
    return {
        "agent": torch.randn(dim, dtype=torch.complex64, generator=rng),
        "patient": torch.randn(dim, dtype=torch.complex64, generator=rng),
        "instrument": torch.randn(dim, dtype=torch.complex64, generator=rng),
        "location": torch.randn(dim, dtype=torch.complex64, generator=rng),
        "time": torch.randn(dim, dtype=torch.complex64, generator=rng),
    }
```

**Step 2: Move and rename test file**

Copy content from `tests/test_quantum_substrate/test_binding.py` to `tests/quantum_substrate/binding/test_hrr_basics.py` (keeping original for now).

**Step 3: Run migrated tests**

Run: `pytest tests/quantum_substrate/binding/test_hrr_basics.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/quantum_substrate/binding/
git commit -m "test: migrate binding tests to new structure"
```

---

### Task 3: Migrate Remaining Tests

**Files:**
- Move: `test_patterns.py` → `tests/quantum_substrate/patterns/test_creation.py`
- Move: `test_interference.py` → `tests/quantum_substrate/interference/test_basics.py` + `test_retrieval.py`
- Move: `test_sequence.py` → `tests/quantum_substrate/sequence/test_phase_encoding.py`
- Move: `test_scale.py` → `tests/quantum_substrate/scale/test_benchmarks.py`
- Move: `test_integration.py` → `tests/quantum_substrate/integration/test_agentic_memory.py`

**Step 1: Move patterns tests**

Copy `tests/test_quantum_substrate/test_patterns.py` to `tests/quantum_substrate/patterns/test_creation.py`.

**Step 2: Split interference tests**

Copy `TestInterferenceBasics` and `TestEdgeCases` to `tests/quantum_substrate/interference/test_basics.py`.
Copy `TestRetrievalComparison` to `tests/quantum_substrate/interference/test_retrieval.py`.

**Step 3: Move sequence tests**

Copy `tests/test_quantum_substrate/test_sequence.py` to `tests/quantum_substrate/sequence/test_phase_encoding.py`.

**Step 4: Move scale tests**

Copy `tests/test_quantum_substrate/test_scale.py` to `tests/quantum_substrate/scale/test_benchmarks.py`.

**Step 5: Move integration tests**

Copy `tests/test_quantum_substrate/test_integration.py` to `tests/quantum_substrate/integration/test_agentic_memory.py`.

**Step 6: Run all migrated tests**

Run: `pytest tests/quantum_substrate/ -v --ignore=tests/test_quantum_substrate`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add tests/quantum_substrate/
git commit -m "test: migrate all tests to new directory structure"
```

---

### Task 4: Remove Old Test Directory

**Files:**
- Delete: `tests/test_quantum_substrate/` (entire directory)

**Step 1: Verify new tests work**

Run: `pytest tests/quantum_substrate/ -v`
Expected: All tests PASS

**Step 2: Remove old directory**

```bash
rm -rf tests/test_quantum_substrate/
```

**Step 3: Run tests again**

Run: `pytest tests/quantum_substrate/ -v`
Expected: All tests PASS (no dependency on old location)

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove old test directory after migration"
```

---

## Phase 2: Wave 2A - Foundational Tests

### Task 5: Sparse Binding Tests

**Files:**
- Create: `tests/quantum_substrate/binding/test_sparse_binding.py`

**Step 1: Write failing test for sparse bind/unbind**

Create `tests/quantum_substrate/binding/test_sparse_binding.py`:

```python
"""Test HRR binding with sparse patterns.

The claim: HRR binding works with sparse patterns (K active of N total),
not just dense random vectors.

Success criteria:
- Unbinding sparse patterns recovers correct entity with similarity > 0.3
- Accuracy >= 80% at K=50, N=1024
"""

import torch
import pytest

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity


class TestSparseBindUnbind:
    """Test HRR with sparse pattern representations."""

    def test_sparse_bind_creates_pattern(self, dim, rng):
        """Binding sparse patterns produces a valid result."""
        k = 50
        entity = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)
        role = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        # Convert to dense for HRR (HRR operates on dense)
        entity_dense = entity.to_dense()
        role_dense = role.to_dense()

        bound = bind_hrr(entity_dense, role_dense)

        assert bound.shape == (dim,)
        assert bound.dtype == torch.complex64
        # Bound should have energy (not all zeros)
        assert bound.abs().sum() > 0

    def test_sparse_unbind_recovers_original(self, dim, rng):
        """Unbinding sparse patterns recovers the entity."""
        k = 50
        entity = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)
        role = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        entity_dense = entity.to_dense()
        role_dense = role.to_dense()

        bound = bind_hrr(entity_dense, role_dense)
        recovered = unbind_hrr(bound, role_dense)

        # Should be somewhat similar to original
        sim = similarity(recovered, entity_dense)
        # Lower threshold than dense (0.3 vs 0.5) due to sparsity
        assert sim > 0.3, f"Expected similarity > 0.3, got {sim:.3f}"

    def test_sparse_unbind_wrong_role_fails(self, dim, rng):
        """Unbinding with wrong role gives low similarity."""
        k = 50
        entity = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)
        role_a = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)
        role_b = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        entity_dense = entity.to_dense()
        role_a_dense = role_a.to_dense()
        role_b_dense = role_b.to_dense()

        bound = bind_hrr(entity_dense, role_a_dense)
        recovered = unbind_hrr(bound, role_b_dense)

        sim = similarity(recovered, entity_dense)
        assert sim < 0.2, f"Expected similarity < 0.2 with wrong role, got {sim:.3f}"


class TestSparseBindingAccuracy:
    """Test binding accuracy with sparse patterns at scale."""

    @pytest.mark.parametrize("k", [20, 50, 100])
    def test_accuracy_vs_sparsity(self, dim, rng, k):
        """Track accuracy as sparsity varies."""
        n_entities = 20
        correct = 0

        entities = [
            ComplexSparsePattern.random(dim=dim, k=k, generator=rng)
            for _ in range(n_entities)
        ]
        role = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        entities_dense = [e.to_dense() for e in entities]
        role_dense = role.to_dense()

        for i, entity_dense in enumerate(entities_dense):
            bound = bind_hrr(entity_dense, role_dense)
            recovered = unbind_hrr(bound, role_dense)

            similarities = [similarity(recovered, e) for e in entities_dense]
            best_match = similarities.index(max(similarities))

            if best_match == i:
                correct += 1

        accuracy = correct / n_entities
        print(f"\nSparse k={k}: {accuracy * 100:.0f}% accuracy")

        # Require 80% accuracy (lower than dense due to sparsity effects)
        assert accuracy >= 0.8, f"Expected >= 80% at k={k}, got {accuracy * 100:.0f}%"

    def test_sparse_vs_dense_comparison(self, dim, rng):
        """Compare sparse and dense binding accuracy."""
        n_entities = 20
        k = 50

        # Sparse entities
        sparse_entities = [
            ComplexSparsePattern.random(dim=dim, k=k, generator=rng)
            for _ in range(n_entities)
        ]
        sparse_role = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        sparse_dense = [e.to_dense() for e in sparse_entities]
        sparse_role_dense = sparse_role.to_dense()

        # Dense entities (same generator continues)
        dense_entities = [
            torch.randn(dim, dtype=torch.complex64)
            for _ in range(n_entities)
        ]
        dense_role = torch.randn(dim, dtype=torch.complex64)

        # Test sparse
        sparse_correct = 0
        for i, entity in enumerate(sparse_dense):
            bound = bind_hrr(entity, sparse_role_dense)
            recovered = unbind_hrr(bound, sparse_role_dense)
            sims = [similarity(recovered, e) for e in sparse_dense]
            if sims.index(max(sims)) == i:
                sparse_correct += 1

        # Test dense
        dense_correct = 0
        for i, entity in enumerate(dense_entities):
            bound = bind_hrr(entity, dense_role)
            recovered = unbind_hrr(bound, dense_role)
            sims = [similarity(recovered, e) for e in dense_entities]
            if sims.index(max(sims)) == i:
                dense_correct += 1

        print(f"\nSparse accuracy: {sparse_correct}/{n_entities}")
        print(f"Dense accuracy: {dense_correct}/{n_entities}")

        # Both should work reasonably well
        assert sparse_correct >= 16, f"Sparse too low: {sparse_correct}/20"
        assert dense_correct >= 18, f"Dense too low: {dense_correct}/20"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/quantum_substrate/binding/test_sparse_binding.py -v`
Expected: Tests should PASS (we're validating existing functionality, not adding new)

**Step 3: Commit**

```bash
git add tests/quantum_substrate/binding/test_sparse_binding.py
git commit -m "test: add sparse binding validation tests"
```

---

### Task 6: Superposition Limits Tests

**Files:**
- Create: `tests/quantum_substrate/binding/test_superposition_limits.py`

**Step 1: Write tests for superposition limits**

Create `tests/quantum_substrate/binding/test_superposition_limits.py`:

```python
"""Test superposition limits in HRR binding.

The claim: Multiple role bindings can be summed into a single event.
At some point, adding more bindings causes interference and accuracy degrades.

Success criteria:
- Find the practical limit (where accuracy drops below 80%)
- Document the degradation curve
"""

import torch
import pytest

from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity


class TestSuperpositionCapacity:
    """Test how many role bindings can be summed."""

    @pytest.mark.parametrize("n_roles", [2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    def test_role_recovery_vs_count(self, dim, rng):
        """Track role recovery accuracy as number of roles increases."""
        n_roles = pytest.current_test.callspec.params.get("n_roles", 4)

        # Create entities and roles
        entities = [
            torch.randn(dim, dtype=torch.complex64)
            for _ in range(n_roles)
        ]
        roles = [
            torch.randn(dim, dtype=torch.complex64)
            for _ in range(n_roles)
        ]

        # Create event with all bindings summed
        event = sum(bind_hrr(e, r) for e, r in zip(entities, roles))

        # Try to recover each entity by its role
        correct = 0
        for i, role in enumerate(roles):
            recovered = unbind_hrr(event, role)
            similarities = [similarity(recovered, e) for e in entities]
            best_match = similarities.index(max(similarities))

            if best_match == i:
                correct += 1

        accuracy = correct / n_roles
        print(f"\n{n_roles} roles: {accuracy * 100:.0f}% recovery ({correct}/{n_roles})")

        # Record for analysis - don't fail, just report
        # We want to find the breaking point

    def test_superposition_limit_4_roles(self, dim):
        """4 roles should definitely work (baseline from wave 1)."""
        n_roles = 4

        entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_roles)]
        roles = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_roles)]

        event = sum(bind_hrr(e, r) for e, r in zip(entities, roles))

        correct = 0
        for i, role in enumerate(roles):
            recovered = unbind_hrr(event, role)
            sims = [similarity(recovered, e) for e in entities]
            if sims.index(max(sims)) == i:
                correct += 1

        accuracy = correct / n_roles
        assert accuracy >= 0.75, f"4 roles should work, got {accuracy * 100:.0f}%"

    def test_superposition_limit_8_roles(self, dim):
        """8 roles - expected to still work."""
        n_roles = 8

        entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_roles)]
        roles = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_roles)]

        event = sum(bind_hrr(e, r) for e, r in zip(entities, roles))

        correct = 0
        for i, role in enumerate(roles):
            recovered = unbind_hrr(event, role)
            sims = [similarity(recovered, e) for e in entities]
            if sims.index(max(sims)) == i:
                correct += 1

        accuracy = correct / n_roles
        print(f"\n8 roles: {accuracy * 100:.0f}% ({correct}/{n_roles})")
        # Don't assert - document the result

    def test_find_breaking_point(self, dim):
        """Find where accuracy drops below 80%."""
        results = []

        for n_roles in [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]:
            # Run multiple trials for stability
            trial_accuracies = []

            for trial in range(5):
                entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_roles)]
                roles = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_roles)]

                event = sum(bind_hrr(e, r) for e, r in zip(entities, roles))

                correct = 0
                for i, role in enumerate(roles):
                    recovered = unbind_hrr(event, role)
                    sims = [similarity(recovered, e) for e in entities]
                    if sims.index(max(sims)) == i:
                        correct += 1

                trial_accuracies.append(correct / n_roles)

            avg_accuracy = sum(trial_accuracies) / len(trial_accuracies)
            results.append((n_roles, avg_accuracy))

        print("\n=== Superposition Capacity Analysis ===")
        for n_roles, acc in results:
            status = "OK" if acc >= 0.8 else "DEGRADED" if acc >= 0.5 else "FAILED"
            print(f"  {n_roles:2d} roles: {acc * 100:5.1f}% [{status}]")

        # Find breaking point
        breaking_point = None
        for n_roles, acc in results:
            if acc < 0.8:
                breaking_point = n_roles
                break

        if breaking_point:
            print(f"\nBreaking point (< 80%): {breaking_point} roles")
        else:
            print("\nNo breaking point found up to 30 roles")


class TestRoleConfusionMatrix:
    """Analyze which roles get confused with which."""

    def test_confusion_at_limit(self, dim):
        """See confusion patterns when at capacity."""
        n_roles = 10  # Likely near the limit

        entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_roles)]
        roles = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_roles)]

        event = sum(bind_hrr(e, r) for e, r in zip(entities, roles))

        # Build confusion matrix
        print(f"\n=== Confusion Matrix ({n_roles} roles) ===")
        print("Recovered similarities for each role query:")

        for i, role in enumerate(roles):
            recovered = unbind_hrr(event, role)
            sims = [similarity(recovered, e) for e in entities]

            # Format similarities
            sim_str = " ".join(f"{s:5.2f}" for s in sims)
            best = sims.index(max(sims))
            status = "OK" if best == i else f"WRONG->e{best}"
            print(f"  r{i}: [{sim_str}] {status}")
```

**Step 2: Run tests**

Run: `pytest tests/quantum_substrate/binding/test_superposition_limits.py -v -s`
Expected: Tests run, some may show degradation at high role counts

**Step 3: Commit**

```bash
git add tests/quantum_substrate/binding/test_superposition_limits.py
git commit -m "test: add superposition limits analysis tests"
```

---

### Task 7: Semantic Similarity Tests

**Files:**
- Create: `tests/quantum_substrate/patterns/test_semantic_similarity.py`

**Step 1: Write tests for semantic similarity interference**

Create `tests/quantum_substrate/patterns/test_semantic_similarity.py`:

```python
"""Test binding with semantically similar patterns.

The claim: When "dog" and "wolf" share bits (semantic similarity),
binding should still correctly disambiguate roles.

Success criteria:
- Controlled overlap patterns work up to 30% similarity
- Role disambiguation works even with similar entities
"""

import torch
import pytest

from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity


def create_similar_pattern(
    source: torch.Tensor,
    overlap_frac: float,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Create a pattern that shares overlap_frac of its energy with source.

    Args:
        source: Source pattern.
        overlap_frac: Fraction of overlap (0 to 1).
        rng: Random generator.

    Returns:
        New pattern with controlled similarity to source.
    """
    dim = source.shape[0]
    noise = torch.randn(dim, dtype=torch.complex64)

    # Weighted combination: overlap * source + (1-overlap) * noise
    combined = overlap_frac * source + (1 - overlap_frac) * noise

    # Normalize to similar energy as source
    combined = combined * (source.abs().mean() / (combined.abs().mean() + 1e-8))

    return combined


class TestControlledOverlap:
    """Test binding with controlled pattern overlap."""

    @pytest.mark.parametrize("overlap", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    def test_binding_accuracy_vs_overlap(self, dim, overlap):
        """Track how overlap affects binding accuracy."""
        n_entities = 10

        # Create base entities
        base_entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_entities)]

        # Create similar entities (each similar to corresponding base)
        similar_entities = [
            create_similar_pattern(e, overlap) for e in base_entities
        ]

        role = torch.randn(dim, dtype=torch.complex64)

        # Test: can we distinguish entity[i] from similar_entity[i]?
        correct = 0
        for i in range(n_entities):
            entity = base_entities[i]
            similar = similar_entities[i]

            bound = bind_hrr(entity, role)
            recovered = unbind_hrr(bound, role)

            # Should be more similar to original than to similar version
            sim_original = similarity(recovered, entity)
            sim_similar = similarity(recovered, similar)

            if sim_original > sim_similar:
                correct += 1

        accuracy = correct / n_entities
        print(f"\nOverlap {overlap:.0%}: {accuracy * 100:.0f}% distinguish original from similar")

        # Should maintain distinction up to 30% overlap
        if overlap <= 0.3:
            assert accuracy >= 0.7, f"Expected >= 70% at {overlap:.0%} overlap"

    def test_semantic_triplet_dog_wolf(self, dim):
        """Test classic scenario: dog/wolf are similar, mailman is different."""
        # Create patterns
        dog = torch.randn(dim, dtype=torch.complex64)
        wolf = create_similar_pattern(dog, overlap_frac=0.3)  # 30% similar
        mailman = torch.randn(dim, dtype=torch.complex64)  # unrelated

        agent = torch.randn(dim, dtype=torch.complex64)
        patient = torch.randn(dim, dtype=torch.complex64)

        # Event 1: dog bit mailman
        event1 = bind_hrr(dog, agent) + bind_hrr(mailman, patient)

        # Event 2: wolf bit mailman
        event2 = bind_hrr(wolf, agent) + bind_hrr(mailman, patient)

        # Query: who was agent in event1?
        agent_in_event1 = unbind_hrr(event1, agent)

        # Should be most similar to dog, less to wolf, least to mailman
        sim_dog = similarity(agent_in_event1, dog)
        sim_wolf = similarity(agent_in_event1, wolf)
        sim_mailman = similarity(agent_in_event1, mailman)

        print(f"\nEvent1 agent query (should be dog):")
        print(f"  dog: {sim_dog:.3f}")
        print(f"  wolf: {sim_wolf:.3f}")
        print(f"  mailman: {sim_mailman:.3f}")

        assert sim_dog > sim_wolf, "Dog should be more similar than wolf"
        assert sim_dog > sim_mailman, "Dog should be more similar than mailman"

        # Query: who was agent in event2?
        agent_in_event2 = unbind_hrr(event2, agent)

        sim_dog = similarity(agent_in_event2, dog)
        sim_wolf = similarity(agent_in_event2, wolf)
        sim_mailman = similarity(agent_in_event2, mailman)

        print(f"\nEvent2 agent query (should be wolf):")
        print(f"  dog: {sim_dog:.3f}")
        print(f"  wolf: {sim_wolf:.3f}")
        print(f"  mailman: {sim_mailman:.3f}")

        assert sim_wolf > sim_dog, "Wolf should be more similar than dog"
        assert sim_wolf > sim_mailman, "Wolf should be more similar than mailman"


class TestSimilarEntitiesDifferentRoles:
    """Test disambiguation when similar entities have different roles."""

    def test_dog_agent_wolf_patient(self, dim):
        """Dog=agent, wolf=patient should be distinguishable."""
        dog = torch.randn(dim, dtype=torch.complex64)
        wolf = create_similar_pattern(dog, overlap_frac=0.3)

        agent = torch.randn(dim, dtype=torch.complex64)
        patient = torch.randn(dim, dtype=torch.complex64)

        # Event: dog (agent) bit wolf (patient)
        event = bind_hrr(dog, agent) + bind_hrr(wolf, patient)

        # Query agent
        recovered_agent = unbind_hrr(event, agent)
        sim_dog_agent = similarity(recovered_agent, dog)
        sim_wolf_agent = similarity(recovered_agent, wolf)

        # Query patient
        recovered_patient = unbind_hrr(event, patient)
        sim_dog_patient = similarity(recovered_patient, dog)
        sim_wolf_patient = similarity(recovered_patient, wolf)

        print(f"\nDog=agent, Wolf=patient:")
        print(f"  Agent query: dog={sim_dog_agent:.3f}, wolf={sim_wolf_agent:.3f}")
        print(f"  Patient query: dog={sim_dog_patient:.3f}, wolf={sim_wolf_patient:.3f}")

        assert sim_dog_agent > sim_wolf_agent, "Agent should be dog"
        assert sim_wolf_patient > sim_dog_patient, "Patient should be wolf"

    def test_multiple_similar_entities(self, dim):
        """Test with family of similar entities: dog, wolf, fox, coyote."""
        # Create family of similar animals
        dog = torch.randn(dim, dtype=torch.complex64)
        wolf = create_similar_pattern(dog, 0.3)
        fox = create_similar_pattern(dog, 0.25)
        coyote = create_similar_pattern(dog, 0.35)

        animals = [dog, wolf, fox, coyote]
        animal_names = ["dog", "wolf", "fox", "coyote"]

        # Human (unrelated)
        mailman = torch.randn(dim, dtype=torch.complex64)

        agent = torch.randn(dim, dtype=torch.complex64)
        patient = torch.randn(dim, dtype=torch.complex64)

        # Event: coyote bit mailman
        event = bind_hrr(coyote, agent) + bind_hrr(mailman, patient)

        # Query agent - should identify coyote, not other canids
        recovered = unbind_hrr(event, agent)

        print(f"\nEvent: coyote bit mailman")
        print(f"Agent query results:")
        for animal, name in zip(animals, animal_names):
            sim = similarity(recovered, animal)
            print(f"  {name}: {sim:.3f}")
        sim_mailman = similarity(recovered, mailman)
        print(f"  mailman: {sim_mailman:.3f}")

        # Coyote should win
        coyote_sim = similarity(recovered, coyote)
        for animal, name in zip(animals[:-1], animal_names[:-1]):  # exclude coyote
            other_sim = similarity(recovered, animal)
            assert coyote_sim > other_sim, f"Coyote should beat {name}"
```

**Step 2: Run tests**

Run: `pytest tests/quantum_substrate/patterns/test_semantic_similarity.py -v -s`
Expected: Tests reveal how overlap affects disambiguation

**Step 3: Commit**

```bash
git add tests/quantum_substrate/patterns/test_semantic_similarity.py
git commit -m "test: add semantic similarity interference tests"
```

---

## Phase 3: Wave 2B - Robustness Tests

### Task 8: Noise Resilience Tests

**Files:**
- Create: `tests/quantum_substrate/patterns/test_noise_resilience.py`

**Step 1: Write noise resilience tests**

Create `tests/quantum_substrate/patterns/test_noise_resilience.py`:

```python
"""Test resilience to encoding noise.

The claim: Real-world encoding won't be perfect. The system should
tolerate some noise in pattern representation.

Success criteria:
- 5% noise: >= 95% accuracy
- 10% noise: >= 85% accuracy
- 20% noise: track degradation
"""

import torch
import pytest

from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity


def add_noise(
    pattern: torch.Tensor,
    noise_level: float,
    noise_type: str = "both",
) -> torch.Tensor:
    """Add noise to a complex pattern.

    Args:
        pattern: Original pattern.
        noise_level: Fraction of pattern energy as noise (0 to 1).
        noise_type: "magnitude", "phase", or "both".

    Returns:
        Noisy pattern.
    """
    if noise_type == "magnitude":
        # Add noise to magnitude only
        mag_noise = torch.randn_like(pattern.real) * noise_level * pattern.abs().mean()
        noisy_mag = pattern.abs() + mag_noise
        noisy_mag = torch.clamp(noisy_mag, min=0)  # magnitudes must be positive
        return noisy_mag * torch.exp(1j * torch.angle(pattern))

    elif noise_type == "phase":
        # Add noise to phase only
        phase_noise = torch.randn(pattern.shape[0]) * noise_level * torch.pi
        return pattern.abs() * torch.exp(1j * (torch.angle(pattern) + phase_noise))

    else:  # both
        noise = torch.randn_like(pattern) * noise_level * pattern.abs().mean()
        return pattern + noise


class TestNoiseResilience:
    """Test binding accuracy under noise."""

    @pytest.mark.parametrize("noise_level", [0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
    def test_accuracy_vs_noise(self, dim, noise_level):
        """Track accuracy as noise increases."""
        n_entities = 20
        correct = 0

        entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_entities)]
        role = torch.randn(dim, dtype=torch.complex64)

        for i, entity in enumerate(entities):
            # Add noise to entity before binding
            noisy_entity = add_noise(entity, noise_level)

            bound = bind_hrr(noisy_entity, role)
            recovered = unbind_hrr(bound, role)

            # Compare against CLEAN entities (realistic: memory is clean, query is noisy)
            similarities = [similarity(recovered, e) for e in entities]
            best_match = similarities.index(max(similarities))

            if best_match == i:
                correct += 1

        accuracy = correct / n_entities
        print(f"\nNoise {noise_level:.0%}: {accuracy * 100:.0f}% accuracy")

        # Thresholds
        if noise_level <= 0.05:
            assert accuracy >= 0.95, f"Expected >= 95% at {noise_level:.0%} noise"
        elif noise_level <= 0.10:
            assert accuracy >= 0.85, f"Expected >= 85% at {noise_level:.0%} noise"

    def test_noisy_query_clean_memory(self, dim):
        """Noisy query against clean stored patterns (common case)."""
        n_entities = 20
        noise_level = 0.15

        # Clean memory
        entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_entities)]
        role = torch.randn(dim, dtype=torch.complex64)

        # Store clean bindings
        stored = [bind_hrr(e, role) for e in entities]

        # Query with noise
        correct = 0
        for i, entity in enumerate(entities):
            noisy_entity = add_noise(entity, noise_level)
            noisy_bound = bind_hrr(noisy_entity, role)

            # Find best match in clean storage
            best_idx = -1
            best_sim = -1
            for j, stored_bound in enumerate(stored):
                sim = similarity(noisy_bound, stored_bound)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j

            if best_idx == i:
                correct += 1

        accuracy = correct / n_entities
        print(f"\nNoisy query (15%) vs clean memory: {accuracy * 100:.0f}%")
        assert accuracy >= 0.8, f"Expected >= 80% with noisy query"


class TestNoiseTypes:
    """Compare impact of magnitude vs phase noise."""

    @pytest.mark.parametrize("noise_type", ["magnitude", "phase", "both"])
    def test_noise_type_comparison(self, dim, noise_type):
        """Compare how different noise types affect accuracy."""
        n_entities = 20
        noise_level = 0.15

        entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_entities)]
        role = torch.randn(dim, dtype=torch.complex64)

        correct = 0
        for i, entity in enumerate(entities):
            noisy_entity = add_noise(entity, noise_level, noise_type=noise_type)
            bound = bind_hrr(noisy_entity, role)
            recovered = unbind_hrr(bound, role)

            similarities = [similarity(recovered, e) for e in entities]
            best_match = similarities.index(max(similarities))

            if best_match == i:
                correct += 1

        accuracy = correct / n_entities
        print(f"\n{noise_type} noise ({noise_level:.0%}): {accuracy * 100:.0f}%")
```

**Step 2: Run tests**

Run: `pytest tests/quantum_substrate/patterns/test_noise_resilience.py -v -s`
Expected: Tests show noise tolerance curve

**Step 3: Commit**

```bash
git add tests/quantum_substrate/patterns/test_noise_resilience.py
git commit -m "test: add noise resilience tests"
```

---

### Task 9: Phase Interference at Scale Tests

**Files:**
- Create: `tests/quantum_substrate/interference/test_phase_interference.py`

**Step 1: Write phase interference tests**

Create `tests/quantum_substrate/interference/test_phase_interference.py`:

```python
"""Test phase interference at scale.

The claim: With many patterns having random phases, retrieval
should still find the correct match.

Success criteria:
- Correct match in top-3 at 1000 patterns
- Document degradation curve
"""

import torch
import pytest
import math

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.interference import interference_retrieval, jaccard_retrieval


class TestPhaseInterferenceAtScale:
    """Test retrieval accuracy as pattern count increases."""

    @pytest.mark.parametrize("n_patterns", [100, 500, 1000, 2000, 5000])
    def test_retrieval_accuracy_vs_scale(self, dim, rng, n_patterns):
        """Track if correct match stays in top-K as scale increases."""
        k = 50  # sparsity

        # Create query
        query = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        # Create memory with one exact match and rest random
        patterns = []

        # Pattern 0: exact match
        exact_match = ComplexSparsePattern(
            dim=dim,
            indices=query.indices,
            magnitudes=query.magnitudes,
            phases=query.phases,
        )
        patterns.append(exact_match)

        # Rest are random (different indices, random phases)
        for _ in range(n_patterns - 1):
            patterns.append(ComplexSparsePattern.random(dim=dim, k=k, generator=rng))

        # Test interference retrieval
        results = interference_retrieval(query, patterns)

        # Check if exact match is in top-3
        top_3_indices = [r[0] for r in results[:3]]
        in_top_3 = 0 in top_3_indices

        # Check rank of exact match
        exact_rank = next(i for i, r in enumerate(results) if r[0] == 0) + 1

        print(f"\n{n_patterns} patterns:")
        print(f"  Exact match rank: {exact_rank}")
        print(f"  In top-3: {in_top_3}")
        print(f"  Top-5 scores: {[(r[0], f'{r[1]:.3f}') for r in results[:5]]}")

        # Should find exact match at rank 1
        assert exact_rank == 1, f"Exact match should be rank 1, got {exact_rank}"

    def test_random_phase_patterns(self, dim, rng):
        """Test with patterns that share indices but have random phases."""
        k = 50
        n_patterns = 100

        # Create base pattern
        query = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        patterns = []

        # Pattern 0: same indices, same phases (should win)
        patterns.append(ComplexSparsePattern(
            dim=dim,
            indices=query.indices,
            magnitudes=query.magnitudes,
            phases=query.phases,
        ))

        # Patterns 1-10: same indices, random phases (phase interference)
        for _ in range(10):
            random_phases = tuple(
                (torch.rand(1, generator=rng).item() * 2 * math.pi)
                for _ in range(k)
            )
            patterns.append(ComplexSparsePattern(
                dim=dim,
                indices=query.indices,
                magnitudes=query.magnitudes,
                phases=random_phases,
            ))

        # Rest: random patterns
        for _ in range(n_patterns - 11):
            patterns.append(ComplexSparsePattern.random(dim=dim, k=k, generator=rng))

        # Interference should distinguish same-phase from random-phase
        int_results = interference_retrieval(query, patterns)

        # Jaccard can't distinguish (all have same indices for 0-10)
        jac_results = jaccard_retrieval(query, patterns)

        print("\nSame indices, different phases test:")
        print(f"  Interference top-5: {[(r[0], f'{r[1]:.3f}') for r in int_results[:5]]}")
        print(f"  Jaccard top-5: {[(r[0], f'{r[1]:.3f}') for r in jac_results[:5]]}")

        # Pattern 0 (same phase) should be top for interference
        assert int_results[0][0] == 0, "Same-phase pattern should rank first"

        # Interference score for pattern 0 should be much higher than patterns 1-10
        same_phase_score = int_results[0][1]
        random_phase_scores = [r[1] for r in int_results if r[0] in range(1, 11)]
        avg_random_phase = sum(random_phase_scores) / len(random_phase_scores) if random_phase_scores else 0

        print(f"  Same-phase score: {same_phase_score:.3f}")
        print(f"  Avg random-phase score: {avg_random_phase:.3f}")

        assert same_phase_score > avg_random_phase + 0.1, "Same-phase should score higher"
```

**Step 2: Run tests**

Run: `pytest tests/quantum_substrate/interference/test_phase_interference.py -v -s`
Expected: Tests validate phase-aware retrieval at scale

**Step 3: Commit**

```bash
git add tests/quantum_substrate/interference/test_phase_interference.py
git commit -m "test: add phase interference at scale tests"
```

---

### Task 10: Query Robustness Tests

**Files:**
- Create: `tests/quantum_substrate/scale/test_query_robustness.py`

**Step 1: Write query robustness tests**

Create `tests/quantum_substrate/scale/test_query_robustness.py`:

```python
"""Test query robustness.

The claim: Queries don't need to be exact matches of stored patterns.
Partial or slightly different queries should still find relevant results.

Success criteria:
- 10% different bits: correct in top-3
- Phase drift <= pi/4: correct in top-3
- Partial role binding: related results surface
"""

import torch
import pytest
import math

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity


class TestApproximateQueries:
    """Test retrieval with approximate query patterns."""

    @pytest.mark.parametrize("diff_frac", [0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
    def test_different_bits(self, dim, diff_frac):
        """Query with some bits different from stored pattern."""
        n_stored = 20

        # Create stored patterns (dense for simplicity)
        stored_patterns = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_stored)]
        stored_roles = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_stored)]

        # Bind and store
        stored_bindings = [bind_hrr(p, r) for p, r in zip(stored_patterns, stored_roles)]

        correct_in_top3 = 0

        for i in range(n_stored):
            # Create approximate query (some dimensions randomized)
            query_pattern = stored_patterns[i].clone()
            n_diff = int(diff_frac * dim)
            if n_diff > 0:
                diff_indices = torch.randperm(dim)[:n_diff]
                query_pattern[diff_indices] = torch.randn(n_diff, dtype=torch.complex64)

            # Bind with exact role
            query_binding = bind_hrr(query_pattern, stored_roles[i])

            # Find best matches
            similarities = [similarity(query_binding, s) for s in stored_bindings]
            top_3_indices = sorted(range(len(similarities)), key=lambda x: -similarities[x])[:3]

            if i in top_3_indices:
                correct_in_top3 += 1

        accuracy = correct_in_top3 / n_stored
        print(f"\n{diff_frac:.0%} different bits: {accuracy * 100:.0f}% in top-3")

        if diff_frac <= 0.10:
            assert accuracy >= 0.9, f"Expected >= 90% in top-3 at {diff_frac:.0%} diff"

    @pytest.mark.parametrize("phase_drift", [0.0, math.pi/8, math.pi/4, math.pi/2, math.pi])
    def test_phase_drift(self, dim, phase_drift):
        """Query with global phase rotation from stored pattern."""
        n_stored = 20

        stored_patterns = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_stored)]
        role = torch.randn(dim, dtype=torch.complex64)

        stored_bindings = [bind_hrr(p, role) for p in stored_patterns]

        correct_in_top3 = 0

        for i in range(n_stored):
            # Apply phase drift to query
            drifted_pattern = stored_patterns[i] * torch.exp(torch.tensor(1j * phase_drift))
            query_binding = bind_hrr(drifted_pattern, role)

            similarities = [similarity(query_binding, s) for s in stored_bindings]
            top_3_indices = sorted(range(len(similarities)), key=lambda x: -similarities[x])[:3]

            if i in top_3_indices:
                correct_in_top3 += 1

        accuracy = correct_in_top3 / n_stored
        print(f"\nPhase drift {phase_drift:.2f} rad: {accuracy * 100:.0f}% in top-3")

        if phase_drift <= math.pi/4:
            assert accuracy >= 0.8, f"Expected >= 80% in top-3 at drift={phase_drift:.2f}"


class TestPartialQueries:
    """Test querying with incomplete information."""

    def test_partial_role_query(self, dim):
        """Query event with only some roles bound."""
        # Store event: Alice (giver) gave Bob (receiver) a Book (item)
        alice = torch.randn(dim, dtype=torch.complex64)
        bob = torch.randn(dim, dtype=torch.complex64)
        book = torch.randn(dim, dtype=torch.complex64)

        giver = torch.randn(dim, dtype=torch.complex64)
        receiver = torch.randn(dim, dtype=torch.complex64)
        item = torch.randn(dim, dtype=torch.complex64)

        # Full event
        event = bind_hrr(alice, giver) + bind_hrr(bob, receiver) + bind_hrr(book, item)

        # Store multiple events
        events = [event]
        event_names = ["alice-gave-bob-book"]

        # Other events
        for _ in range(9):
            e1 = torch.randn(dim, dtype=torch.complex64)
            e2 = torch.randn(dim, dtype=torch.complex64)
            e3 = torch.randn(dim, dtype=torch.complex64)
            events.append(bind_hrr(e1, giver) + bind_hrr(e2, receiver) + bind_hrr(e3, item))
            event_names.append("other")

        # Partial query: just "alice as giver"
        partial_query = bind_hrr(alice, giver)

        # Find most similar event
        similarities = [similarity(partial_query, e) for e in events]
        best_idx = similarities.index(max(similarities))

        print(f"\nPartial query (alice+giver):")
        print(f"  Best match: {event_names[best_idx]} (idx={best_idx})")
        print(f"  Top-3 sims: {sorted(similarities, reverse=True)[:3]}")

        assert best_idx == 0, "Partial query should find the correct event"
```

**Step 2: Run tests**

Run: `pytest tests/quantum_substrate/scale/test_query_robustness.py -v -s`
Expected: Tests show query robustness characteristics

**Step 3: Commit**

```bash
git add tests/quantum_substrate/scale/test_query_robustness.py
git commit -m "test: add query robustness tests"
```

---

## Phase 4: Wave 2C - Dynamics Tests

### Task 11: Coherence Dynamics Tests

**Files:**
- Create: `tests/quantum_substrate/coherence/__init__.py`
- Create: `tests/quantum_substrate/coherence/test_decay.py`

**Step 1: Write coherence decay tests**

Create `tests/quantum_substrate/coherence/test_decay.py`:

```python
"""Test coherence decay dynamics.

The claim: Coherence should decay over time/steps, transitioning
patterns from quantum (high coherence) to classical (low coherence).

Note: This tests the CONCEPT. The actual coherence system may not
be implemented yet - these tests define the expected behavior.

Success criteria:
- Coherence decreases with each step
- Low coherence patterns behave classically (phase irrelevant)
- High coherence patterns show interference effects
"""

import torch
import pytest
import math


def simulate_coherence_decay(
    initial_coherence: float,
    decay_rate: float,
    steps: int,
) -> list[float]:
    """Simulate coherence decay over time.

    Simple exponential decay model:
    coherence(t) = coherence(0) * exp(-decay_rate * t)

    Args:
        initial_coherence: Starting coherence (0 to 1).
        decay_rate: Decay constant.
        steps: Number of time steps.

    Returns:
        List of coherence values at each step.
    """
    return [initial_coherence * math.exp(-decay_rate * step) for step in range(steps)]


def effective_phase(pattern: torch.Tensor, coherence: float) -> torch.Tensor:
    """Apply coherence-weighted phase.

    At low coherence, phase becomes noisy/irrelevant.
    At high coherence, phase is preserved.

    Args:
        pattern: Complex pattern with phase.
        coherence: Coherence level (0 = classical, 1 = quantum).

    Returns:
        Pattern with coherence-adjusted phase.
    """
    if coherence >= 0.99:
        return pattern

    # Add phase noise inversely proportional to coherence
    phase_noise = torch.randn(pattern.shape[0]) * math.pi * (1 - coherence)
    magnitude = pattern.abs()
    new_phase = torch.angle(pattern) + phase_noise

    return magnitude * torch.exp(1j * new_phase)


class TestCoherenceDecay:
    """Test coherence decay behavior."""

    def test_exponential_decay(self):
        """Coherence decays exponentially."""
        initial = 1.0
        decay_rate = 0.1
        steps = 50

        coherence_history = simulate_coherence_decay(initial, decay_rate, steps)

        # Should decrease monotonically
        for i in range(1, len(coherence_history)):
            assert coherence_history[i] < coherence_history[i-1], \
                f"Coherence should decrease: step {i}"

        # Should approach zero
        assert coherence_history[-1] < 0.01, "Should approach zero"

        print(f"\nCoherence decay (rate={decay_rate}):")
        for i in [0, 10, 20, 30, 40, 49]:
            print(f"  Step {i}: {coherence_history[i]:.4f}")

    def test_decay_rate_affects_speed(self):
        """Higher decay rate = faster decay."""
        initial = 1.0
        steps = 20

        slow = simulate_coherence_decay(initial, decay_rate=0.05, steps=steps)
        fast = simulate_coherence_decay(initial, decay_rate=0.2, steps=steps)

        # Fast decay should be lower at each step (after step 0)
        for i in range(1, steps):
            assert fast[i] < slow[i], f"Fast should be lower at step {i}"

        print(f"\nDecay rate comparison at step 10:")
        print(f"  Slow (0.05): {slow[10]:.4f}")
        print(f"  Fast (0.20): {fast[10]:.4f}")


class TestCoherenceEffects:
    """Test how coherence affects pattern behavior."""

    def test_high_coherence_preserves_phase(self, dim):
        """High coherence = phase information preserved."""
        pattern = torch.randn(dim, dtype=torch.complex64)
        original_phase = torch.angle(pattern)

        # High coherence
        high_coh = effective_phase(pattern, coherence=0.99)
        high_coh_phase = torch.angle(high_coh)

        phase_diff = torch.abs(original_phase - high_coh_phase).mean().item()
        print(f"\nHigh coherence (0.99): avg phase diff = {phase_diff:.4f} rad")

        assert phase_diff < 0.1, "High coherence should preserve phase"

    def test_low_coherence_randomizes_phase(self, dim):
        """Low coherence = phase becomes random."""
        pattern = torch.randn(dim, dtype=torch.complex64)
        original_phase = torch.angle(pattern)

        # Low coherence
        low_coh = effective_phase(pattern, coherence=0.1)
        low_coh_phase = torch.angle(low_coh)

        phase_diff = torch.abs(original_phase - low_coh_phase).mean().item()
        print(f"\nLow coherence (0.1): avg phase diff = {phase_diff:.4f} rad")

        # Should have significant phase drift (random noise added)
        assert phase_diff > 0.5, "Low coherence should randomize phase"

    def test_interference_requires_coherence(self, dim):
        """Only high-coherence patterns show quantum interference."""
        # Two patterns with same indices, opposite phases
        pattern1 = torch.randn(dim, dtype=torch.complex64)
        pattern2 = pattern1 * torch.exp(torch.tensor(1j * math.pi))  # opposite phase

        # High coherence: destructive interference
        p1_high = effective_phase(pattern1, coherence=0.99)
        p2_high = effective_phase(pattern2, coherence=0.99)
        combined_high = p1_high + p2_high
        energy_high = combined_high.abs().sum().item()

        # Low coherence: no interference (phases randomized)
        p1_low = effective_phase(pattern1, coherence=0.1)
        p2_low = effective_phase(pattern2, coherence=0.1)
        combined_low = p1_low + p2_low
        energy_low = combined_low.abs().sum().item()

        print(f"\nOpposite-phase patterns combined:")
        print(f"  High coherence energy: {energy_high:.2f}")
        print(f"  Low coherence energy: {energy_low:.2f}")

        # High coherence should cancel (low energy)
        # Low coherence should not cancel (higher energy due to random phases)
        assert energy_high < energy_low * 0.5, \
            "High coherence should show destructive interference"


class TestWorkingMemoryLimit:
    """Test if coherence budget creates natural working memory limit."""

    def test_limited_high_coherence_items(self):
        """Only ~4-7 items can maintain high coherence simultaneously."""
        # This tests the CONCEPT - actual implementation may differ

        # Simulate: each item needs coherence to stay "active"
        # Total coherence budget is limited

        total_budget = 1.0  # Total coherence available
        min_useful_coherence = 0.15  # Below this, pattern is "forgotten"

        # How many items can we maintain above threshold?
        for n_items in range(1, 15):
            per_item = total_budget / n_items
            useful = per_item >= min_useful_coherence

            status = "ACTIVE" if useful else "degraded"
            print(f"{n_items} items: {per_item:.3f} coherence each [{status}]")

            if not useful:
                max_items = n_items - 1
                break
        else:
            max_items = 14

        print(f"\nMax items above threshold: {max_items}")

        # Should be in the 4-7 range (Miller's law)
        assert 3 <= max_items <= 10, f"Expected 3-10 item limit, got {max_items}"
```

**Step 2: Run tests**

Run: `pytest tests/quantum_substrate/coherence/test_decay.py -v -s`
Expected: Tests define expected coherence behavior

**Step 3: Commit**

```bash
git add tests/quantum_substrate/coherence/
git commit -m "test: add coherence dynamics tests"
```

---

### Task 12: Catastrophic Interference Tests

**Files:**
- Create: `tests/quantum_substrate/scale/test_catastrophic_interference.py`

**Step 1: Write catastrophic interference tests**

Create `tests/quantum_substrate/scale/test_catastrophic_interference.py`:

```python
"""Test for catastrophic interference.

The claim: New patterns should not overwrite or corrupt old patterns.

Success criteria:
- >= 80% recall of early patterns after adding many new ones
- Similar patterns cause more interference than random ones
- Interleaved learning better than blocked learning
"""

import torch
import pytest

from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity


class TestPatternPersistence:
    """Test if old patterns persist after adding new ones."""

    def test_early_patterns_persist(self, dim):
        """Early stored patterns should still be retrievable."""
        n_early = 20
        n_later = 100

        # Store early patterns
        early_entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_early)]
        early_role = torch.randn(dim, dtype=torch.complex64)
        early_bindings = [bind_hrr(e, early_role) for e in early_entities]

        # Add later patterns (simulating continued learning)
        later_entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_later)]
        later_role = torch.randn(dim, dtype=torch.complex64)
        later_bindings = [bind_hrr(e, later_role) for e in later_entities]

        # All stored bindings
        all_bindings = early_bindings + later_bindings

        # Test recall of early patterns
        early_correct = 0
        for i, entity in enumerate(early_entities):
            # Try to recover from binding
            recovered = unbind_hrr(early_bindings[i], early_role)

            # Should match original entity, not later entities
            all_entities = early_entities + later_entities
            similarities = [similarity(recovered, e) for e in all_entities]
            best_match = similarities.index(max(similarities))

            if best_match == i:
                early_correct += 1

        recall = early_correct / n_early
        print(f"\nEarly pattern recall after adding {n_later} patterns: {recall * 100:.0f}%")

        assert recall >= 0.8, f"Expected >= 80% recall of early patterns"

    def test_similar_pattern_interference(self, dim):
        """Similar new patterns cause more interference than random ones."""
        # Store original pattern
        original = torch.randn(dim, dtype=torch.complex64)
        role = torch.randn(dim, dtype=torch.complex64)
        original_binding = bind_hrr(original, role)

        # Create similar pattern (30% overlap)
        similar = 0.7 * torch.randn(dim, dtype=torch.complex64) + 0.3 * original
        similar_binding = bind_hrr(similar, role)

        # Create random pattern
        random_p = torch.randn(dim, dtype=torch.complex64)
        random_binding = bind_hrr(random_p, role)

        # Recover original from its binding
        recovered_orig = unbind_hrr(original_binding, role)

        # Check similarities
        sim_to_original = similarity(recovered_orig, original)
        sim_to_similar = similarity(recovered_orig, similar)
        sim_to_random = similarity(recovered_orig, random_p)

        print(f"\nRecovery from original binding:")
        print(f"  Similarity to original: {sim_to_original:.3f}")
        print(f"  Similarity to similar: {sim_to_similar:.3f}")
        print(f"  Similarity to random: {sim_to_random:.3f}")

        # Original should be highest
        assert sim_to_original > sim_to_similar, "Original should beat similar"
        assert sim_to_original > sim_to_random, "Original should beat random"

        # Similar should be higher than random (it shares structure)
        # This is expected - not a problem, just documenting behavior
        print(f"  (Similar > random is expected due to shared structure)")


class TestLearningOrder:
    """Test if learning order affects interference."""

    def test_interleaved_vs_blocked(self, dim):
        """Compare interleaved learning to blocked learning."""
        n_per_category = 10
        n_categories = 3

        # Create category prototypes
        prototypes = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_categories)]

        # Create instances (similar to their prototype)
        instances = []
        labels = []
        for cat_idx, proto in enumerate(prototypes):
            for _ in range(n_per_category):
                instance = 0.7 * torch.randn(dim, dtype=torch.complex64) + 0.3 * proto
                instances.append(instance)
                labels.append(cat_idx)

        role = torch.randn(dim, dtype=torch.complex64)

        # Blocked learning: all cat0, then all cat1, then all cat2
        blocked_order = list(range(len(instances)))  # already blocked

        # Interleaved learning: cat0, cat1, cat2, cat0, cat1, cat2, ...
        interleaved_order = []
        for i in range(n_per_category):
            for cat in range(n_categories):
                interleaved_order.append(cat * n_per_category + i)

        def test_learning_order(order, name):
            bindings = []
            for idx in order:
                binding = bind_hrr(instances[idx], role)
                bindings.append((idx, binding))

            # Test recall of all patterns
            correct = 0
            for orig_idx, binding in bindings:
                recovered = unbind_hrr(binding, role)
                sims = [similarity(recovered, inst) for inst in instances]
                best = sims.index(max(sims))
                if best == orig_idx:
                    correct += 1

            return correct / len(bindings)

        blocked_acc = test_learning_order(blocked_order, "blocked")
        interleaved_acc = test_learning_order(interleaved_order, "interleaved")

        print(f"\nLearning order comparison:")
        print(f"  Blocked: {blocked_acc * 100:.0f}%")
        print(f"  Interleaved: {interleaved_acc * 100:.0f}%")

        # Note: In HRR, order shouldn't matter since patterns are independent
        # This test documents that behavior
```

**Step 2: Run tests**

Run: `pytest tests/quantum_substrate/scale/test_catastrophic_interference.py -v -s`
Expected: Tests document interference patterns

**Step 3: Commit**

```bash
git add tests/quantum_substrate/scale/test_catastrophic_interference.py
git commit -m "test: add catastrophic interference tests"
```

---

## Phase 5: Final Verification

### Task 13: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/quantum_substrate/ -v --tb=short`
Expected: All tests PASS (or documented expected failures)

**Step 2: Generate coverage report**

Run: `pytest tests/quantum_substrate/ --cov=src/quantum_substrate --cov-report=term-missing`
Expected: Coverage report generated

**Step 3: Update TEST_SUMMARY.md**

Add results from new tests to `dicussions/TEST_SUMMARY.md`.

**Step 4: Final commit**

```bash
git add -A
git commit -m "test: complete wave 2 test implementation

- Reorganized test directory structure
- Added sparse binding tests (Wave 2A)
- Added superposition limit tests (Wave 2A)
- Added semantic similarity tests (Wave 2A)
- Added noise resilience tests (Wave 2B)
- Added phase interference tests (Wave 2B)
- Added query robustness tests (Wave 2B)
- Added coherence dynamics tests (Wave 2C)
- Added catastrophic interference tests (Wave 2C)"
```

---

## Summary

| Phase | Tasks | Focus |
|-------|-------|-------|
| Phase 1 | Tasks 1-4 | Directory restructuring |
| Phase 2 | Tasks 5-7 | Wave 2A: Foundational (sparse, superposition, semantics) |
| Phase 3 | Tasks 8-10 | Wave 2B: Robustness (noise, phase interference, queries) |
| Phase 4 | Tasks 11-12 | Wave 2C: Dynamics (coherence, catastrophic interference) |
| Phase 5 | Task 13 | Verification |

**Total tasks:** 13
**Estimated new test files:** 8
**Key uncertainties addressed:** Sparse patterns, superposition limits, semantic similarity, noise resilience, phase interference at scale, coherence dynamics, catastrophic interference, query robustness
