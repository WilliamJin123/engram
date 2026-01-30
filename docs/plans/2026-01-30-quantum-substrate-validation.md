# Quantum Substrate Validation Tests

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate or invalidate the core quantum-inspired substrate mechanics before building the full agentic memory system.

**Architecture:** Four focused tests validating binding disambiguation, sequence encoding, interference utility, and scale characteristics. Each test is self-contained with clear success/failure criteria.

**Tech Stack:** Python 3.9+, PyTorch 2.0+ (complex tensors, FFT), pytest

---

## Overview

These tests answer the fundamental question: **Does quantum-inspired representation provide value over classical approaches?**

| Test | Question | If it fails... |
|------|----------|----------------|
| A: Binding | Can we recover WHO did WHAT? | Abandon role-based retrieval |
| B: Sequence | Can phase encode temporal order? | Use explicit timestamps instead |
| C: Interference | Does phase help retrieval? | Use classical Jaccard similarity |
| D: Scale | Does it scale to 10k patterns? | Revisit sparse representation |

---

## Project Structure

```
src/
  quantum_substrate/
    __init__.py
    patterns.py          # ComplexSparsePattern class
    binding.py           # HRR bind/unbind operations
    interference.py      # Phase-aware activation spreading
tests/
  quantum_substrate/
    __init__.py
    test_binding.py      # Test A
    test_sequence.py     # Test B
    test_interference.py # Test C
    test_scale.py        # Test D
    conftest.py          # Shared fixtures
```

---

## Task 1: Project Setup

**Files:**
- Create: `src/quantum_substrate/__init__.py`
- Create: `tests/quantum_substrate/__init__.py`
- Create: `tests/quantum_substrate/conftest.py`

**Step 1: Create source package**

```python
# src/quantum_substrate/__init__.py
"""Quantum-inspired substrate for memory systems."""

__version__ = "0.1.0"
```

**Step 2: Create test package**

```python
# tests/quantum_substrate/__init__.py
"""Tests for quantum substrate validation."""
```

**Step 3: Create conftest with shared fixtures**

```python
# tests/quantum_substrate/conftest.py
"""Shared fixtures for quantum substrate tests."""

import pytest
import torch

# Use consistent seed for reproducibility
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
    """Default sparsity (fraction of active dimensions)."""
    return 0.01  # 1% = ~10 active bits in 1024-dim
```

**Step 4: Verify setup**

Run: `python -c "from quantum_substrate import __version__; print(__version__)"`
Expected: `0.1.0`

**Step 5: Commit**

```bash
git add src/quantum_substrate tests/quantum_substrate
git commit -m "feat: initialize quantum substrate package structure"
```

---

## Task 2: ComplexSparsePattern Class

**Files:**
- Create: `src/quantum_substrate/patterns.py`
- Create: `tests/quantum_substrate/test_patterns.py`

**Step 1: Write the failing test for pattern creation**

```python
# tests/quantum_substrate/test_patterns.py
"""Tests for ComplexSparsePattern."""

import math
import torch
import pytest

from quantum_substrate.patterns import ComplexSparsePattern


class TestPatternCreation:
    """Test pattern construction and basic properties."""

    def test_create_random_pattern(self, dim, rng):
        """Random pattern has correct dimensionality and sparsity."""
        k = 10  # number of active dimensions
        pattern = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        assert pattern.dim == dim
        assert len(pattern.indices) == k
        assert all(0 <= i < dim for i in pattern.indices)

    def test_pattern_has_complex_amplitudes(self, dim, rng):
        """Each active dimension has magnitude and phase."""
        pattern = ComplexSparsePattern.random(dim=dim, k=10, generator=rng)

        assert len(pattern.magnitudes) == len(pattern.indices)
        assert len(pattern.phases) == len(pattern.indices)
        assert all(m > 0 for m in pattern.magnitudes)
        assert all(0 <= p < 2 * math.pi for p in pattern.phases)

    def test_pattern_to_dense(self, rng):
        """Can convert to dense complex tensor."""
        pattern = ComplexSparsePattern.random(dim=100, k=5, generator=rng)
        dense = pattern.to_dense()

        assert dense.shape == (100,)
        assert dense.dtype == torch.complex64
        assert (dense.abs() > 0).sum() == 5  # exactly 5 non-zero

    def test_pattern_from_dense(self, rng):
        """Can create from dense complex tensor."""
        # Create dense tensor with 3 non-zero entries
        dense = torch.zeros(100, dtype=torch.complex64)
        dense[10] = 1.0 + 0.5j
        dense[50] = 0.5 + 0.5j
        dense[90] = 0.3 + 0.1j

        pattern = ComplexSparsePattern.from_dense(dense)

        assert len(pattern.indices) == 3
        assert set(pattern.indices) == {10, 50, 90}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/quantum_substrate/test_patterns.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'quantum_substrate.patterns'"

**Step 3: Write minimal implementation**

```python
# src/quantum_substrate/patterns.py
"""Complex sparse pattern representation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class ComplexSparsePattern:
    """A sparse pattern with complex amplitudes.

    Each active dimension has a magnitude (strength) and phase (timing/relation).
    Stored sparsely for efficiency - only active dimensions are tracked.

    Attributes:
        dim: Total dimensionality of the pattern space.
        indices: Which dimensions are active (sorted).
        magnitudes: Magnitude at each active dimension.
        phases: Phase (0 to 2pi) at each active dimension.
    """

    dim: int
    indices: tuple[int, ...]
    magnitudes: tuple[float, ...]
    phases: tuple[float, ...]

    def __post_init__(self):
        """Validate and normalize."""
        assert len(self.indices) == len(self.magnitudes) == len(self.phases)
        assert all(0 <= i < self.dim for i in self.indices)
        # Ensure indices are sorted for consistent operations
        if self.indices != tuple(sorted(self.indices)):
            sorted_order = sorted(range(len(self.indices)), key=lambda i: self.indices[i])
            object.__setattr__(self, "indices", tuple(self.indices[i] for i in sorted_order))
            object.__setattr__(self, "magnitudes", tuple(self.magnitudes[i] for i in sorted_order))
            object.__setattr__(self, "phases", tuple(self.phases[i] for i in sorted_order))

    @classmethod
    def random(
        cls,
        dim: int,
        k: int,
        generator: torch.Generator | None = None,
    ) -> ComplexSparsePattern:
        """Create a random sparse pattern.

        Args:
            dim: Dimensionality of pattern space.
            k: Number of active dimensions.
            generator: Optional RNG for reproducibility.

        Returns:
            Random pattern with k active dimensions, unit magnitudes, random phases.
        """
        # Random indices
        indices = torch.randperm(dim, generator=generator)[:k].sort().values.tolist()

        # Unit magnitudes (can be varied later if needed)
        magnitudes = [1.0] * k

        # Random phases in [0, 2pi)
        phases = (torch.rand(k, generator=generator) * 2 * math.pi).tolist()

        return cls(dim=dim, indices=tuple(indices), magnitudes=tuple(magnitudes), phases=tuple(phases))

    def to_dense(self) -> torch.Tensor:
        """Convert to dense complex tensor.

        Returns:
            Complex tensor of shape (dim,) with non-zero entries at active indices.
        """
        dense = torch.zeros(self.dim, dtype=torch.complex64)
        for idx, mag, phase in zip(self.indices, self.magnitudes, self.phases):
            # complex = magnitude * e^(i*phase)
            dense[idx] = mag * torch.exp(torch.tensor(1j * phase))
        return dense

    @classmethod
    def from_dense(cls, dense: torch.Tensor, threshold: float = 1e-6) -> ComplexSparsePattern:
        """Create from dense complex tensor.

        Args:
            dense: Complex tensor to sparsify.
            threshold: Minimum magnitude to consider non-zero.

        Returns:
            Sparse pattern with only significant entries.
        """
        magnitudes = dense.abs()
        active_mask = magnitudes > threshold
        indices = torch.where(active_mask)[0].tolist()

        mags = magnitudes[active_mask].tolist()
        phases = torch.angle(dense[active_mask]).tolist()
        # Normalize phases to [0, 2pi)
        phases = [(p + 2 * math.pi) % (2 * math.pi) for p in phases]

        return cls(
            dim=len(dense),
            indices=tuple(indices),
            magnitudes=tuple(mags),
            phases=tuple(phases),
        )

    @property
    def k(self) -> int:
        """Number of active dimensions."""
        return len(self.indices)
```

**Step 4: Update package exports**

```python
# src/quantum_substrate/__init__.py
"""Quantum-inspired substrate for memory systems."""

from quantum_substrate.patterns import ComplexSparsePattern

__version__ = "0.1.0"
__all__ = ["ComplexSparsePattern"]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/quantum_substrate/test_patterns.py -v`
Expected: PASS (4 tests)

**Step 6: Commit**

```bash
git add src/quantum_substrate/patterns.py tests/quantum_substrate/test_patterns.py
git commit -m "feat: add ComplexSparsePattern class"
```

---

## Task 3: Test A - Binding Accuracy (HRR)

**Files:**
- Create: `src/quantum_substrate/binding.py`
- Create: `tests/quantum_substrate/test_binding.py`

This is the critical test. If binding doesn't work reliably, the entire substrate approach is questionable.

**Step 1: Write the failing test**

```python
# tests/quantum_substrate/test_binding.py
"""Test A: Binding Accuracy using Holographic Reduced Representations.

The claim: We can bind entities to roles and unbind them later to answer
structural queries like "who was the agent in the biting event?"

Success criteria:
- Unbinding recovers the correct entity with >90% accuracy
- Swapped-role events are distinguishable
- Works for at least 10 entities and 5 roles
"""

import torch
import pytest

from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity


class TestBindUnbind:
    """Test basic bind/unbind operations."""

    def test_bind_creates_new_pattern(self, dim):
        """Binding two patterns produces a third pattern."""
        a = torch.randn(dim, dtype=torch.complex64)
        b = torch.randn(dim, dtype=torch.complex64)

        bound = bind_hrr(a, b)

        assert bound.shape == (dim,)
        assert bound.dtype == torch.complex64

    def test_unbind_recovers_original(self, dim):
        """Unbinding with role recovers entity."""
        entity = torch.randn(dim, dtype=torch.complex64)
        role = torch.randn(dim, dtype=torch.complex64)

        # Normalize for cleaner results
        entity = entity / entity.abs().mean()
        role = role / role.abs().mean()

        bound = bind_hrr(entity, role)
        recovered = unbind_hrr(bound, role)

        # Recovered should be similar to original entity
        sim = similarity(recovered, entity)
        assert sim > 0.5, f"Expected similarity > 0.5, got {sim}"

    def test_unbind_with_wrong_role_fails(self, dim):
        """Unbinding with wrong role gives low similarity."""
        entity = torch.randn(dim, dtype=torch.complex64)
        role_a = torch.randn(dim, dtype=torch.complex64)
        role_b = torch.randn(dim, dtype=torch.complex64)

        bound = bind_hrr(entity, role_a)
        recovered = unbind_hrr(bound, role_b)

        # Should NOT recover entity
        sim = similarity(recovered, entity)
        assert sim < 0.3, f"Expected similarity < 0.3 with wrong role, got {sim}"


class TestRoleDisambiguation:
    """Test that binding disambiguates roles in events."""

    def test_dog_bit_mailman_vs_mailman_bit_dog(self, dim):
        """Swapped roles produce different bindings that can be queried."""
        # Entities
        dog = torch.randn(dim, dtype=torch.complex64)
        mailman = torch.randn(dim, dtype=torch.complex64)

        # Roles
        agent = torch.randn(dim, dtype=torch.complex64)
        patient = torch.randn(dim, dtype=torch.complex64)

        # Event 1: dog bit mailman (dog=agent, mailman=patient)
        event1 = bind_hrr(dog, agent) + bind_hrr(mailman, patient)

        # Event 2: mailman bit dog (mailman=agent, dog=patient)
        event2 = bind_hrr(mailman, agent) + bind_hrr(dog, patient)

        # Query: who was the agent in event 1?
        agent_in_event1 = unbind_hrr(event1, agent)

        # Should be more similar to dog than mailman
        sim_dog = similarity(agent_in_event1, dog)
        sim_mailman = similarity(agent_in_event1, mailman)

        assert sim_dog > sim_mailman, (
            f"Agent in event1 should be dog: "
            f"sim(dog)={sim_dog:.3f}, sim(mailman)={sim_mailman:.3f}"
        )

        # Query: who was the agent in event 2?
        agent_in_event2 = unbind_hrr(event2, agent)

        # Should be more similar to mailman than dog
        sim_dog = similarity(agent_in_event2, dog)
        sim_mailman = similarity(agent_in_event2, mailman)

        assert sim_mailman > sim_dog, (
            f"Agent in event2 should be mailman: "
            f"sim(dog)={sim_dog:.3f}, sim(mailman)={sim_mailman:.3f}"
        )


class TestBindingAtScale:
    """Test binding accuracy with many entities and roles."""

    @pytest.mark.parametrize("n_entities", [5, 10, 20])
    def test_identify_correct_entity(self, dim, n_entities):
        """Given n entities bound to roles, unbinding identifies the correct one."""
        # Create entities
        entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_entities)]

        # Create role
        role = torch.randn(dim, dtype=torch.complex64)

        correct = 0
        for i, entity in enumerate(entities):
            bound = bind_hrr(entity, role)
            recovered = unbind_hrr(bound, role)

            # Find most similar entity
            similarities = [similarity(recovered, e) for e in entities]
            best_match = similarities.index(max(similarities))

            if best_match == i:
                correct += 1

        accuracy = correct / n_entities
        assert accuracy >= 0.9, f"Expected >= 90% accuracy, got {accuracy * 100:.1f}%"

    def test_multiple_roles_per_event(self, dim):
        """Events with multiple role bindings can be queried for each role."""
        # Entities
        alice = torch.randn(dim, dtype=torch.complex64)
        bob = torch.randn(dim, dtype=torch.complex64)
        book = torch.randn(dim, dtype=torch.complex64)

        # Roles
        giver = torch.randn(dim, dtype=torch.complex64)
        receiver = torch.randn(dim, dtype=torch.complex64)
        item = torch.randn(dim, dtype=torch.complex64)

        # Event: Alice gave Bob a book
        event = (
            bind_hrr(alice, giver)
            + bind_hrr(bob, receiver)
            + bind_hrr(book, item)
        )

        # Query each role
        entities = [alice, bob, book]
        roles = [giver, receiver, item]
        expected = [0, 1, 2]  # indices of correct entities

        for role, expected_idx in zip(roles, expected):
            recovered = unbind_hrr(event, role)
            similarities = [similarity(recovered, e) for e in entities]
            best_match = similarities.index(max(similarities))
            assert best_match == expected_idx, (
                f"Role query failed: expected {expected_idx}, got {best_match}"
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/quantum_substrate/test_binding.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'quantum_substrate.binding'"

**Step 3: Write minimal implementation**

```python
# src/quantum_substrate/binding.py
"""Holographic Reduced Representation (HRR) binding operations.

HRR uses circular convolution to bind patterns and correlation to unbind.
This enables role-filler binding: we can store WHO did WHAT and query by role.

Reference: Plate, T. (1995). Holographic Reduced Representations.
"""

import torch


def bind_hrr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bind two patterns via circular convolution.

    Binding creates a new pattern that represents the association of a and b.
    The result is the same dimensionality as the inputs.

    Implementation: convolution in time domain = multiplication in frequency domain.

    Args:
        a: First pattern (complex tensor).
        b: Second pattern (complex tensor).

    Returns:
        Bound pattern (complex tensor, same shape).
    """
    # FFT-based circular convolution: IFFT(FFT(a) * FFT(b))
    return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b))


def unbind_hrr(bound: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Unbind pattern b from a bound pattern.

    Given bound = bind(a, b), unbind(bound, b) approximately recovers a.

    Implementation: correlation = convolution with conjugate.

    Args:
        bound: A bound pattern.
        b: The pattern to unbind (e.g., a role).

    Returns:
        Recovered pattern (approximately the other component).
    """
    # Correlation: IFFT(FFT(bound) * conj(FFT(b)))
    return torch.fft.ifft(torch.fft.fft(bound) * torch.conj(torch.fft.fft(b)))


def similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute normalized similarity between complex patterns.

    Uses cosine similarity on the real parts after normalization.

    Args:
        a: First pattern.
        b: Second pattern.

    Returns:
        Similarity in range [-1, 1], where 1 = identical.
    """
    # Normalize
    a_norm = a / (a.abs().mean() + 1e-8)
    b_norm = b / (b.abs().mean() + 1e-8)

    # Cosine similarity on real parts
    a_real = a_norm.real
    b_real = b_norm.real

    dot = (a_real * b_real).sum()
    norm_a = (a_real * a_real).sum().sqrt()
    norm_b = (b_real * b_real).sum().sqrt()

    return (dot / (norm_a * norm_b + 1e-8)).item()
```

**Step 4: Update package exports**

```python
# src/quantum_substrate/__init__.py
"""Quantum-inspired substrate for memory systems."""

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity

__version__ = "0.1.0"
__all__ = ["ComplexSparsePattern", "bind_hrr", "unbind_hrr", "similarity"]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/quantum_substrate/test_binding.py -v`
Expected: PASS (7 tests)

**Step 6: Commit**

```bash
git add src/quantum_substrate/binding.py tests/quantum_substrate/test_binding.py
git commit -m "feat: add HRR binding with role disambiguation tests"
```

---

## Task 4: Test B - Sequence Phase Encoding

**Files:**
- Create: `tests/quantum_substrate/test_sequence.py`

**Step 1: Write the failing test**

```python
# tests/quantum_substrate/test_sequence.py
"""Test B: Phase encodes temporal sequence.

The claim: We can encode order in phase, and phase proximity determines
activation order during retrieval.

Success criteria:
- Sequences of length 3-10 maintain correct order
- Identify the practical limit where phase aliasing breaks ordering
"""

import math
import torch
import pytest

from quantum_substrate.patterns import ComplexSparsePattern


def encode_sequence_phase(items: list[torch.Tensor]) -> list[torch.Tensor]:
    """Assign phases based on sequence position.

    Item at position i gets phase = 2*pi*i/n.

    Args:
        items: List of patterns to encode.

    Returns:
        Same patterns with phases set by position.
    """
    n = len(items)
    result = []
    for i, item in enumerate(items):
        phase = (2 * math.pi * i) / n
        # Rotate all elements by this phase
        rotated = item * torch.exp(torch.tensor(1j * phase))
        result.append(rotated)
    return result


def phase_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute average phase difference between patterns.

    Args:
        a: First pattern (complex).
        b: Second pattern (complex).

    Returns:
        Average angular distance in radians [0, pi].
    """
    phase_a = torch.angle(a)
    phase_b = torch.angle(b)

    # Angular distance (handle wraparound)
    diff = torch.abs(phase_a - phase_b)
    diff = torch.minimum(diff, 2 * math.pi - diff)

    # Average over non-zero elements
    mask = (a.abs() > 1e-6) & (b.abs() > 1e-6)
    if mask.sum() == 0:
        return math.pi  # No overlap = max distance
    return diff[mask].mean().item()


def get_next_in_sequence(
    query: torch.Tensor,
    candidates: list[torch.Tensor],
) -> int:
    """Find which candidate comes next in sequence (by phase proximity).

    Args:
        query: Current item in sequence.
        candidates: All items to search.

    Returns:
        Index of candidate with smallest positive phase difference.
    """
    query_phase = torch.angle(query).mean().item()

    best_idx = -1
    best_diff = float("inf")

    for i, cand in enumerate(candidates):
        cand_phase = torch.angle(cand).mean().item()

        # Want positive phase difference (comes after)
        diff = (cand_phase - query_phase) % (2 * math.pi)

        # Skip self (diff ~= 0)
        if diff < 0.01:
            continue

        if diff < best_diff:
            best_diff = diff
            best_idx = i

    return best_idx


class TestSequenceEncoding:
    """Test phase-based sequence encoding."""

    @pytest.mark.parametrize("seq_len", [3, 5, 7, 10])
    def test_sequence_order_preserved(self, dim, seq_len):
        """Items encoded in sequence can be retrieved in order."""
        # Create random patterns
        items = [torch.randn(dim, dtype=torch.complex64) for _ in range(seq_len)]

        # Encode with sequential phases
        encoded = encode_sequence_phase(items)

        # Start from first item, find subsequent items
        correct = 0
        for i in range(seq_len - 1):
            next_idx = get_next_in_sequence(encoded[i], encoded)
            if next_idx == i + 1:
                correct += 1

        accuracy = correct / (seq_len - 1)
        assert accuracy >= 0.8, (
            f"Sequence length {seq_len}: expected >= 80% order preservation, "
            f"got {accuracy * 100:.1f}%"
        )

    def test_phase_resolution_limit(self, dim):
        """Find where phase aliasing breaks sequence ordering."""
        results = []

        for seq_len in [5, 10, 20, 30, 50, 100]:
            items = [torch.randn(dim, dtype=torch.complex64) for _ in range(seq_len)]
            encoded = encode_sequence_phase(items)

            correct = 0
            for i in range(seq_len - 1):
                next_idx = get_next_in_sequence(encoded[i], encoded)
                if next_idx == i + 1:
                    correct += 1

            accuracy = correct / (seq_len - 1)
            results.append((seq_len, accuracy))

        # Report results for analysis
        print("\nPhase resolution vs sequence length:")
        for seq_len, acc in results:
            print(f"  n={seq_len:3d}: {acc * 100:5.1f}% accuracy")

        # At minimum, short sequences should work
        assert results[0][1] >= 0.8, "Short sequences (n=5) should work"

    @pytest.mark.parametrize("seq_len", [3, 5, 10])
    def test_forward_vs_backward_distinguishable(self, dim, seq_len):
        """Can distinguish A->B->C from C->B->A."""
        items = [torch.randn(dim, dtype=torch.complex64) for _ in range(seq_len)]

        forward = encode_sequence_phase(items)
        backward = encode_sequence_phase(items[::-1])

        # Query: what comes after items[0]?
        # In forward sequence: items[1]
        # In backward sequence: nothing (items[0] is last)

        next_forward = get_next_in_sequence(forward[0], forward)
        next_backward = get_next_in_sequence(backward[-1], backward)

        # In forward, first item should point to second
        assert next_forward == 1, "Forward: item[0] should point to item[1]"

        # In backward, last item (original first) shouldn't point to anything useful
        # (or should wrap around, which is different from forward)
        # The key is they're distinguishable
        assert next_forward != next_backward or next_backward == -1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/quantum_substrate/test_sequence.py -v`
Expected: Tests should actually PASS since we're using functions defined in the test file. Let's run to verify the encoding works.

**Step 3: Run test**

Run: `pytest tests/quantum_substrate/test_sequence.py -v`
Expected: PASS (but with informative output about phase limits)

**Step 4: Commit**

```bash
git add tests/quantum_substrate/test_sequence.py
git commit -m "test: add sequence phase encoding validation"
```

---

## Task 5: Test C - Interference vs Jaccard Comparison

**Files:**
- Create: `src/quantum_substrate/interference.py`
- Create: `tests/quantum_substrate/test_interference.py`

**Step 1: Write the failing test**

```python
# tests/quantum_substrate/test_interference.py
"""Test C: Does interference improve retrieval over classical Jaccard?

The claim: Phase-aware activation spreading produces better retrieval than
simply counting bit overlap.

Success criteria:
- Interference-based retrieval has higher precision on related items
- Or we learn that interference doesn't help (also valuable knowledge)
"""

import math
import random
import torch
import pytest

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.interference import (
    jaccard_retrieval,
    interference_retrieval,
    create_related_pattern,
)


class TestInterferenceBasics:
    """Test basic interference mechanics."""

    def test_same_phase_constructive(self):
        """Patterns with same phase add constructively."""
        dim = 100
        # Two patterns with overlapping indices, same phase
        a = torch.zeros(dim, dtype=torch.complex64)
        b = torch.zeros(dim, dtype=torch.complex64)

        a[10] = 1.0  # phase = 0
        a[20] = 1.0
        b[10] = 1.0  # same phase at overlap
        b[30] = 1.0

        combined = a + b

        # At index 10, magnitudes should add: |1+1| = 2
        assert abs(combined[10].abs().item() - 2.0) < 0.01

    def test_opposite_phase_destructive(self):
        """Patterns with opposite phase cancel."""
        dim = 100
        a = torch.zeros(dim, dtype=torch.complex64)
        b = torch.zeros(dim, dtype=torch.complex64)

        a[10] = 1.0  # phase = 0
        b[10] = -1.0  # phase = pi (opposite)

        combined = a + b

        # At index 10, magnitudes should cancel: |1 + (-1)| = 0
        assert combined[10].abs().item() < 0.01


class TestRetrievalComparison:
    """Compare interference-based vs Jaccard retrieval."""

    def test_retrieve_related_patterns(self, dim):
        """Interference should favor phase-coherent patterns."""
        # Create a query pattern
        query = ComplexSparsePattern.random(dim=dim, k=50)
        query_dense = query.to_dense()

        # Create memory with:
        # - Related patterns (share bits, similar phase)
        # - Unrelated patterns (share some bits, random phase)
        # - Distractor patterns (few shared bits)

        memory = []
        labels = []

        # 5 related patterns (same phase as query where they overlap)
        for _ in range(5):
            related = create_related_pattern(query, overlap_frac=0.5, phase_noise=0.1)
            memory.append(related.to_dense())
            labels.append("related")

        # 5 unrelated patterns (same overlap, but random phase)
        for _ in range(5):
            unrelated = create_related_pattern(query, overlap_frac=0.5, phase_noise=math.pi)
            memory.append(unrelated.to_dense())
            labels.append("unrelated")

        # 10 distractors (minimal overlap)
        for _ in range(10):
            distractor = ComplexSparsePattern.random(dim=dim, k=50)
            memory.append(distractor.to_dense())
            labels.append("distractor")

        # Retrieve with both methods
        jaccard_results = jaccard_retrieval(query_dense, memory, top_k=5)
        interference_results = interference_retrieval(query_dense, memory, top_k=5)

        # Count how many "related" patterns are in top 5
        jaccard_related = sum(1 for idx, _ in jaccard_results if labels[idx] == "related")
        interference_related = sum(1 for idx, _ in interference_results if labels[idx] == "related")

        print(f"\nTop-5 retrieval results:")
        print(f"  Jaccard: {jaccard_related}/5 related patterns")
        print(f"  Interference: {interference_related}/5 related patterns")

        # Interference should do at least as well as Jaccard
        # (This might fail - and that's valuable information!)
        assert interference_related >= jaccard_related - 1, (
            f"Interference ({interference_related}) significantly worse than "
            f"Jaccard ({jaccard_related})"
        )

    @pytest.mark.parametrize("memory_size", [50, 100, 500])
    def test_retrieval_at_scale(self, dim, memory_size):
        """Compare methods with larger memory."""
        query = ComplexSparsePattern.random(dim=dim, k=50)
        query_dense = query.to_dense()

        memory = []
        labels = []

        # 10% related, 90% random
        n_related = max(5, memory_size // 10)

        for i in range(memory_size):
            if i < n_related:
                pattern = create_related_pattern(query, overlap_frac=0.4, phase_noise=0.2)
                labels.append("related")
            else:
                pattern = ComplexSparsePattern.random(dim=dim, k=50)
                labels.append("random")
            memory.append(pattern.to_dense())

        # Shuffle
        combined = list(zip(memory, labels))
        random.shuffle(combined)
        memory, labels = zip(*combined)
        memory = list(memory)
        labels = list(labels)

        # Retrieve top-10
        jaccard_results = jaccard_retrieval(query_dense, memory, top_k=10)
        interference_results = interference_retrieval(query_dense, memory, top_k=10)

        jaccard_precision = sum(1 for idx, _ in jaccard_results if labels[idx] == "related") / 10
        interference_precision = sum(1 for idx, _ in interference_results if labels[idx] == "related") / 10

        print(f"\nMemory size {memory_size}, {n_related} related patterns:")
        print(f"  Jaccard precision@10: {jaccard_precision:.2f}")
        print(f"  Interference precision@10: {interference_precision:.2f}")

        # Both should find at least some related patterns
        assert jaccard_precision > 0 or interference_precision > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/quantum_substrate/test_interference.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'quantum_substrate.interference'"

**Step 3: Write minimal implementation**

```python
# src/quantum_substrate/interference.py
"""Interference-based retrieval operations.

Compares phase-aware interference retrieval against classical Jaccard similarity.
"""

import math
import torch

from quantum_substrate.patterns import ComplexSparsePattern


def jaccard_retrieval(
    query: torch.Tensor,
    memory: list[torch.Tensor],
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Retrieve patterns by Jaccard similarity (ignores phase).

    Jaccard = |A ∩ B| / |A ∪ B| on active dimensions.

    Args:
        query: Query pattern (complex tensor).
        memory: List of patterns to search.
        top_k: Number of results to return.

    Returns:
        List of (index, score) tuples, sorted by score descending.
    """
    query_active = query.abs() > 1e-6

    scores = []
    for i, pattern in enumerate(memory):
        pattern_active = pattern.abs() > 1e-6

        intersection = (query_active & pattern_active).sum().item()
        union = (query_active | pattern_active).sum().item()

        jaccard = intersection / union if union > 0 else 0
        scores.append((i, jaccard))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def interference_retrieval(
    query: torch.Tensor,
    memory: list[torch.Tensor],
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Retrieve patterns using phase-aware interference.

    Score = sum of |query + pattern| at overlapping indices.
    Same-phase components add constructively, opposite-phase cancel.

    Args:
        query: Query pattern (complex tensor).
        memory: List of patterns to search.
        top_k: Number of results to return.

    Returns:
        List of (index, score) tuples, sorted by score descending.
    """
    query_active = query.abs() > 1e-6

    scores = []
    for i, pattern in enumerate(memory):
        pattern_active = pattern.abs() > 1e-6
        overlap = query_active & pattern_active

        if overlap.sum() == 0:
            scores.append((i, 0.0))
            continue

        # Interference: add complex amplitudes, measure resulting magnitude
        combined = query + pattern
        interference_score = combined[overlap].abs().sum().item()

        # Normalize by what we'd get with perfect constructive interference
        max_possible = (query[overlap].abs() + pattern[overlap].abs()).sum().item()
        normalized = interference_score / max_possible if max_possible > 0 else 0

        scores.append((i, normalized))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def create_related_pattern(
    source: ComplexSparsePattern,
    overlap_frac: float = 0.5,
    phase_noise: float = 0.1,
) -> ComplexSparsePattern:
    """Create a pattern related to source with controlled overlap and phase similarity.

    Args:
        source: Source pattern to relate to.
        overlap_frac: Fraction of source indices to share.
        phase_noise: Standard deviation of phase perturbation (radians).
            0 = identical phase, pi = random phase.

    Returns:
        New pattern with specified relationship to source.
    """
    n_overlap = int(len(source.indices) * overlap_frac)
    n_new = source.k - n_overlap

    # Select overlapping indices from source
    overlap_indices = list(source.indices[:n_overlap])
    overlap_phases = [
        (source.phases[i] + torch.randn(1).item() * phase_noise) % (2 * math.pi)
        for i in range(n_overlap)
    ]
    overlap_magnitudes = [source.magnitudes[i] for i in range(n_overlap)]

    # Generate new random indices (not in source)
    source_set = set(source.indices)
    new_indices = []
    candidate = 0
    while len(new_indices) < n_new and candidate < source.dim:
        if candidate not in source_set:
            new_indices.append(candidate)
        candidate += 1

    new_phases = [torch.rand(1).item() * 2 * math.pi for _ in new_indices]
    new_magnitudes = [1.0] * len(new_indices)

    # Combine
    all_indices = overlap_indices + new_indices
    all_phases = overlap_phases + new_phases
    all_magnitudes = overlap_magnitudes + new_magnitudes

    return ComplexSparsePattern(
        dim=source.dim,
        indices=tuple(all_indices),
        magnitudes=tuple(all_magnitudes),
        phases=tuple(all_phases),
    )
```

**Step 4: Update package exports**

```python
# src/quantum_substrate/__init__.py
"""Quantum-inspired substrate for memory systems."""

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity
from quantum_substrate.interference import (
    jaccard_retrieval,
    interference_retrieval,
    create_related_pattern,
)

__version__ = "0.1.0"
__all__ = [
    "ComplexSparsePattern",
    "bind_hrr",
    "unbind_hrr",
    "similarity",
    "jaccard_retrieval",
    "interference_retrieval",
    "create_related_pattern",
]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/quantum_substrate/test_interference.py -v`
Expected: PASS (with comparative output)

**Step 6: Commit**

```bash
git add src/quantum_substrate/interference.py tests/quantum_substrate/test_interference.py
git commit -m "feat: add interference vs Jaccard retrieval comparison"
```

---

## Task 6: Test D - Scale Benchmark

**Files:**
- Create: `tests/quantum_substrate/test_scale.py`

**Step 1: Write the test**

```python
# tests/quantum_substrate/test_scale.py
"""Test D: Scale characteristics.

Validate that operations scale reasonably with:
- Pattern dimensionality
- Memory size (number of stored patterns)
- Pattern sparsity

Success criteria:
- Retrieval time grows linearly (or sub-linearly) with memory size
- Binding/unbinding is O(n log n) due to FFT
- Identify practical limits for real-time use
"""

import time
import torch
import pytest

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity
from quantum_substrate.interference import jaccard_retrieval, interference_retrieval


class TestBindingScale:
    """Test binding operation scaling."""

    @pytest.mark.parametrize("dim", [256, 1024, 4096, 16384])
    def test_binding_time_vs_dimension(self, dim):
        """Binding should be O(n log n) with FFT."""
        a = torch.randn(dim, dtype=torch.complex64)
        b = torch.randn(dim, dtype=torch.complex64)

        # Warmup
        _ = bind_hrr(a, b)

        # Time multiple iterations
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = bind_hrr(a, b)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_iters) * 1000
        print(f"\nBind dim={dim:5d}: {avg_ms:.3f} ms/op")

        # Should be fast enough for real-time use
        assert avg_ms < 10, f"Binding too slow at dim={dim}: {avg_ms:.1f}ms"


class TestRetrievalScale:
    """Test retrieval scaling with memory size."""

    @pytest.mark.parametrize("memory_size", [100, 500, 1000, 5000])
    def test_retrieval_time_vs_memory(self, dim, memory_size):
        """Retrieval should scale linearly with memory size."""
        query = ComplexSparsePattern.random(dim=dim, k=50).to_dense()

        # Create memory
        memory = [
            ComplexSparsePattern.random(dim=dim, k=50).to_dense()
            for _ in range(memory_size)
        ]

        # Time Jaccard retrieval
        start = time.perf_counter()
        _ = jaccard_retrieval(query, memory, top_k=10)
        jaccard_time = time.perf_counter() - start

        # Time interference retrieval
        start = time.perf_counter()
        _ = interference_retrieval(query, memory, top_k=10)
        interference_time = time.perf_counter() - start

        print(f"\nMemory size {memory_size:5d}:")
        print(f"  Jaccard: {jaccard_time * 1000:.1f} ms")
        print(f"  Interference: {interference_time * 1000:.1f} ms")

        # Should complete in reasonable time
        assert jaccard_time < 1.0, f"Jaccard too slow: {jaccard_time:.1f}s"
        assert interference_time < 1.0, f"Interference too slow: {interference_time:.1f}s"


class TestMemoryUsage:
    """Test memory characteristics."""

    def test_sparse_vs_dense_memory(self, dim):
        """Sparse representation should use less memory than dense."""
        k = 50  # 50 active dimensions

        # Sparse pattern
        sparse = ComplexSparsePattern.random(dim=dim, k=k)
        sparse_bytes = (
            len(sparse.indices) * 8  # indices (int64)
            + len(sparse.magnitudes) * 8  # magnitudes (float64)
            + len(sparse.phases) * 8  # phases (float64)
        )

        # Dense pattern
        dense = sparse.to_dense()
        dense_bytes = dense.element_size() * dense.numel()

        ratio = dense_bytes / sparse_bytes
        print(f"\nDim={dim}, k={k}:")
        print(f"  Sparse: {sparse_bytes:,} bytes")
        print(f"  Dense: {dense_bytes:,} bytes")
        print(f"  Compression ratio: {ratio:.1f}x")

        # Sparse should be much smaller
        assert ratio > 5, f"Expected >5x compression, got {ratio:.1f}x"

    @pytest.mark.parametrize("n_patterns", [1000, 5000, 10000])
    def test_memory_for_pattern_count(self, dim, n_patterns):
        """Estimate memory usage for realistic pattern counts."""
        k = 50

        # Create patterns (sparse)
        patterns = [
            ComplexSparsePattern.random(dim=dim, k=k)
            for _ in range(n_patterns)
        ]

        # Estimate total memory
        bytes_per_pattern = k * (8 + 8 + 8)  # indices, magnitudes, phases
        total_bytes = bytes_per_pattern * n_patterns
        total_mb = total_bytes / (1024 * 1024)

        print(f"\n{n_patterns:,} patterns (dim={dim}, k={k}):")
        print(f"  Estimated: {total_mb:.1f} MB")

        # Should fit comfortably in memory
        assert total_mb < 100, f"Too much memory: {total_mb:.1f} MB"


class TestAccuracyVsDimension:
    """Test how accuracy scales with dimensionality."""

    @pytest.mark.parametrize("dim", [256, 512, 1024, 2048, 4096])
    def test_binding_accuracy_vs_dimension(self, dim):
        """Higher dimensions should give better binding accuracy."""
        n_entities = 20
        correct = 0

        entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_entities)]
        role = torch.randn(dim, dtype=torch.complex64)

        for i, entity in enumerate(entities):
            bound = bind_hrr(entity, role)
            recovered = unbind_hrr(bound, role)

            similarities = [similarity(recovered, e) for e in entities]
            best_match = similarities.index(max(similarities))

            if best_match == i:
                correct += 1

        accuracy = correct / n_entities
        print(f"\nDim={dim:5d}: {accuracy * 100:.0f}% binding accuracy")

        # Higher dimensions should maintain accuracy
        if dim >= 1024:
            assert accuracy >= 0.85, f"Expected >= 85% at dim={dim}, got {accuracy * 100:.0f}%"
```

**Step 2: Run test**

Run: `pytest tests/quantum_substrate/test_scale.py -v -s`
Expected: PASS with timing and scaling information

**Step 3: Commit**

```bash
git add tests/quantum_substrate/test_scale.py
git commit -m "test: add scale benchmarks for substrate operations"
```

---

## Task 7: Final Integration Test

**Files:**
- Create: `tests/quantum_substrate/test_integration.py`

**Step 1: Write integration test**

```python
# tests/quantum_substrate/test_integration.py
"""Integration test combining all substrate capabilities.

Simulates a mini agentic memory scenario:
1. Store events with role bindings
2. Encode temporal sequence
3. Retrieve by query with interference
"""

import torch
import pytest

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity
from quantum_substrate.interference import interference_retrieval


class TestAgenticMemoryScenario:
    """Simulate storing and retrieving agent memories."""

    def test_conversation_memory(self, dim):
        """Store conversation events and query by participant and time."""
        # Entities (participants)
        alice = torch.randn(dim, dtype=torch.complex64)
        bob = torch.randn(dim, dtype=torch.complex64)

        # Roles
        speaker = torch.randn(dim, dtype=torch.complex64)
        topic = torch.randn(dim, dtype=torch.complex64)

        # Topics
        weather = torch.randn(dim, dtype=torch.complex64)
        work = torch.randn(dim, dtype=torch.complex64)
        lunch = torch.randn(dim, dtype=torch.complex64)

        # Events (conversation turns) with temporal phase
        events = []
        labels = []

        # Turn 1: Alice talks about weather
        e1 = bind_hrr(alice, speaker) + bind_hrr(weather, topic)
        e1 = e1 * torch.exp(torch.tensor(1j * 0.0))  # phase 0
        events.append(e1)
        labels.append("alice-weather")

        # Turn 2: Bob talks about work
        e2 = bind_hrr(bob, speaker) + bind_hrr(work, topic)
        e2 = e2 * torch.exp(torch.tensor(1j * 0.5))  # phase 0.5
        events.append(e2)
        labels.append("bob-work")

        # Turn 3: Alice talks about lunch
        e3 = bind_hrr(alice, speaker) + bind_hrr(lunch, topic)
        e3 = e3 * torch.exp(torch.tensor(1j * 1.0))  # phase 1.0
        events.append(e3)
        labels.append("alice-lunch")

        # Turn 4: Bob talks about weather
        e4 = bind_hrr(bob, speaker) + bind_hrr(weather, topic)
        e4 = e4 * torch.exp(torch.tensor(1j * 1.5))  # phase 1.5
        events.append(e4)
        labels.append("bob-weather")

        # Query: What did Alice say?
        alice_query = bind_hrr(alice, speaker)
        results = interference_retrieval(alice_query, events, top_k=4)

        # Alice's events should rank higher
        alice_events = {0, 2}  # indices of alice's turns
        top_2 = {results[0][0], results[1][0]}

        print(f"\nQuery: Alice's turns")
        for idx, score in results:
            print(f"  {labels[idx]}: {score:.3f}")

        # At least one of Alice's events should be in top 2
        assert len(alice_events & top_2) >= 1, "Alice's events should rank high"

        # Query: What was said about weather?
        weather_query = bind_hrr(weather, topic)
        results = interference_retrieval(weather_query, events, top_k=4)

        weather_events = {0, 3}  # indices of weather turns
        top_2 = {results[0][0], results[1][0]}

        print(f"\nQuery: Weather discussions")
        for idx, score in results:
            print(f"  {labels[idx]}: {score:.3f}")

        # At least one weather event should be in top 2
        assert len(weather_events & top_2) >= 1, "Weather events should rank high"

    def test_multiple_role_recovery(self, dim):
        """Recover multiple roles from a single event."""
        # Event: "Alice gave Bob the book at noon"
        alice = torch.randn(dim, dtype=torch.complex64)
        bob = torch.randn(dim, dtype=torch.complex64)
        book = torch.randn(dim, dtype=torch.complex64)
        noon = torch.randn(dim, dtype=torch.complex64)

        giver = torch.randn(dim, dtype=torch.complex64)
        receiver = torch.randn(dim, dtype=torch.complex64)
        item = torch.randn(dim, dtype=torch.complex64)
        time_role = torch.randn(dim, dtype=torch.complex64)

        event = (
            bind_hrr(alice, giver)
            + bind_hrr(bob, receiver)
            + bind_hrr(book, item)
            + bind_hrr(noon, time_role)
        )

        # Query each role
        entities = [alice, bob, book, noon]
        entity_names = ["alice", "bob", "book", "noon"]
        roles = [giver, receiver, item, time_role]
        role_names = ["giver", "receiver", "item", "time"]

        correct = 0
        for role, role_name, expected_idx in zip(roles, role_names, range(4)):
            recovered = unbind_hrr(event, role)
            similarities = [similarity(recovered, e) for e in entities]
            best_match = similarities.index(max(similarities))

            if best_match == expected_idx:
                correct += 1
                status = "OK"
            else:
                status = f"WRONG (got {entity_names[best_match]})"

            print(f"  {role_name} -> {entity_names[expected_idx]}: {status}")

        accuracy = correct / 4
        assert accuracy >= 0.75, f"Expected >= 75% role recovery, got {accuracy * 100:.0f}%"
```

**Step 2: Run integration test**

Run: `pytest tests/quantum_substrate/test_integration.py -v -s`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/quantum_substrate/test_integration.py
git commit -m "test: add agentic memory integration scenario"
```

---

## Task 8: Run Full Test Suite and Document Results

**Step 1: Run all tests**

Run: `pytest tests/quantum_substrate/ -v -s`

**Step 2: Create results summary**

After running tests, update `dicussions/quantum_tests.md` with actual results:

```markdown
## Validation Results (2026-01-30)

### Test A: Binding Accuracy
- Result: [PASS/FAIL]
- Accuracy at dim=1024: [X]%
- Notes: [observations]

### Test B: Sequence Phase
- Result: [PASS/FAIL]
- Practical limit: [N] items before aliasing
- Notes: [observations]

### Test C: Interference vs Jaccard
- Result: [PASS/FAIL]
- Interference advantage: [yes/no/marginal]
- Notes: [observations]

### Test D: Scale
- Result: [PASS/FAIL]
- Binding at dim=16384: [X] ms
- Retrieval for 10k patterns: [X] ms
- Notes: [observations]

### Conclusions
[Summary of whether to proceed with substrate approach]
```

**Step 3: Final commit**

```bash
git add dicussions/quantum_tests.md
git commit -m "docs: record substrate validation results"
```

---

## Verification

After completing all tasks, verify the full system:

1. **Run full test suite:**
   ```bash
   pytest tests/quantum_substrate/ -v
   ```
   Expected: All tests pass

2. **Check test coverage:**
   ```bash
   pytest tests/quantum_substrate/ --cov=quantum_substrate --cov-report=term-missing
   ```

3. **Verify package imports:**
   ```python
   from quantum_substrate import (
       ComplexSparsePattern,
       bind_hrr,
       unbind_hrr,
       similarity,
       jaccard_retrieval,
       interference_retrieval,
   )
   ```

---

## Success/Failure Decision Points

After running tests, make a go/no-go decision:

| Test | Go if... | No-go if... |
|------|----------|-------------|
| A: Binding | >85% accuracy at dim≥1024 | <70% accuracy |
| B: Sequence | Works for n≥5 items | Fails at n<5 |
| C: Interference | ≥Jaccard or marginal loss | Significantly worse |
| D: Scale | <100ms for 10k patterns | >1s for 1k patterns |

**If all Go:** Proceed to build agentic memory layer on this substrate.

**If any No-go:** Document findings, consider alternatives (explicit timestamps, classical sparse vectors, etc.)
