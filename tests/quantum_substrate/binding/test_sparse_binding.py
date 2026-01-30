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
