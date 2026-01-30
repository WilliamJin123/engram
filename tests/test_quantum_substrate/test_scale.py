"""Test D: Scale characteristics.

Validate that operations scale reasonably with:
- Pattern dimensionality
- Memory size (number of stored patterns)
- Pattern sparsity

Success criteria:
- Binding < 10ms per operation at dim=16384
- Retrieval < 1s for 5000 patterns
- Sparse representation > 5x compression vs dense
- Binding accuracy >= 85% at dim >= 1024
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

        # Should be fast enough for real-time use (< 10ms per operation)
        assert avg_ms < 10, f"Binding too slow at dim={dim}: {avg_ms:.1f}ms"


class TestRetrievalScale:
    """Test retrieval scaling with memory size."""

    @pytest.mark.parametrize("memory_size", [100, 500, 1000, 5000])
    def test_retrieval_time_vs_memory(self, dim, memory_size, rng):
        """Retrieval should scale linearly with memory size."""
        k = 50
        query = ComplexSparsePattern.random(dim=dim, k=k, generator=rng)

        # Create memory
        memory = [
            ComplexSparsePattern.random(dim=dim, k=k, generator=rng)
            for _ in range(memory_size)
        ]

        # Time Jaccard retrieval
        start = time.perf_counter()
        _ = jaccard_retrieval(query, memory)
        jaccard_time = time.perf_counter() - start

        # Time interference retrieval
        start = time.perf_counter()
        _ = interference_retrieval(query, memory)
        interference_time = time.perf_counter() - start

        print(f"\nMemory size {memory_size:5d}:")
        print(f"  Jaccard: {jaccard_time * 1000:.1f} ms")
        print(f"  Interference: {interference_time * 1000:.1f} ms")

        # Should complete in reasonable time (< 1s for retrieval)
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

        # Sparse should be much smaller (> 5x compression)
        assert ratio > 5, f"Expected >5x compression, got {ratio:.1f}x"

    @pytest.mark.parametrize("n_patterns", [1000, 5000, 10000])
    def test_memory_for_pattern_count(self, dim, n_patterns, rng):
        """Estimate memory usage for realistic pattern counts."""
        k = 50

        # Create patterns (sparse)
        patterns = [
            ComplexSparsePattern.random(dim=dim, k=k, generator=rng)
            for _ in range(n_patterns)
        ]

        # Estimate total memory
        bytes_per_pattern = k * (8 + 8 + 8)  # indices, magnitudes, phases
        total_bytes = bytes_per_pattern * n_patterns
        total_mb = total_bytes / (1024 * 1024)

        print(f"\n{n_patterns:,} patterns (dim={dim}, k={k}):")
        print(f"  Estimated: {total_mb:.1f} MB")

        # Should fit comfortably in memory (< 100 MB)
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

        # Higher dimensions should maintain accuracy (>= 85% at dim >= 1024)
        if dim >= 1024:
            assert accuracy >= 0.85, f"Expected >= 85% at dim={dim}, got {accuracy * 100:.0f}%"
