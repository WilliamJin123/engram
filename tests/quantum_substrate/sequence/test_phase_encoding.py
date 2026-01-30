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


def get_sequence_phase(pattern: torch.Tensor, reference: torch.Tensor) -> float:
    """Extract the sequence phase by comparing to a reference pattern.

    The sequence phase is the rotation applied to reference to get pattern.
    We compute it by looking at phase differences at non-zero positions.

    Args:
        pattern: The rotated pattern.
        reference: The original (unrotated) pattern.

    Returns:
        Estimated sequence phase in [0, 2*pi).
    """
    # Only look at positions where both have significant magnitude
    mask = (pattern.abs() > 1e-6) & (reference.abs() > 1e-6)
    if mask.sum() == 0:
        return 0.0

    # Phase difference = angle(pattern) - angle(reference)
    phase_diff = torch.angle(pattern[mask]) - torch.angle(reference[mask])

    # Use circular mean to handle wraparound
    # Convert to unit vectors and average
    unit_vectors = torch.exp(1j * phase_diff)
    mean_vector = unit_vectors.mean()
    mean_phase = torch.angle(mean_vector).item()

    # Normalize to [0, 2*pi)
    return (mean_phase + 2 * math.pi) % (2 * math.pi)


def get_next_in_sequence(
    query: torch.Tensor,
    candidates: list[torch.Tensor],
    references: list[torch.Tensor] | None = None,
) -> int:
    """Find which candidate comes next in sequence (by phase proximity).

    Args:
        query: Current item in sequence.
        candidates: All items to search.
        references: Original unrotated patterns (same order as candidates).
                   If None, assumes candidates share structure with query.

    Returns:
        Index of candidate with smallest positive phase difference.
    """
    if references is None:
        # Fall back to comparing patterns directly via their phase relationship
        # This works when all patterns started from the same base pattern
        references = candidates

    # Get query's sequence phase
    query_idx = None
    for i, cand in enumerate(candidates):
        if torch.allclose(cand, query, atol=1e-6):
            query_idx = i
            break

    if query_idx is None:
        # Query not in candidates, can't determine sequence
        return -1

    query_seq_phase = get_sequence_phase(candidates[query_idx], references[query_idx])

    best_idx = -1
    best_diff = float("inf")

    for i, (cand, ref) in enumerate(zip(candidates, references)):
        if i == query_idx:
            continue

        cand_seq_phase = get_sequence_phase(cand, ref)

        # Want positive phase difference (comes after)
        diff = (cand_seq_phase - query_seq_phase) % (2 * math.pi)

        # Skip very small differences (likely same position)
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
            next_idx = get_next_in_sequence(encoded[i], encoded, references=items)
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
                next_idx = get_next_in_sequence(encoded[i], encoded, references=items)
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
        backward_items = items[::-1]
        backward = encode_sequence_phase(backward_items)

        # Query: what comes after items[0]?
        # In forward sequence: items[1]
        # In backward sequence: nothing (items[0] is last)

        next_forward = get_next_in_sequence(forward[0], forward, references=items)
        next_backward = get_next_in_sequence(backward[-1], backward, references=backward_items)

        # In forward, first item should point to second
        assert next_forward == 1, "Forward: item[0] should point to item[1]"

        # In backward, last item (original first) shouldn't point to anything useful
        # (or should wrap around, which is different from forward)
        # The key is they're distinguishable
        assert next_forward != next_backward or next_backward == -1
