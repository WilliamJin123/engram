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
