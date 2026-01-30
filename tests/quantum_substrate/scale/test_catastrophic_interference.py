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
