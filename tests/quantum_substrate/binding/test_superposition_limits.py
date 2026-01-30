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
