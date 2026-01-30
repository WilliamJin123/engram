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
