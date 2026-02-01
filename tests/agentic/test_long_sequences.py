# tests/agentic/test_long_sequences.py
"""Tests for very long sequence phase encoding.

Addresses uncertainty #3: "Phase encoding at 1000+ items?"
"""

import pytest
import math
import torch
from agentic.evolving_pattern import EvolvingPattern


def create_sequence_patterns(n: int, dim: int = 1024, k: int = 50) -> list[EvolvingPattern]:
    """Create n patterns with sequential phases."""
    patterns = []
    for i in range(n):
        p = EvolvingPattern.from_text(f"item_{i}", dim=dim, k=k)

        seq_phase = (2 * math.pi * i) / n
        for bit in p.bits:
            p.phases[bit] = seq_phase

        patterns.append(p)

    return patterns


def measure_sequence_accuracy(patterns: list[EvolvingPattern]) -> float:
    """Measure how accurately we can recover sequence order from phase."""
    n = len(patterns)
    correct = 0

    for i in range(n - 1):
        current_phase = list(patterns[i].phases.values())[0]

        expected_delta = (2 * math.pi) / n
        expected_next = (current_phase + expected_delta) % (2 * math.pi)

        min_diff = float('inf')
        closest_idx = -1

        for j in range(n):
            if j == i:
                continue
            other_phase = list(patterns[j].phases.values())[0]

            diff = abs(expected_next - other_phase)
            diff = min(diff, 2 * math.pi - diff)

            if diff < min_diff:
                min_diff = diff
                closest_idx = j

        if closest_idx == i + 1:
            correct += 1

    return correct / (n - 1) if n > 1 else 1.0


class TestSequenceLengthLimits:
    """Test phase encoding at various sequence lengths."""

    @pytest.mark.parametrize("length", [10, 50, 100, 200, 500])
    def test_medium_sequences(self, length):
        """Test medium-length sequences."""
        patterns = create_sequence_patterns(length)
        accuracy = measure_sequence_accuracy(patterns)

        assert accuracy >= 0.95, f"Accuracy {accuracy:.2%} too low at length {length}"

    @pytest.mark.parametrize("length", [1000, 2000, 5000])
    def test_long_sequences(self, length):
        """Test long sequences (the uncertainty region)."""
        patterns = create_sequence_patterns(length)
        accuracy = measure_sequence_accuracy(patterns)

        print(f"Sequence length {length}: accuracy = {accuracy:.4f}")

    @pytest.mark.parametrize("length", [10000, 20000])
    def test_very_long_sequences(self, length):
        """Test very long sequences (expected to show degradation)."""
        patterns = create_sequence_patterns(length)
        accuracy = measure_sequence_accuracy(patterns)

        print(f"Sequence length {length}: accuracy = {accuracy:.4f}")


class TestPhaseResolutionTheory:
    """Theoretical analysis of phase resolution limits."""

    def test_phase_resolution_calculation(self):
        """Calculate theoretical phase resolution at various lengths."""
        for n in [100, 1000, 10000, 100000]:
            resolution_rad = 2 * math.pi / n
            resolution_deg = math.degrees(resolution_rad)

            print(f"n={n:6d}: resolution = {resolution_rad:.6f} rad = {resolution_deg:.4f} degrees")

    def test_noise_threshold_estimate(self):
        """Estimate noise threshold that would cause errors."""
        noise_rad = 0.01
        threshold_n = int(2 * math.pi / (2 * noise_rad))

        print(f"With noise = {noise_rad} rad, expect errors above n = {threshold_n}")


class TestPracticalSequenceScenarios:
    """Test realistic sequence scenarios."""

    def test_conversation_history(self):
        """Simulate encoding a long conversation history."""
        n_turns = 200

        patterns = []
        for i in range(n_turns):
            speaker = ["Alice", "Bob", "Carol"][i % 3]
            text = f"{speaker} said turn number {i} content here"

            p = EvolvingPattern.from_text(text, dim=1024, k=50)

            seq_phase = (2 * math.pi * i) / n_turns
            for bit in p.bits:
                original_phase = p.phases[bit]
                p.phases[bit] = (original_phase + seq_phase) / 2

            patterns.append(p)

        def avg_phase(p):
            return sum(p.phases.values()) / len(p.phases) if p.phases else 0

        phase_order = sorted(range(n_turns), key=lambda i: avg_phase(patterns[i]))

        correct_positions = sum(1 for i, j in enumerate(phase_order) if abs(i - j) <= 5)
        position_accuracy = correct_positions / n_turns

        print(f"Conversation ordering accuracy (within 5): {position_accuracy:.2%}")
