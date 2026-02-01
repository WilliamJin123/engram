# tests/agentic/test_text_encoder.py
"""Tests for text-to-pattern encoding."""

import pytest
import math
from agentic.text_encoder import TextEncoder, EncodingConfig


class TestTextEncoderDeterminism:
    """Test that encoding is deterministic and reproducible."""

    def test_same_text_same_pattern(self):
        """Identical text produces identical patterns."""
        encoder = TextEncoder(dim=1024, k=50)

        p1 = encoder.encode("hello world")
        p2 = encoder.encode("hello world")

        assert p1.bits == p2.bits
        assert p1.phases == p2.phases

    def test_different_text_different_pattern(self):
        """Different text produces different patterns."""
        encoder = TextEncoder(dim=1024, k=50)

        p1 = encoder.encode("hello world")
        p2 = encoder.encode("goodbye world")

        overlap = len(p1.bits & p2.bits)
        assert overlap < len(p1.bits)  # Not identical
        assert overlap > 0  # Some overlap from shared word

    def test_sparsity_maintained(self):
        """Output pattern has exactly k active bits."""
        encoder = TextEncoder(dim=1024, k=50)

        p = encoder.encode("test sentence with multiple words")

        assert len(p.bits) == 50


class TestTextEncoderLexicalOverlap:
    """Test that lexical similarity produces pattern overlap."""

    def test_shared_words_increase_overlap(self):
        """More shared words = higher Jaccard similarity."""
        encoder = TextEncoder(dim=1024, k=100)

        p1 = encoder.encode("the quick brown fox")
        p2 = encoder.encode("the quick brown dog")  # 3 shared words
        p3 = encoder.encode("a slow red cat")       # 0 shared words

        overlap_p1_p2 = len(p1.bits & p2.bits) / len(p1.bits | p2.bits)
        overlap_p1_p3 = len(p1.bits & p3.bits) / len(p1.bits | p3.bits)

        assert overlap_p1_p2 > overlap_p1_p3

    def test_word_order_affects_phase(self):
        """Same words in different order have different phases."""
        encoder = TextEncoder(dim=1024, k=50)

        p1 = encoder.encode("dog chased cat")
        p2 = encoder.encode("cat chased dog")

        # Same bits (same words)
        assert p1.bits == p2.bits

        # Different phases (different order)
        shared_bits = list(p1.bits & p2.bits)
        phase_diffs = [abs(p1.phases[b] - p2.phases[b]) for b in shared_bits[:10]]
        assert any(d > 0.1 for d in phase_diffs), "Phases should differ for different word order"


class TestTextEncoderPhaseEncoding:
    """Test positional phase encoding for word order."""

    def test_phases_in_valid_range(self):
        """All phases are between 0 and 2*pi."""
        encoder = TextEncoder(dim=1024, k=50)

        p = encoder.encode("some test text")

        for bit, phase in p.phases.items():
            assert 0 <= phase < 2 * math.pi

    def test_sequential_words_have_sequential_phases(self):
        """Words earlier in text have earlier phases."""
        encoder = TextEncoder(dim=1024, k=50)

        p_first = encoder.encode("first")
        p_second = encoder.encode("second")
        p_both = encoder.encode("first second")

        # Phases for "first" bits should be earlier than "second" bits


class TestEncodingConfigOptions:
    """Test configurable encoding options."""

    def test_custom_dimension(self):
        """Can create encoder with custom dimension."""
        encoder = TextEncoder(dim=2048, k=100)
        p = encoder.encode("test")

        assert all(b < 2048 for b in p.bits)

    def test_custom_sparsity(self):
        """Can create encoder with custom sparsity."""
        encoder = TextEncoder(dim=1024, k=25)
        p = encoder.encode("test")

        assert len(p.bits) == 25

    def test_ngram_encoding(self):
        """Can encode with n-grams for sub-word features."""
        config = EncodingConfig(use_ngrams=True, ngram_sizes=[2, 3])
        encoder = TextEncoder(dim=1024, k=50, config=config)

        p1 = encoder.encode("testing")
        p2 = encoder.encode("tester")  # Shares "test" prefix

        overlap = len(p1.bits & p2.bits) / len(p1.bits | p2.bits)

        assert overlap > 0
