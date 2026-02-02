# tests/agentic/test_text_encoder.py
"""Tests for text-to-pattern encoding."""

import pytest
import math
from agentic.text_encoder import TextEncoder, EncodingConfig, EncodedPattern


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


class TestTextEncoderValidation:
    """Test input validation."""

    def test_k_minimum_enforced(self):
        """k < 10 should raise ValueError."""
        with pytest.raises(ValueError, match="k must be >= 10"):
            TextEncoder(dim=1024, k=5)

        with pytest.raises(ValueError, match="k must be >= 10"):
            TextEncoder(dim=1024, k=9)

    def test_k_at_minimum_works(self):
        """k = 10 should work."""
        encoder = TextEncoder(dim=1024, k=10)
        pattern = encoder.encode("test")
        assert len(pattern.bits) == 10


class TestTextEncoderCollisions:
    """Test hash collision resistance."""

    def test_no_collisions_medium_corpus(self):
        """100+ unique words should produce 100+ unique patterns."""
        encoder = TextEncoder(dim=1024, k=10)

        # Common English words - all unique
        words = [
            "apple", "banana", "cherry", "date", "elderberry",
            "fig", "grape", "honeydew", "imbe", "jackfruit",
            "kiwi", "lemon", "mango", "nectarine", "orange",
            "papaya", "quince", "raspberry", "strawberry", "tangerine",
            "ugli", "vanilla", "watermelon", "ximenia", "yuzu",
            "dog", "cat", "bird", "fish", "horse",
            "elephant", "tiger", "lion", "bear", "wolf",
            "fox", "rabbit", "deer", "mouse", "rat",
            "snake", "lizard", "frog", "turtle", "whale",
            "dolphin", "shark", "octopus", "crab", "lobster",
            "red", "blue", "green", "yellow", "purple",
            "color", "pink", "brown", "black", "white",
            "gray", "silver", "gold", "bronze", "copper",
            "run", "walk", "jump", "swim", "fly",
            "eat", "drink", "sleep", "wake", "think",
            "talk", "listen", "watch", "read", "write",
            "happy", "sad", "angry", "scared", "excited",
            "tired", "hungry", "thirsty", "cold", "hot",
            "big", "small", "tall", "short", "wide",
            "narrow", "thick", "thin", "heavy", "light",
            "house", "car", "tree", "flower", "mountain",
            "river", "ocean", "lake", "forest", "desert",
        ]

        # Encode all words
        patterns = {word: encoder.encode(word) for word in words}

        # Check for bit-set collisions (identical bit patterns)
        bit_sets = {}
        collisions = []
        for word, pattern in patterns.items():
            bit_tuple = tuple(sorted(pattern.bits))
            if bit_tuple in bit_sets:
                collisions.append((word, bit_sets[bit_tuple]))
            else:
                bit_sets[bit_tuple] = word

        assert len(collisions) == 0, f"Found collisions: {collisions}"

    def test_single_char_difference_distinct(self):
        """Words differing by one char should produce distinct patterns."""
        encoder = TextEncoder(dim=1024, k=50)

        pairs = [
            ("cat", "bat"),
            ("dog", "log"),
            ("run", "sun"),
            ("hot", "pot"),
        ]

        for word1, word2 in pairs:
            p1 = encoder.encode(word1)
            p2 = encoder.encode(word2)
            # They should share some bits (similar words) but not be identical
            assert p1.bits != p2.bits, f"{word1} and {word2} should differ"

    def test_deterministic_encoding(self):
        """Same text always produces same pattern."""
        encoder = TextEncoder(dim=1024, k=50)

        text = "the quick brown fox"
        p1 = encoder.encode(text)
        p2 = encoder.encode(text)

        assert p1.bits == p2.bits
        assert p1.phases == p2.phases
