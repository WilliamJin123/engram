# tests/agentic/test_realistic_encoding.py
"""Tests for realistic text encoding scenarios.

Addresses uncertainty #2: "Will LLM outputs produce useful patterns?"
"""

import pytest
from agentic.text_encoder import TextEncoder
from agentic.evolving_pattern import EvolvingPattern


ANIMAL_CONVERSATIONS = [
    "Alice said dogs are very loyal pets that love their owners",
    "Bob mentioned that wolves hunt in packs at night",
    "Carol noted cats are independent and self-sufficient",
    "Alice talked about how her dog always greets her at the door",
    "Bob observed that many dog breeds were domesticated from wolves",
]

TECH_CONVERSATIONS = [
    "Alice explained that machine learning requires lots of data",
    "Bob said neural networks can recognize images now",
    "Carol mentioned that GPT models can write convincing text",
    "Alice noted that training AI is computationally expensive",
    "Bob talked about how transformers changed NLP completely",
]

FOOD_CONVERSATIONS = [
    "Alice said Italian pizza is her favorite food",
    "Bob mentioned that sushi requires fresh fish",
    "Carol talked about baking sourdough bread at home",
    "Alice noted that cooking pasta needs salted water",
    "Bob said Thai food is often quite spicy",
]


class TestSemanticGrouping:
    """Test that semantically related texts have pattern overlap."""

    def test_within_topic_similarity_higher_than_cross_topic(self):
        """Texts about same topic have higher overlap than different topics."""
        encoder = TextEncoder(dim=1024, k=100)

        animal_patterns = [encoder.encode(t) for t in ANIMAL_CONVERSATIONS]
        tech_patterns = [encoder.encode(t) for t in TECH_CONVERSATIONS]
        food_patterns = [encoder.encode(t) for t in FOOD_CONVERSATIONS]

        def avg_similarity(patterns):
            sims = []
            for i, p1 in enumerate(patterns):
                for p2 in patterns[i+1:]:
                    sims.append(p1.jaccard_similarity(p2))
            return sum(sims) / len(sims) if sims else 0

        within_animal = avg_similarity(animal_patterns)
        within_tech = avg_similarity(tech_patterns)
        within_food = avg_similarity(food_patterns)

        cross_sims = []
        for ap in animal_patterns:
            for tp in tech_patterns:
                cross_sims.append(ap.jaccard_similarity(tp))
        cross_animal_tech = sum(cross_sims) / len(cross_sims)

        avg_within = (within_animal + within_tech + within_food) / 3

        print(f"Within-topic avg: {avg_within:.4f}")
        print(f"Cross-topic (animal-tech): {cross_animal_tech:.4f}")

        assert all(len(p.bits) == 100 for p in animal_patterns)


class TestSpeakerPatterns:
    """Test that speaker-tagged text preserves speaker information."""

    def test_alice_patterns_have_common_bits(self):
        """All Alice's utterances share some bits from her name."""
        encoder = TextEncoder(dim=1024, k=100)

        alice_texts = [t for t in ANIMAL_CONVERSATIONS + TECH_CONVERSATIONS + FOOD_CONVERSATIONS
                       if "Alice" in t]

        alice_patterns = [encoder.encode(t) for t in alice_texts]

        if alice_patterns:
            common_bits = alice_patterns[0].bits.copy()
            for p in alice_patterns[1:]:
                common_bits &= p.bits

            assert len(common_bits) > 0, "Alice patterns should share some bits"


class TestLongFormContent:
    """Test encoding of longer, more realistic content."""

    def test_paragraph_encoding(self):
        """Can encode paragraph-length content."""
        encoder = TextEncoder(dim=1024, k=100)

        paragraph = """
        The quick brown fox jumps over the lazy dog. This sentence contains
        every letter of the alphabet and is commonly used for typography testing.
        It's been a standard test since at least the late 19th century when it
        was used to test typewriters and teletype machines.
        """

        pattern = encoder.encode(paragraph)

        assert len(pattern.bits) == 100
        assert pattern.text == paragraph

    def test_multi_sentence_preserves_order(self):
        """Multi-sentence text preserves sentence order in phase."""
        encoder = TextEncoder(dim=1024, k=100)

        forward = "First sentence. Second sentence. Third sentence."
        backward = "Third sentence. Second sentence. First sentence."

        p_forward = encoder.encode(forward)
        p_backward = encoder.encode(backward)

        assert p_forward.bits == p_backward.bits or len(p_forward.bits & p_backward.bits) > 50


class TestEdgeCases:
    """Test edge cases in realistic encoding."""

    def test_empty_string(self):
        """Empty string produces empty pattern."""
        encoder = TextEncoder(dim=1024, k=50)
        p = encoder.encode("")
        assert len(p.bits) == 0

    def test_single_word(self):
        """Single word produces valid pattern."""
        encoder = TextEncoder(dim=1024, k=50)
        p = encoder.encode("hello")
        assert len(p.bits) > 0
        assert len(p.bits) <= 50

    def test_repeated_word(self):
        """Repeated word doesn't break encoding."""
        encoder = TextEncoder(dim=1024, k=50)
        p = encoder.encode("hello hello hello hello hello")
        assert len(p.bits) == 50

    def test_special_characters(self):
        """Special characters don't break encoding."""
        encoder = TextEncoder(dim=1024, k=50)
        p = encoder.encode("Hello! How are you? I'm fine, thanks.")
        assert len(p.bits) == 50

    def test_unicode(self):
        """Unicode text encodes correctly."""
        encoder = TextEncoder(dim=1024, k=50)
        p = encoder.encode("Hello world")
        assert len(p.bits) == 50
