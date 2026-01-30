"""Test binding with semantically similar patterns.

The claim: When "dog" and "wolf" share bits (semantic similarity),
binding should still correctly disambiguate roles.

Success criteria:
- Controlled overlap patterns work up to 30% similarity
- Role disambiguation works even with similar entities
"""

import torch
import pytest

from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity


def create_similar_pattern(
    source: torch.Tensor,
    overlap_frac: float,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Create a pattern that shares overlap_frac of its energy with source.

    Args:
        source: Source pattern.
        overlap_frac: Fraction of overlap (0 to 1).
        rng: Random generator.

    Returns:
        New pattern with controlled similarity to source.
    """
    dim = source.shape[0]
    noise = torch.randn(dim, dtype=torch.complex64)

    # Weighted combination: overlap * source + (1-overlap) * noise
    combined = overlap_frac * source + (1 - overlap_frac) * noise

    # Normalize to similar energy as source
    combined = combined * (source.abs().mean() / (combined.abs().mean() + 1e-8))

    return combined


class TestControlledOverlap:
    """Test binding with controlled pattern overlap."""

    @pytest.mark.parametrize("overlap", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    def test_binding_accuracy_vs_overlap(self, dim, overlap):
        """Track how overlap affects binding accuracy."""
        n_entities = 10

        # Create base entities
        base_entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_entities)]

        # Create similar entities (each similar to corresponding base)
        similar_entities = [
            create_similar_pattern(e, overlap) for e in base_entities
        ]

        role = torch.randn(dim, dtype=torch.complex64)

        # Test: can we distinguish entity[i] from similar_entity[i]?
        correct = 0
        for i in range(n_entities):
            entity = base_entities[i]
            similar = similar_entities[i]

            bound = bind_hrr(entity, role)
            recovered = unbind_hrr(bound, role)

            # Should be more similar to original than to similar version
            sim_original = similarity(recovered, entity)
            sim_similar = similarity(recovered, similar)

            if sim_original > sim_similar:
                correct += 1

        accuracy = correct / n_entities
        print(f"\nOverlap {overlap:.0%}: {accuracy * 100:.0f}% distinguish original from similar")

        # Should maintain distinction up to 30% overlap
        if overlap <= 0.3:
            assert accuracy >= 0.7, f"Expected >= 70% at {overlap:.0%} overlap"

    def test_semantic_triplet_dog_wolf(self, dim):
        """Test classic scenario: dog/wolf are similar, mailman is different."""
        # Create patterns
        dog = torch.randn(dim, dtype=torch.complex64)
        wolf = create_similar_pattern(dog, overlap_frac=0.3)  # 30% similar
        mailman = torch.randn(dim, dtype=torch.complex64)  # unrelated

        agent = torch.randn(dim, dtype=torch.complex64)
        patient = torch.randn(dim, dtype=torch.complex64)

        # Event 1: dog bit mailman
        event1 = bind_hrr(dog, agent) + bind_hrr(mailman, patient)

        # Event 2: wolf bit mailman
        event2 = bind_hrr(wolf, agent) + bind_hrr(mailman, patient)

        # Query: who was agent in event1?
        agent_in_event1 = unbind_hrr(event1, agent)

        # Should be most similar to dog, less to wolf, least to mailman
        sim_dog = similarity(agent_in_event1, dog)
        sim_wolf = similarity(agent_in_event1, wolf)
        sim_mailman = similarity(agent_in_event1, mailman)

        print(f"\nEvent1 agent query (should be dog):")
        print(f"  dog: {sim_dog:.3f}")
        print(f"  wolf: {sim_wolf:.3f}")
        print(f"  mailman: {sim_mailman:.3f}")

        assert sim_dog > sim_wolf, "Dog should be more similar than wolf"
        assert sim_dog > sim_mailman, "Dog should be more similar than mailman"

        # Query: who was agent in event2?
        agent_in_event2 = unbind_hrr(event2, agent)

        sim_dog = similarity(agent_in_event2, dog)
        sim_wolf = similarity(agent_in_event2, wolf)
        sim_mailman = similarity(agent_in_event2, mailman)

        print(f"\nEvent2 agent query (should be wolf):")
        print(f"  dog: {sim_dog:.3f}")
        print(f"  wolf: {sim_wolf:.3f}")
        print(f"  mailman: {sim_mailman:.3f}")

        assert sim_wolf > sim_dog, "Wolf should be more similar than dog"
        assert sim_wolf > sim_mailman, "Wolf should be more similar than mailman"


class TestSimilarEntitiesDifferentRoles:
    """Test disambiguation when similar entities have different roles."""

    def test_dog_agent_wolf_patient(self, dim):
        """Dog=agent, wolf=patient should be distinguishable."""
        dog = torch.randn(dim, dtype=torch.complex64)
        wolf = create_similar_pattern(dog, overlap_frac=0.3)

        agent = torch.randn(dim, dtype=torch.complex64)
        patient = torch.randn(dim, dtype=torch.complex64)

        # Event: dog (agent) bit wolf (patient)
        event = bind_hrr(dog, agent) + bind_hrr(wolf, patient)

        # Query agent
        recovered_agent = unbind_hrr(event, agent)
        sim_dog_agent = similarity(recovered_agent, dog)
        sim_wolf_agent = similarity(recovered_agent, wolf)

        # Query patient
        recovered_patient = unbind_hrr(event, patient)
        sim_dog_patient = similarity(recovered_patient, dog)
        sim_wolf_patient = similarity(recovered_patient, wolf)

        print(f"\nDog=agent, Wolf=patient:")
        print(f"  Agent query: dog={sim_dog_agent:.3f}, wolf={sim_wolf_agent:.3f}")
        print(f"  Patient query: dog={sim_dog_patient:.3f}, wolf={sim_wolf_patient:.3f}")

        assert sim_dog_agent > sim_wolf_agent, "Agent should be dog"
        assert sim_wolf_patient > sim_dog_patient, "Patient should be wolf"

    def test_multiple_similar_entities(self, dim):
        """Test with family of similar entities: dog, wolf, fox, coyote."""
        # Create family of similar animals
        dog = torch.randn(dim, dtype=torch.complex64)
        wolf = create_similar_pattern(dog, 0.3)
        fox = create_similar_pattern(dog, 0.25)
        coyote = create_similar_pattern(dog, 0.35)

        animals = [dog, wolf, fox, coyote]
        animal_names = ["dog", "wolf", "fox", "coyote"]

        # Human (unrelated)
        mailman = torch.randn(dim, dtype=torch.complex64)

        agent = torch.randn(dim, dtype=torch.complex64)
        patient = torch.randn(dim, dtype=torch.complex64)

        # Event: coyote bit mailman
        event = bind_hrr(coyote, agent) + bind_hrr(mailman, patient)

        # Query agent - should identify coyote, not other canids
        recovered = unbind_hrr(event, agent)

        print(f"\nEvent: coyote bit mailman")
        print(f"Agent query results:")
        for animal, name in zip(animals, animal_names):
            sim = similarity(recovered, animal)
            print(f"  {name}: {sim:.3f}")
        sim_mailman = similarity(recovered, mailman)
        print(f"  mailman: {sim_mailman:.3f}")

        # Coyote should win
        coyote_sim = similarity(recovered, coyote)
        for animal, name in zip(animals[:-1], animal_names[:-1]):  # exclude coyote
            other_sim = similarity(recovered, animal)
            assert coyote_sim > other_sim, f"Coyote should beat {name}"
