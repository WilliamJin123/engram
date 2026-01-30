"""Test A: Binding Accuracy using Holographic Reduced Representations.

The claim: We can bind entities to roles and unbind them later to answer
structural queries like "who was the agent in the biting event?"

Success criteria:
- Unbinding recovers the correct entity with >90% accuracy
- Swapped-role events are distinguishable
- Works for at least 10 entities and 5 roles
"""

import torch
import pytest

from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity


class TestBindUnbind:
    """Test basic bind/unbind operations."""

    def test_bind_creates_new_pattern(self, dim):
        """Binding two patterns produces a third pattern."""
        a = torch.randn(dim, dtype=torch.complex64)
        b = torch.randn(dim, dtype=torch.complex64)

        bound = bind_hrr(a, b)

        assert bound.shape == (dim,)
        assert bound.dtype == torch.complex64

    def test_unbind_recovers_original(self, dim):
        """Unbinding with role recovers entity."""
        entity = torch.randn(dim, dtype=torch.complex64)
        role = torch.randn(dim, dtype=torch.complex64)

        # Normalize for cleaner results
        entity = entity / entity.abs().mean()
        role = role / role.abs().mean()

        bound = bind_hrr(entity, role)
        recovered = unbind_hrr(bound, role)

        # Recovered should be similar to original entity
        sim = similarity(recovered, entity)
        assert sim > 0.5, f"Expected similarity > 0.5, got {sim}"

    def test_unbind_with_wrong_role_fails(self, dim):
        """Unbinding with wrong role gives low similarity."""
        entity = torch.randn(dim, dtype=torch.complex64)
        role_a = torch.randn(dim, dtype=torch.complex64)
        role_b = torch.randn(dim, dtype=torch.complex64)

        bound = bind_hrr(entity, role_a)
        recovered = unbind_hrr(bound, role_b)

        # Should NOT recover entity
        sim = similarity(recovered, entity)
        assert sim < 0.3, f"Expected similarity < 0.3 with wrong role, got {sim}"


class TestRoleDisambiguation:
    """Test that binding disambiguates roles in events."""

    def test_dog_bit_mailman_vs_mailman_bit_dog(self, dim):
        """Swapped roles produce different bindings that can be queried."""
        # Entities
        dog = torch.randn(dim, dtype=torch.complex64)
        mailman = torch.randn(dim, dtype=torch.complex64)

        # Roles
        agent = torch.randn(dim, dtype=torch.complex64)
        patient = torch.randn(dim, dtype=torch.complex64)

        # Event 1: dog bit mailman (dog=agent, mailman=patient)
        event1 = bind_hrr(dog, agent) + bind_hrr(mailman, patient)

        # Event 2: mailman bit dog (mailman=agent, dog=patient)
        event2 = bind_hrr(mailman, agent) + bind_hrr(dog, patient)

        # Query: who was the agent in event 1?
        agent_in_event1 = unbind_hrr(event1, agent)

        # Should be more similar to dog than mailman
        sim_dog = similarity(agent_in_event1, dog)
        sim_mailman = similarity(agent_in_event1, mailman)

        assert sim_dog > sim_mailman, (
            f"Agent in event1 should be dog: "
            f"sim(dog)={sim_dog:.3f}, sim(mailman)={sim_mailman:.3f}"
        )

        # Query: who was the agent in event 2?
        agent_in_event2 = unbind_hrr(event2, agent)

        # Should be more similar to mailman than dog
        sim_dog = similarity(agent_in_event2, dog)
        sim_mailman = similarity(agent_in_event2, mailman)

        assert sim_mailman > sim_dog, (
            f"Agent in event2 should be mailman: "
            f"sim(dog)={sim_dog:.3f}, sim(mailman)={sim_mailman:.3f}"
        )


class TestBindingAtScale:
    """Test binding accuracy with many entities and roles."""

    @pytest.mark.parametrize("n_entities", [5, 10, 20])
    def test_identify_correct_entity(self, dim, n_entities):
        """Given n entities bound to roles, unbinding identifies the correct one."""
        # Create entities
        entities = [torch.randn(dim, dtype=torch.complex64) for _ in range(n_entities)]

        # Create role
        role = torch.randn(dim, dtype=torch.complex64)

        correct = 0
        for i, entity in enumerate(entities):
            bound = bind_hrr(entity, role)
            recovered = unbind_hrr(bound, role)

            # Find most similar entity
            similarities = [similarity(recovered, e) for e in entities]
            best_match = similarities.index(max(similarities))

            if best_match == i:
                correct += 1

        accuracy = correct / n_entities
        assert accuracy >= 0.9, f"Expected >= 90% accuracy, got {accuracy * 100:.1f}%"

    def test_multiple_roles_per_event(self, dim):
        """Events with multiple role bindings can be queried for each role."""
        # Entities
        alice = torch.randn(dim, dtype=torch.complex64)
        bob = torch.randn(dim, dtype=torch.complex64)
        book = torch.randn(dim, dtype=torch.complex64)

        # Roles
        giver = torch.randn(dim, dtype=torch.complex64)
        receiver = torch.randn(dim, dtype=torch.complex64)
        item = torch.randn(dim, dtype=torch.complex64)

        # Event: Alice gave Bob a book
        event = (
            bind_hrr(alice, giver)
            + bind_hrr(bob, receiver)
            + bind_hrr(book, item)
        )

        # Query each role
        entities = [alice, bob, book]
        roles = [giver, receiver, item]
        expected = [0, 1, 2]  # indices of correct entities

        for role, expected_idx in zip(roles, expected):
            recovered = unbind_hrr(event, role)
            similarities = [similarity(recovered, e) for e in entities]
            best_match = similarities.index(max(similarities))
            assert best_match == expected_idx, (
                f"Role query failed: expected {expected_idx}, got {best_match}"
            )
