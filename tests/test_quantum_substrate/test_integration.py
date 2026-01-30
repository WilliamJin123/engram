"""Integration test combining all substrate capabilities.

Simulates a mini agentic memory scenario:
1. Store events with role bindings
2. Encode temporal sequence with phase
3. Retrieve by query with interference
"""

import torch
import pytest

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity


def interference_retrieval_dense(
    query: torch.Tensor,
    patterns: list[torch.Tensor],
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Retrieve dense patterns using phase-aware interference.

    Score = sum of |query + pattern| at overlapping indices.
    Same-phase components add constructively, opposite-phase cancel.

    Args:
        query: Query pattern (complex tensor).
        patterns: List of patterns to search.
        top_k: Number of results to return.

    Returns:
        List of (index, score) tuples, sorted by score descending.
    """
    query_active = query.abs() > 1e-6

    scores = []
    for i, pattern in enumerate(patterns):
        pattern_active = pattern.abs() > 1e-6
        overlap = query_active & pattern_active

        if overlap.sum() == 0:
            scores.append((i, 0.0))
            continue

        # Interference: add complex amplitudes, measure resulting magnitude
        combined = query + pattern
        interference_score = combined[overlap].abs().sum().item()

        # Normalize by what we'd get with perfect constructive interference
        max_possible = (query[overlap].abs() + pattern[overlap].abs()).sum().item()
        normalized = interference_score / max_possible if max_possible > 0 else 0

        scores.append((i, normalized))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


class TestAgenticMemoryScenario:
    """Simulate storing and retrieving agent memories."""

    def test_conversation_memory(self, dim):
        """Store conversation events and query by participant and topic.

        Scenario: Alice and Bob have a conversation about weather, work, and lunch.
        We encode each turn with speaker/topic role bindings and temporal phase,
        then query by participant and by topic.

        Success criteria: At least one correct result in top-2 for each query type.
        """
        # Entities (participants)
        alice = torch.randn(dim, dtype=torch.complex64)
        bob = torch.randn(dim, dtype=torch.complex64)

        # Roles
        speaker = torch.randn(dim, dtype=torch.complex64)
        topic = torch.randn(dim, dtype=torch.complex64)

        # Topics
        weather = torch.randn(dim, dtype=torch.complex64)
        work = torch.randn(dim, dtype=torch.complex64)
        lunch = torch.randn(dim, dtype=torch.complex64)

        # Events (conversation turns) with temporal phase
        events = []
        labels = []

        # Turn 1: Alice talks about weather (phase 0.0)
        e1 = bind_hrr(alice, speaker) + bind_hrr(weather, topic)
        e1 = e1 * torch.exp(torch.tensor(1j * 0.0))
        events.append(e1)
        labels.append("alice-weather")

        # Turn 2: Bob talks about work (phase 0.5)
        e2 = bind_hrr(bob, speaker) + bind_hrr(work, topic)
        e2 = e2 * torch.exp(torch.tensor(1j * 0.5))
        events.append(e2)
        labels.append("bob-work")

        # Turn 3: Alice talks about lunch (phase 1.0)
        e3 = bind_hrr(alice, speaker) + bind_hrr(lunch, topic)
        e3 = e3 * torch.exp(torch.tensor(1j * 1.0))
        events.append(e3)
        labels.append("alice-lunch")

        # Turn 4: Bob talks about weather (phase 1.5)
        e4 = bind_hrr(bob, speaker) + bind_hrr(weather, topic)
        e4 = e4 * torch.exp(torch.tensor(1j * 1.5))
        events.append(e4)
        labels.append("bob-weather")

        # Query: What did Alice say?
        alice_query = bind_hrr(alice, speaker)
        results = interference_retrieval_dense(alice_query, events, top_k=4)

        # Alice's events should rank higher
        alice_events = {0, 2}  # indices of alice's turns
        top_2 = {results[0][0], results[1][0]}

        print(f"\nQuery: Alice's turns")
        for idx, score in results:
            print(f"  {labels[idx]}: {score:.3f}")

        # At least one of Alice's events should be in top 2
        alice_in_top2 = len(alice_events & top_2)
        assert alice_in_top2 >= 1, (
            f"Alice's events should rank high, got top_2={top_2}, alice_events={alice_events}"
        )

        # Query: What was said about weather?
        weather_query = bind_hrr(weather, topic)
        results = interference_retrieval_dense(weather_query, events, top_k=4)

        weather_events = {0, 3}  # indices of weather turns
        top_2 = {results[0][0], results[1][0]}

        print(f"\nQuery: Weather discussions")
        for idx, score in results:
            print(f"  {labels[idx]}: {score:.3f}")

        # At least one weather event should be in top 2
        weather_in_top2 = len(weather_events & top_2)
        assert weather_in_top2 >= 1, (
            f"Weather events should rank high, got top_2={top_2}, weather_events={weather_events}"
        )

    def test_multiple_role_recovery(self, dim):
        """Recover multiple roles from a single event.

        Scenario: "Alice gave Bob the book at noon"
        We encode this event with 4 role bindings: giver, receiver, item, time.
        Then we query each role to recover the bound entity.

        Success criteria: >= 75% role recovery accuracy (3/4 correct).
        """
        # Entities
        alice = torch.randn(dim, dtype=torch.complex64)
        bob = torch.randn(dim, dtype=torch.complex64)
        book = torch.randn(dim, dtype=torch.complex64)
        noon = torch.randn(dim, dtype=torch.complex64)

        # Roles
        giver = torch.randn(dim, dtype=torch.complex64)
        receiver = torch.randn(dim, dtype=torch.complex64)
        item = torch.randn(dim, dtype=torch.complex64)
        time_role = torch.randn(dim, dtype=torch.complex64)

        # Event: "Alice gave Bob the book at noon"
        event = (
            bind_hrr(alice, giver)
            + bind_hrr(bob, receiver)
            + bind_hrr(book, item)
            + bind_hrr(noon, time_role)
        )

        # Query each role
        entities = [alice, bob, book, noon]
        entity_names = ["alice", "bob", "book", "noon"]
        roles = [giver, receiver, item, time_role]
        role_names = ["giver", "receiver", "item", "time"]

        print("\nRole recovery test: 'Alice gave Bob the book at noon'")

        correct = 0
        for role, role_name, expected_idx in zip(roles, role_names, range(4)):
            recovered = unbind_hrr(event, role)
            similarities = [similarity(recovered, e) for e in entities]
            best_match = similarities.index(max(similarities))

            if best_match == expected_idx:
                correct += 1
                status = "OK"
            else:
                status = f"WRONG (got {entity_names[best_match]})"

            print(f"  {role_name} -> {entity_names[expected_idx]}: {status}")
            print(f"    similarities: {[f'{s:.3f}' for s in similarities]}")

        accuracy = correct / 4
        print(f"\nRole recovery accuracy: {correct}/4 = {accuracy * 100:.0f}%")

        assert accuracy >= 0.75, (
            f"Expected >= 75% role recovery, got {accuracy * 100:.0f}%"
        )
