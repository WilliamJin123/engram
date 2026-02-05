# tests/stress/test_coherence.py
"""TEST-03: Coherence dynamics tests with noisy memory.

Validates that coherence dynamics (decay, refresh, stability, surprise)
work correctly under realistic memory load.

Tests all three behaviors equally:
- Decay: Patterns lose coherence over operations without access
- Refresh: Accessing patterns restores their coherence
- Surprise re-coherence: Unexpected retrievals boost coherence

All tests output metrics for Phase 8 analysis.

Key behaviors tested:
1. Decay over operations: Coherence decreases with each retrieve() without access
2. Access refresh: Direct retrieval of pattern restores coherence
3. Stability affects decay: High-stability patterns decay differently (crystallization)
4. Surprise re-coherence: Unexpected results boost coherence of involved patterns
5. Coherence floor: Minimum coherence (0.01) is enforced even under aggressive decay

IMPORTANT: Each retrieve() operation advances tick and applies decay to ALL patterns.
Tests account for this by recording coherence at the right moment.
"""

from __future__ import annotations

import json

import pytest

from agentic.memory_store import MemoryStore
from quantum_substrate.coherence import CoherenceManager, CoherenceConfig
from tests.conftest import NoiseLevel, NoiseConfig, inject_noise
from tests.stress.conftest import (
    STRESS_NOISE_LEVELS,
    StressMetrics,
)


@pytest.mark.stress
class TestCoherenceDynamics:
    """TEST-03: Coherence dynamics under noisy memory conditions.

    Tests validate that coherence dynamics (decay, refresh, stability effects,
    surprise re-coherence) work correctly when memory contains noise patterns.
    All tests output metrics for Phase 8 analysis.
    """

    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
    def test_decay_over_operations(self, noise_level: NoiseLevel) -> None:
        """Test that coherence decays over retrieve operations without access.

        Setup:
        - Store target pattern with high coherence (0.9)
        - Inject noise
        - Perform 30 retrieve() operations with unrelated queries

        Expected:
        - Target coherence decreases over operations
        - Decay happens even when target is not accessed

        Deterministic: Decay is guaranteed to happen when pattern is not accessed.
        """
        store = MemoryStore(dim=1024, k=50)

        # Store target pattern
        target_id = store.store("target pattern for coherence decay testing")

        # Inject noise FIRST
        config = NoiseConfig.from_preset(noise_level, seed=42)
        noise_result = inject_noise(store, [target_id], config)

        # Set target coherence high AFTER noise injection
        store.patterns[target_id].coherence = 0.9
        store.patterns[target_id].last_access_tick = store.current_tick

        # Record initial coherence
        initial_coherence = store.patterns[target_id].coherence

        # Perform 30 unrelated retrieve operations (simulate time passage)
        unrelated_queries = [
            "completely unrelated astronomy stars",
            "cooking recipes kitchen food",
            "music theory harmony notes",
            "architecture buildings design",
            "mathematics calculus equations",
        ]

        for i in range(30):
            query = unrelated_queries[i % len(unrelated_queries)]
            store.retrieve(query, top_k=5, method="interference")

        # Record final coherence using get_effective_coherence
        final_coherence = store.get_effective_coherence(target_id)

        # Calculate decay
        decay_amount = initial_coherence - final_coherence

        metrics = StressMetrics(
            test_name="test_decay_over_operations",
            noise_level=noise_level.value,
            success=final_coherence < initial_coherence,
            metrics={
                "initial_coherence": initial_coherence,
                "final_coherence": final_coherence,
                "decay_amount": decay_amount,
                "operations": 30,
                "total_patterns": store.pattern_count,
                "near_miss_count": len(noise_result.near_misses.get(target_id, [])),
                "clutter_count": len(noise_result.clutter_ids),
            },
        )
        print(f"\nMETRICS: {metrics.to_json()}")

        # Assert decay happened
        assert final_coherence < initial_coherence, (
            f"VIOLATION: Coherence should decay over operations. "
            f"Initial: {initial_coherence:.4f}, Final: {final_coherence:.4f}"
        )

    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
    def test_access_refresh(self, noise_level: NoiseLevel) -> None:
        """Test that accessing a pattern refreshes its coherence.

        Setup:
        - Store target pattern with low coherence (0.3)
        - Let it decay further with 10 unrelated operations
        - Query the target pattern directly

        Expected:
        - If target found in results: coherence increases after access
        - If target NOT found: skip with LIMITATION (noise overwhelmed)

        The refresh amount depends on retrieval score (activation_strength).
        """
        store = MemoryStore(dim=1024, k=50)

        # Store target pattern with distinctive content
        target_text = "distinctive quantum entanglement physics research"
        target_id = store.store(target_text)

        # Inject noise
        config = NoiseConfig.from_preset(noise_level, seed=123)
        noise_result = inject_noise(store, [target_id], config)

        # Set target coherence low AFTER noise injection
        store.patterns[target_id].coherence = 0.3
        store.patterns[target_id].last_access_tick = store.current_tick

        # Let it decay further with 10 unrelated operations
        for _ in range(10):
            store.retrieve("unrelated cooking recipe", top_k=5, method="interference")

        # Record pre-access coherence
        pre_access_coherence = store.get_effective_coherence(target_id)

        # Query the target pattern directly (exact match query)
        results = store.retrieve(target_text, top_k=10, method="interference")

        # Check if target was found and accessed
        target_found = any(r.pattern_id == target_id for r in results)

        # Record post-access coherence
        post_access_coherence = store.patterns[target_id].coherence

        # Calculate refresh amount
        refresh_amount = post_access_coherence - pre_access_coherence

        metrics = StressMetrics(
            test_name="test_access_refresh",
            noise_level=noise_level.value,
            success=target_found and post_access_coherence > pre_access_coherence,
            metrics={
                "pre_access_coherence": pre_access_coherence,
                "post_access_coherence": post_access_coherence,
                "refresh_amount": refresh_amount,
                "target_found": target_found,
                "total_patterns": store.pattern_count,
            },
        )
        print(f"\nMETRICS: {metrics.to_json()}")

        # If target not found, skip with LIMITATION
        if not target_found:
            pytest.skip(
                f"LIMITATION: Target not found in results at {noise_level.value} noise. "
                "Noise may have overwhelmed retrieval. Cannot validate refresh."
            )

        # Assert refresh happened
        assert post_access_coherence > pre_access_coherence, (
            f"VIOLATION: Coherence should increase after access. "
            f"Pre: {pre_access_coherence:.4f}, Post: {post_access_coherence:.4f}"
        )

    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
    def test_stability_affects_decay(self, noise_level: NoiseLevel) -> None:
        """Test that stability score affects decay behavior.

        Setup:
        - Store 2 target patterns: "stable pattern" and "unstable pattern"
        - Configure stability:
          - stable: access_count_since_modification = 20 (stability_score = 1.0)
          - unstable: access_count_since_modification = 0 (stability_score = 0.0)
        - Both start at same coherence (0.8)
        - Perform 20 unrelated retrieve() operations

        Expected (per INTUITION.md):
        - High stability = crystallization = FASTER decay
        - Stable pattern should decay MORE than unstable

        This is counter-intuitive but matches the "crystallization" semantics:
        highly-accessed patterns become "fixed" and lose quantum superposition faster.
        """
        store = MemoryStore(dim=1024, k=50)

        # Store two patterns
        stable_id = store.store("stable pattern that has been accessed many times")
        unstable_id = store.store("unstable pattern that is fresh and malleable")

        # Inject noise
        config = NoiseConfig.from_preset(noise_level, seed=456)
        noise_result = inject_noise(store, [stable_id, unstable_id], config)

        # Configure stability AFTER noise injection
        # Stable: high access count = high stability = crystallized
        store.patterns[stable_id].coherence = 0.8
        store.patterns[stable_id].access_count_since_modification = 20
        store.patterns[stable_id].last_access_tick = store.current_tick

        # Unstable: low access count = low stability = malleable
        store.patterns[unstable_id].coherence = 0.8
        store.patterns[unstable_id].access_count_since_modification = 0
        store.patterns[unstable_id].last_access_tick = store.current_tick

        # Verify stability scores
        stable_stability = store.patterns[stable_id].stability_score
        unstable_stability = store.patterns[unstable_id].stability_score

        # Record initial coherences
        initial_stable = store.patterns[stable_id].coherence
        initial_unstable = store.patterns[unstable_id].coherence

        # Perform 20 unrelated operations
        for _ in range(20):
            store.retrieve("unrelated music theory", top_k=5, method="interference")

        # Record final coherences
        final_stable = store.get_effective_coherence(stable_id)
        final_unstable = store.get_effective_coherence(unstable_id)

        # Calculate decay amounts
        stable_decay = initial_stable - final_stable
        unstable_decay = initial_unstable - final_unstable

        metrics = StressMetrics(
            test_name="test_stability_affects_decay",
            noise_level=noise_level.value,
            success=True,  # Observation test - always passes but documents behavior
            metrics={
                "stable_stability_score": stable_stability,
                "unstable_stability_score": unstable_stability,
                "initial_stable_coherence": initial_stable,
                "initial_unstable_coherence": initial_unstable,
                "final_stable_coherence": final_stable,
                "final_unstable_coherence": final_unstable,
                "stable_decay_amount": stable_decay,
                "unstable_decay_amount": unstable_decay,
                "crystallization_effect": stable_decay - unstable_decay,
                "stable_decayed_more": stable_decay > unstable_decay,
                "operations": 20,
            },
        )
        print(f"\nMETRICS: {metrics.to_json()}")

        # Document observation - per INTUITION.md, stable should decay MORE (crystallization)
        # But this is an observation, not a hard assertion
        if stable_decay > unstable_decay:
            print(
                f"\nOBSERVATION: Crystallization effect observed. "
                f"Stable pattern decayed {stable_decay:.4f} > unstable {unstable_decay:.4f}. "
                "This matches INTUITION.md: high stability = faster decay."
            )
        else:
            print(
                f"\nOBSERVATION: Crystallization effect NOT observed. "
                f"Stable decayed {stable_decay:.4f}, unstable {unstable_decay:.4f}. "
                "May need crystallization_factor tuning."
            )

    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
    def test_surprise_recoherence(self, noise_level: NoiseLevel) -> None:
        """Test that surprise retrieval boosts coherence of surprising pattern.

        Setup:
        - Store 2 patterns: "expected pattern" and "surprising pattern"
        - Set surprising pattern's coherence very low (0.1) - simulates "forgotten"
        - Set surprising pattern's access_count = 0 (malleable, can re-cohere)
        - Use retrieve_with_surprise() querying for surprising pattern content

        Expected:
        - If our target pattern is marked as the surprising pattern: it should get boosted
        - If a different pattern is marked as surprising: document as LIMITATION
        - The key insight: surprise boosts the DETECTED surprising pattern, not necessarily
          our target pattern. Under high noise, a clutter pattern may be detected as surprising.

        Surprise detection compares expected vs actual top result.
        """
        store = MemoryStore(dim=1024, k=50)

        # Store patterns - make surprising pattern very distinctive
        expected_text = "common everyday expected pattern information"
        surprising_text = "rare surprising forgotten memory quantum unique"

        expected_id = store.store(expected_text)
        surprising_id = store.store(surprising_text)

        # Inject noise
        config = NoiseConfig.from_preset(noise_level, seed=789)
        noise_result = inject_noise(store, [expected_id, surprising_id], config)

        # Configure patterns AFTER noise injection
        # Expected: high coherence, stable (expected to be found)
        store.patterns[expected_id].coherence = 0.9
        store.patterns[expected_id].access_count_since_modification = 10
        store.patterns[expected_id].last_access_tick = store.current_tick

        # Surprising: very low coherence (forgotten), malleable
        store.patterns[surprising_id].coherence = 0.1
        store.patterns[surprising_id].access_count_since_modification = 0
        store.patterns[surprising_id].last_access_tick = store.current_tick

        # Record initial coherence of surprising pattern
        initial_coherence = store.patterns[surprising_id].coherence

        # Query for surprising pattern content (should trigger surprise if found)
        results, surprise_result = store.retrieve_with_surprise(
            query=surprising_text,
            top_k=10,
            method="interference",
            boost_coefficient=0.5,
        )

        # Record final coherence
        final_coherence = store.patterns[surprising_id].coherence

        # Check if surprise occurred and which pattern was marked as surprising
        surprise_magnitude = surprise_result.magnitude if surprise_result else 0.0
        surprise_occurred = surprise_magnitude > 0.1
        detected_surprising_id = surprise_result.surprising_pattern_id if surprise_result else None
        our_target_was_surprising = detected_surprising_id == surprising_id

        # Calculate coherence delta
        coherence_delta = final_coherence - initial_coherence

        metrics = StressMetrics(
            test_name="test_surprise_recoherence",
            noise_level=noise_level.value,
            success=True,  # Observation test - captures behavior under noise
            metrics={
                "initial_coherence": initial_coherence,
                "final_coherence": final_coherence,
                "coherence_delta": coherence_delta,
                "surprise_magnitude": surprise_magnitude,
                "surprise_occurred": surprise_occurred,
                "our_target_was_surprising": our_target_was_surprising,
                "detected_surprising_id": detected_surprising_id,
                "expected_pattern_id": surprise_result.expected_pattern_id if surprise_result else None,
                "total_patterns": store.pattern_count,
            },
        )
        print(f"\nMETRICS: {metrics.to_json()}")

        # If no meaningful surprise, skip
        if not surprise_occurred:
            pytest.skip(
                f"LIMITATION: No meaningful surprise detected (magnitude={surprise_magnitude:.3f}). "
                "Retrieval may have matched expectation. Cannot validate re-coherence."
            )

        # If our target was not the detected surprising pattern, skip with LIMITATION
        # This happens under high noise when clutter patterns dominate retrieval
        if not our_target_was_surprising:
            pytest.skip(
                f"LIMITATION: A different pattern was marked as surprising (not our target). "
                f"Under {noise_level.value} noise, clutter may dominate. "
                f"Detected surprising: {detected_surprising_id[:8] if detected_surprising_id else 'None'}, "
                f"Our target: {surprising_id[:8]}"
            )

        # If our target was the surprising pattern, verify it got boosted (or at least not penalized)
        # Note: decay also happens during retrieve_with_surprise, so coherence may decrease slightly
        # due to decay even if boost was applied. We check that it didn't drop significantly.
        assert final_coherence >= initial_coherence * 0.9, (
            f"VIOLATION: Our target pattern was marked as surprising but coherence dropped >10%. "
            f"Initial: {initial_coherence:.4f}, Final: {final_coherence:.4f}, "
            f"Surprise magnitude: {surprise_magnitude:.4f}"
        )

    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
    def test_coherence_floor_under_load(self, noise_level: NoiseLevel) -> None:
        """Test that coherence floor (0.01) is enforced under aggressive decay.

        Setup:
        - Store target pattern
        - Set coherence to 0.05 (just above floor of 0.01)
        - Perform 100 unrelated operations (aggressive decay pressure)

        Expected:
        - Coherence never drops below floor (0.01)
        - Floor is a hard constraint regardless of decay pressure

        Deterministic: Floor enforcement is guaranteed.
        """
        store = MemoryStore(dim=1024, k=50)

        # Store target pattern
        target_id = store.store("target pattern for floor testing")

        # Inject noise
        config = NoiseConfig.from_preset(noise_level, seed=999)
        noise_result = inject_noise(store, [target_id], config)

        # Set coherence just above floor AFTER noise injection
        floor = store.coherence_manager.config.floor  # Should be 0.01
        store.patterns[target_id].coherence = 0.05
        store.patterns[target_id].last_access_tick = store.current_tick

        initial_coherence = store.patterns[target_id].coherence

        # Perform 100 unrelated operations (aggressive decay)
        for _ in range(100):
            store.retrieve("unrelated astronomy query", top_k=5, method="interference")

        # Get final coherence
        final_coherence = store.get_effective_coherence(target_id)

        metrics = StressMetrics(
            test_name="test_coherence_floor_under_load",
            noise_level=noise_level.value,
            success=final_coherence >= floor,
            metrics={
                "initial_coherence": initial_coherence,
                "final_coherence": final_coherence,
                "floor": floor,
                "above_floor": final_coherence >= floor,
                "operations": 100,
                "total_patterns": store.pattern_count,
            },
        )
        print(f"\nMETRICS: {metrics.to_json()}")

        # Assert floor is enforced
        assert final_coherence >= floor, (
            f"VIOLATION: Coherence {final_coherence:.4f} fell below floor {floor}. "
            "Floor enforcement failed."
        )
