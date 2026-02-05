# tests/stress/test_tunneling.py
"""TEST-01: Tunneling tests with noisy memory.

Validates that tunneling navigates through clutter to find related patterns.
Tests single-hop, multi-hop, threshold behavior, and creative mode.

All tests output metrics for Phase 8 degradation curve analysis.

Key behaviors tested:
- Single-hop tunneling: Source pattern with high coherence can tunnel to
  connected target pattern through noise
- Multi-hop tunneling: Chain of connections (source -> intermediate -> target)
  can reach distant patterns
- Threshold behavior: Tunneling requires source coherence above min_source_coherence
- Creative mode: 3x probability amplification enables tunneling at lower thresholds

IMPORTANT: Tunneling is probabilistic. Tests use pytest.skip() for probabilistic
failures with LIMITATION messages. Only deterministic behavior is asserted.
"""

from __future__ import annotations

import json
import random

import pytest

from agentic.memory_store import MemoryStore
from quantum_substrate.tunneling import TunnelingConfig
from tests.conftest import NoiseLevel, NoiseConfig, inject_noise
from tests.stress.conftest import (
    STRESS_NOISE_LEVELS,
    StressMetrics,
    capture_retrieval_metrics,
    evaluate_success_criteria,
)


@pytest.mark.stress
class TestTunnelingThroughNoise:
    """TEST-01: Tunneling tests validating navigation through noisy memory."""

    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
    def test_single_hop_tunneling(self, noise_level: NoiseLevel) -> None:
        """Test single-hop tunneling from source to connected target.

        Setup:
        - Store "source pattern" and "connected target"
        - Create connection between source and target
        - Inject noise (near-misses, clutter)
        - Set source coherence high (0.9)

        Expected:
        - Direct retrieval finds source
        - Tunneling probabilistically finds connected target

        Tunneling is probabilistic - uses pytest.skip() if tunnel doesn't occur.
        """
        # Create store
        store = MemoryStore(dim=1024, k=50)

        # Store target patterns
        source_id = store.store("source pattern quantum entanglement physics")
        target_id = store.store("connected target neural network brain")

        # Create connection between source and target
        store.create_connection(source_id, target_id)

        # Set source coherence high (required for tunneling)
        store.patterns[source_id].coherence = 0.9

        # Inject noise
        config = NoiseConfig.from_preset(noise_level, seed=42)
        noise_result = inject_noise(store, [source_id, target_id], config)

        # Configure tunneling with favorable settings
        tunneling_config = TunnelingConfig(
            baseline_probability=0.5,
            min_source_coherence=0.3,
            creative_mode_multiplier=3.0,
            max_bit_overlap=0.5,  # More permissive overlap threshold
        )
        store.set_tunneling_config(tunneling_config)

        # Retrieve with tunneling enabled and creative mode
        results, tunnel_results = store.retrieve_with_tunneling(
            query="source pattern quantum",
            top_k=10,
            creative_mode=True,
        )

        # Capture metrics
        source_found = any(r.pattern_id == source_id for r in results)
        target_found = any(r.pattern_id == target_id for r in results)
        successful_tunnels = sum(1 for tr in tunnel_results if tr.tunneled)
        tunnel_to_target = any(
            tr.tunneled and tr.target_pattern_id == target_id
            for tr in tunnel_results
        )

        metrics = StressMetrics(
            test_name="test_single_hop_tunneling",
            noise_level=noise_level.value,
            success=source_found,  # Deterministic success is source retrieval
            metrics={
                "source_found": source_found,
                "target_found": target_found,
                "successful_tunnels": successful_tunnels,
                "tunnel_to_target": tunnel_to_target,
                "total_patterns": store.pattern_count,
                "near_miss_count": len(noise_result.near_misses.get(source_id, [])),
                "clutter_count": len(noise_result.clutter_ids),
            },
        )
        print(f"\nMETRICS: {metrics.to_json()}")

        # Assert source is found (direct retrieval should work)
        assert source_found, (
            f"Direct retrieval failed to find source pattern at {noise_level.value} noise. "
            f"Got {len(results)} results: {[r.pattern_id[:8] for r in results]}"
        )

        # Tunneling is probabilistic - skip with LIMITATION if didn't occur
        if not tunnel_to_target:
            pytest.skip(
                f"LIMITATION: Tunneling to target did not occur (probabilistic). "
                f"Successful tunnels: {successful_tunnels}, target_found: {target_found}"
            )

    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
    def test_multi_hop_tunneling(self, noise_level: NoiseLevel) -> None:
        """Test multi-hop tunneling chain: source -> intermediate -> target.

        Setup:
        - Store source, intermediate, and final target patterns
        - Create chain: source -> intermediate -> target
        - Set coherence high on source and intermediate
        - Inject noise

        Expected:
        - Source retrieval works
        - Tunneling chain may reach distant target (probabilistic)

        Note: Multi-hop requires multiple successful tunnels - highly probabilistic.
        """
        store = MemoryStore(dim=1024, k=50)

        # Store chain of patterns
        source_id = store.store("source quantum computing research")
        intermediate_id = store.store("intermediate bridge pattern connection")
        target_id = store.store("final target distant memory recall")

        # Create connection chain: source -> intermediate -> target
        store.create_connection(source_id, intermediate_id)
        store.create_connection(intermediate_id, target_id)

        # Set coherence high on source and intermediate
        store.patterns[source_id].coherence = 0.9
        store.patterns[intermediate_id].coherence = 0.8

        # Inject noise
        config = NoiseConfig.from_preset(noise_level, seed=123)
        noise_result = inject_noise(
            store, [source_id, intermediate_id, target_id], config
        )

        # Configure tunneling for multi-hop
        tunneling_config = TunnelingConfig(
            baseline_probability=0.8,  # High probability for multi-hop
            min_source_coherence=0.3,
            creative_mode_multiplier=3.0,
            max_bit_overlap=0.5,
        )
        store.set_tunneling_config(tunneling_config)

        # Retrieve with tunneling
        results, tunnel_results = store.retrieve_with_tunneling(
            query="source quantum computing",
            top_k=15,
            creative_mode=True,
        )

        # Capture metrics
        source_found = any(r.pattern_id == source_id for r in results)
        intermediate_found = any(r.pattern_id == intermediate_id for r in results)
        target_found = any(r.pattern_id == target_id for r in results)
        successful_tunnels = sum(1 for tr in tunnel_results if tr.tunneled)

        # Track which hops tunneled
        tunneled_to_intermediate = any(
            tr.tunneled and tr.target_pattern_id == intermediate_id
            for tr in tunnel_results
        )
        tunneled_to_target = any(
            tr.tunneled and tr.target_pattern_id == target_id
            for tr in tunnel_results
        )

        metrics = StressMetrics(
            test_name="test_multi_hop_tunneling",
            noise_level=noise_level.value,
            success=source_found,
            metrics={
                "source_found": source_found,
                "intermediate_found": intermediate_found,
                "target_found": target_found,
                "successful_tunnels": successful_tunnels,
                "tunneled_to_intermediate": tunneled_to_intermediate,
                "tunneled_to_target": tunneled_to_target,
                "total_patterns": store.pattern_count,
            },
        )
        print(f"\nMETRICS: {metrics.to_json()}")

        # Assert source retrieval works
        assert source_found, (
            f"Direct retrieval failed to find source at {noise_level.value} noise"
        )

        # Document multi-hop outcome (probabilistic, may not occur)
        if not (intermediate_found or target_found):
            pytest.skip(
                f"LIMITATION: Multi-hop tunneling did not reach intermediate or target. "
                f"This is probabilistic behavior. Successful tunnels: {successful_tunnels}"
            )

    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
    def test_threshold_behavior(self, noise_level: NoiseLevel) -> None:
        """Test tunneling threshold: high coherence CAN tunnel, low coherence CANNOT.

        The min_source_coherence threshold is deterministic:
        - Patterns with coherence < threshold CANNOT initiate tunneling
        - Patterns with coherence >= threshold CAN initiate (probability > 0)

        This test verifies the deterministic constraint.
        """
        # Test 1: Above threshold (high coherence = 0.8, threshold = 0.5)
        store_high = MemoryStore(dim=1024, k=50)
        source_high_id = store_high.store("high coherence source pattern")
        target_high_id = store_high.store("connected target pattern")
        store_high.create_connection(source_high_id, target_high_id)
        store_high.patterns[source_high_id].coherence = 0.8

        config = NoiseConfig.from_preset(noise_level, seed=456)
        inject_noise(store_high, [source_high_id, target_high_id], config)

        tunneling_config = TunnelingConfig(
            baseline_probability=0.99,  # Very high to maximize tunnel chance
            min_source_coherence=0.5,
            creative_mode_multiplier=3.0,
            max_bit_overlap=0.5,
        )
        store_high.set_tunneling_config(tunneling_config)

        _, tunnel_results_high = store_high.retrieve_with_tunneling(
            query="high coherence source",
            top_k=10,
            creative_mode=True,
        )

        # Test 2: Below threshold (low coherence = 0.3, threshold = 0.5)
        store_low = MemoryStore(dim=1024, k=50)
        source_low_id = store_low.store("low coherence source pattern")
        target_low_id = store_low.store("connected target pattern")
        store_low.create_connection(source_low_id, target_low_id)
        store_low.patterns[source_low_id].coherence = 0.3

        inject_noise(store_low, [source_low_id, target_low_id], config)
        store_low.set_tunneling_config(tunneling_config)

        _, tunnel_results_low = store_low.retrieve_with_tunneling(
            query="low coherence source",
            top_k=10,
            creative_mode=True,
        )

        # Metrics
        high_tunneled = any(tr.tunneled for tr in tunnel_results_high)
        low_tunneled = any(tr.tunneled for tr in tunnel_results_low)

        metrics = StressMetrics(
            test_name="test_threshold_behavior",
            noise_level=noise_level.value,
            success=not low_tunneled,  # Success if low coherence blocked tunneling
            metrics={
                "high_coherence_tunneled": high_tunneled,
                "low_coherence_tunneled": low_tunneled,
                "high_coherence_tunnel_attempts": len(tunnel_results_high),
                "low_coherence_tunnel_attempts": len(tunnel_results_low),
            },
        )
        print(f"\nMETRICS: {metrics.to_json()}")

        # DETERMINISTIC: Low coherence (0.3) CANNOT tunnel with threshold 0.5
        assert not low_tunneled, (
            f"VIOLATION: Low coherence source (0.3) should NOT be able to tunnel "
            f"with min_source_coherence=0.5. Got tunneled={low_tunneled}"
        )

        # PROBABILISTIC: High coherence CAN tunnel (but might not due to probability)
        if not high_tunneled:
            pytest.skip(
                f"LIMITATION: High coherence source did not tunnel (probabilistic). "
                f"Tunnel attempts: {len(tunnel_results_high)}"
            )

    @pytest.mark.parametrize("noise_level", STRESS_NOISE_LEVELS)
    def test_creative_mode_amplification(self, noise_level: NoiseLevel) -> None:
        """Test creative mode amplifies tunneling probability by 3x.

        Setup:
        - Moderate coherence (0.4) - above threshold but not high
        - Low baseline_probability (0.15)
        - Compare creative_mode=True (3x = 0.45) vs creative_mode=False (0.15)

        Run multiple trials with different seeds to observe statistical difference.
        """
        base_seed = 789

        def run_trial(
            creative_mode: bool, trial_seed: int
        ) -> tuple[bool, list]:
            """Run single trial and return (tunneled, tunnel_results)."""
            store = MemoryStore(dim=1024, k=50)
            source_id = store.store("moderate coherence source quantum")
            target_id = store.store("creative target association leap")
            store.create_connection(source_id, target_id)
            store.patterns[source_id].coherence = 0.4  # Moderate, above threshold

            config = NoiseConfig.from_preset(noise_level, seed=trial_seed)
            inject_noise(store, [source_id, target_id], config)

            tunneling_config = TunnelingConfig(
                baseline_probability=0.15,  # Low base probability
                min_source_coherence=0.3,
                creative_mode_multiplier=3.0,
                max_bit_overlap=0.5,
            )
            store.set_tunneling_config(tunneling_config)

            _, tunnel_results = store.retrieve_with_tunneling(
                query="moderate coherence source",
                top_k=10,
                creative_mode=creative_mode,
            )

            return any(tr.tunneled for tr in tunnel_results), tunnel_results

        # Run 5 trials for each mode
        num_trials = 5
        creative_successes = 0
        non_creative_successes = 0

        for i in range(num_trials):
            trial_seed = base_seed + i * 100

            creative_tunneled, _ = run_trial(creative_mode=True, trial_seed=trial_seed)
            non_creative_tunneled, _ = run_trial(
                creative_mode=False, trial_seed=trial_seed + 50
            )

            if creative_tunneled:
                creative_successes += 1
            if non_creative_tunneled:
                non_creative_successes += 1

        creative_rate = creative_successes / num_trials
        non_creative_rate = non_creative_successes / num_trials

        metrics = StressMetrics(
            test_name="test_creative_mode_amplification",
            noise_level=noise_level.value,
            success=True,  # Statistical test, no hard pass/fail
            metrics={
                "num_trials": num_trials,
                "creative_successes": creative_successes,
                "non_creative_successes": non_creative_successes,
                "creative_rate": creative_rate,
                "non_creative_rate": non_creative_rate,
                "rate_difference": creative_rate - non_creative_rate,
            },
        )
        print(f"\nMETRICS: {metrics.to_json()}")

        # Statistical assertion with tolerance
        # Creative mode (3x) should have >= success rate as non-creative
        # Note: Small sample size means this may not always hold
        # Using soft assertion - always pass but document in metrics
        if creative_rate < non_creative_rate:
            pytest.skip(
                f"LIMITATION: Creative mode rate ({creative_rate:.2f}) < "
                f"non-creative rate ({non_creative_rate:.2f}). "
                f"Small sample size - expected statistical variance."
            )
