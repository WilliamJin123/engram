# tests/test_noise_generators.py
"""Validation tests for noise generation toolkit.

Tests validate INFRA-01 through INFRA-04 requirements:
- INFRA-01: Near-miss generator creates similar-but-unrelated patterns
- INFRA-02: Clutter generator creates random patterns with correct sparsity
- INFRA-03: Fixture accepts composition parameters (near_miss_ratio, clutter_ratio)
- INFRA-04: NoiseLevel presets produce consistent, reproducible results
"""

from __future__ import annotations

import pytest
import torch

from agentic.evolving_pattern import EvolvingPattern
from agentic.memory_store import MemoryStore
from tests.conftest import (
    NoiseConfig,
    NoiseLevel,
    NoiseResult,
    inject_noise,
)
from tests.noise_generators import (
    create_near_miss,
    generate_near_misses,
    create_clutter,
    generate_clutter_batch,
)


class TestNoiseConfig:
    """Tests for NoiseConfig configuration."""

    def test_noise_levels_exist(self) -> None:
        """NoiseLevel has NONE, LOW, MEDIUM, HIGH presets."""
        assert hasattr(NoiseLevel, "NONE")
        assert hasattr(NoiseLevel, "LOW")
        assert hasattr(NoiseLevel, "MEDIUM")
        assert hasattr(NoiseLevel, "HIGH")
        assert len(NoiseLevel) == 4

    def test_preset_none(self) -> None:
        """NONE preset has zero ratios."""
        config = NoiseConfig.from_preset(NoiseLevel.NONE, seed=42)
        assert config.near_miss_ratio == 0.0
        assert config.clutter_ratio == 0.0
        assert config.seed == 42

    def test_preset_low(self) -> None:
        """LOW preset has 0.5 near-miss, 1.0 clutter ratios."""
        config = NoiseConfig.from_preset(NoiseLevel.LOW, seed=42)
        assert config.near_miss_ratio == 0.5
        assert config.clutter_ratio == 1.0

    def test_preset_medium(self) -> None:
        """MEDIUM preset has 1.0 near-miss, 3.0 clutter ratios."""
        config = NoiseConfig.from_preset(NoiseLevel.MEDIUM, seed=42)
        assert config.near_miss_ratio == 1.0
        assert config.clutter_ratio == 3.0

    def test_preset_high(self) -> None:
        """HIGH preset has 2.0 near-miss, 5.0 clutter ratios."""
        config = NoiseConfig.from_preset(NoiseLevel.HIGH, seed=42)
        assert config.near_miss_ratio == 2.0
        assert config.clutter_ratio == 5.0

    def test_seed_required(self) -> None:
        """NoiseConfig requires seed parameter."""
        with pytest.raises(TypeError, match="seed"):
            NoiseConfig(near_miss_ratio=1.0, clutter_ratio=1.0)

    def test_near_miss_overlap_validation(self) -> None:
        """near_miss_overlap must be < 0.5."""
        with pytest.raises(ValueError, match="near_miss_overlap"):
            NoiseConfig(near_miss_overlap=0.5, seed=42)
        with pytest.raises(ValueError, match="near_miss_overlap"):
            NoiseConfig(near_miss_overlap=0.6, seed=42)

    def test_negative_ratio_validation(self) -> None:
        """Ratios must be non-negative."""
        with pytest.raises(ValueError, match="near_miss_ratio"):
            NoiseConfig(near_miss_ratio=-1.0, seed=42)
        with pytest.raises(ValueError, match="clutter_ratio"):
            NoiseConfig(clutter_ratio=-0.5, seed=42)


class TestNearMissGeneration:
    """Tests for near-miss pattern generation."""

    @pytest.fixture
    def target_pattern(self) -> EvolvingPattern:
        """Create a target pattern for testing."""
        return EvolvingPattern.from_text(
            "The quick brown fox jumps over the lazy dog",
            dim=1024,
            k=50,
        )

    @pytest.fixture
    def config(self) -> NoiseConfig:
        """Create test config."""
        return NoiseConfig.from_preset(NoiseLevel.MEDIUM, seed=42)

    def test_near_miss_bit_overlap(
        self, target_pattern: EvolvingPattern, config: NoiseConfig
    ) -> None:
        """Near-miss shares approximately config.near_miss_overlap bits with target."""
        generator = torch.Generator().manual_seed(config.seed)
        near_miss = create_near_miss(target_pattern, config, generator, 0)

        # Calculate actual overlap
        target_bits = target_pattern.bits
        nm_bits = near_miss.bits
        shared = len(target_bits & nm_bits)
        k = len(target_bits)
        actual_overlap = shared / k

        # Should be approximately 0.4 (config.near_miss_overlap) with tolerance
        assert abs(actual_overlap - config.near_miss_overlap) < 0.15, (
            f"Expected overlap ~{config.near_miss_overlap}, got {actual_overlap}"
        )

    def test_near_miss_sparsity(
        self, target_pattern: EvolvingPattern, config: NoiseConfig
    ) -> None:
        """Near-miss has same sparsity (k bits) as target."""
        generator = torch.Generator().manual_seed(config.seed)
        near_miss = create_near_miss(target_pattern, config, generator, 0)

        target_k = len(target_pattern.bits)
        nm_k = len(near_miss.bits)
        assert nm_k == target_k, f"Expected {target_k} bits, got {nm_k}"

    def test_near_miss_aged_coherence(
        self, target_pattern: EvolvingPattern, config: NoiseConfig
    ) -> None:
        """Near-miss coherence is in config.coherence_range, not 1.0."""
        generator = torch.Generator().manual_seed(config.seed)
        near_miss = create_near_miss(target_pattern, config, generator, 0)

        cmin, cmax = config.coherence_range
        assert cmin <= near_miss.coherence <= cmax, (
            f"Coherence {near_miss.coherence} not in range [{cmin}, {cmax}]"
        )
        assert near_miss.coherence < 1.0, "Coherence should be aged (< 1.0)"

    def test_near_miss_reproducibility(
        self, target_pattern: EvolvingPattern, config: NoiseConfig
    ) -> None:
        """Same seed produces identical near-misses."""
        gen1 = torch.Generator().manual_seed(config.seed)
        gen2 = torch.Generator().manual_seed(config.seed)

        nm1 = create_near_miss(target_pattern, config, gen1, 0)
        nm2 = create_near_miss(target_pattern, config, gen2, 0)

        assert nm1.bits == nm2.bits, "Bits differ between same-seed generations"
        assert nm1.coherence == nm2.coherence, "Coherence differs"
        assert nm1.last_access_tick == nm2.last_access_tick, "last_access_tick differs"

    def test_generate_near_misses_count(
        self, target_pattern: EvolvingPattern
    ) -> None:
        """generate_near_misses respects near_miss_ratio."""
        # Test with ratio = 2.0 (always generates 2)
        config = NoiseConfig(near_miss_ratio=2.0, clutter_ratio=0.0, seed=42)
        generator = torch.Generator().manual_seed(config.seed)
        near_misses = generate_near_misses(target_pattern, config, generator)
        assert len(near_misses) == 2

        # Test with ratio = 0.0 (generates none)
        config_zero = NoiseConfig(near_miss_ratio=0.0, clutter_ratio=0.0, seed=42)
        gen_zero = torch.Generator().manual_seed(config_zero.seed)
        nm_zero = generate_near_misses(target_pattern, config_zero, gen_zero)
        assert len(nm_zero) == 0


class TestClutterGeneration:
    """Tests for clutter pattern generation."""

    @pytest.fixture
    def config(self) -> NoiseConfig:
        """Create test config."""
        return NoiseConfig.from_preset(NoiseLevel.MEDIUM, seed=42)

    def test_clutter_sparsity(self, config: NoiseConfig) -> None:
        """Each clutter pattern has exactly k bits."""
        dim, k = 1024, 50
        generator = torch.Generator().manual_seed(config.seed)
        clutter = create_clutter(dim, k, config, generator, None, 0)

        assert len(clutter.bits) == k, f"Expected {k} bits, got {len(clutter.bits)}"

    def test_clutter_varying_relatedness(self, config: NoiseConfig) -> None:
        """Over many clutter patterns, overlap with targets varies."""
        dim, k = 1024, 50
        generator = torch.Generator().manual_seed(config.seed)

        # Create target bits to measure relatedness against
        target_bits = set(range(100))  # Use first 100 bits as "target"

        # Generate 100 clutter patterns
        clutters = generate_clutter_batch(
            dim, k, 100, config, generator, target_bits
        )

        # Calculate overlaps
        overlaps = []
        for c in clutters:
            overlap = len(c.bits & target_bits) / k
            overlaps.append(overlap)

        # Overlaps should vary (not all the same)
        unique_overlaps = len(set(round(o, 2) for o in overlaps))
        assert unique_overlaps > 1, "All clutter has same overlap - not varying"

    def test_clutter_coherence_range(self, config: NoiseConfig) -> None:
        """Clutter coherence is in clutter_coherence_range."""
        dim, k = 1024, 50
        generator = torch.Generator().manual_seed(config.seed)

        clutters = generate_clutter_batch(dim, k, 20, config, generator, None)

        cmin, cmax = config.clutter_coherence_range
        for c in clutters:
            assert cmin <= c.coherence <= cmax, (
                f"Clutter coherence {c.coherence} not in range [{cmin}, {cmax}]"
            )

    def test_clutter_reproducibility(self, config: NoiseConfig) -> None:
        """Same seed produces identical clutter."""
        dim, k = 1024, 50

        gen1 = torch.Generator().manual_seed(config.seed)
        gen2 = torch.Generator().manual_seed(config.seed)

        c1 = create_clutter(dim, k, config, gen1, None, 0)
        c2 = create_clutter(dim, k, config, gen2, None, 0)

        assert c1.bits == c2.bits, "Bits differ"
        assert c1.coherence == c2.coherence, "Coherence differs"
        assert c1.last_access_tick == c2.last_access_tick, "last_access_tick differs"

    def test_generate_clutter_batch_count(self, config: NoiseConfig) -> None:
        """generate_clutter_batch produces requested count."""
        dim, k = 1024, 50
        generator = torch.Generator().manual_seed(config.seed)

        clutters = generate_clutter_batch(dim, k, 7, config, generator, None)
        assert len(clutters) == 7

        # Test zero count
        gen_zero = torch.Generator().manual_seed(config.seed)
        clutters_zero = generate_clutter_batch(dim, k, 0, config, gen_zero, None)
        assert len(clutters_zero) == 0


class TestInjectNoise:
    """Tests for inject_noise function."""

    def test_inject_creates_correct_counts(self) -> None:
        """inject_noise creates expected number of near-misses and clutter."""
        store = MemoryStore(dim=1024, k=50)
        tid = store.store("The quick brown fox")

        config = NoiseConfig.from_preset(NoiseLevel.MEDIUM, seed=42)
        # MEDIUM: near_miss_ratio=1.0, clutter_ratio=3.0
        result = inject_noise(store, [tid], config)

        # With ratio=1.0 and 1 target, should get 1 near-miss
        assert len(result.near_misses.get(tid, [])) == 1

        # With ratio=3.0 and 1 target, should get 3 clutter
        assert len(result.clutter_ids) == 3

        # Total patterns: 1 target + 1 near-miss + 3 clutter = 5
        assert len(store.patterns) == 5

    def test_inject_updates_coherence(self) -> None:
        """Injected patterns have aged coherence, not 1.0."""
        store = MemoryStore(dim=1024, k=50)
        tid = store.store("The quick brown fox")

        config = NoiseConfig.from_preset(NoiseLevel.MEDIUM, seed=42)
        result = inject_noise(store, [tid], config)

        # Check near-miss coherence
        nm_id = result.near_misses[tid][0]
        nm = store.patterns[nm_id]
        cmin, cmax = config.coherence_range
        assert cmin <= nm.coherence <= cmax, (
            f"Near-miss coherence {nm.coherence} not in [{cmin}, {cmax}]"
        )

        # Check clutter coherence
        clutter_id = result.clutter_ids[0]
        clutter = store.patterns[clutter_id]
        ccmin, ccmax = config.clutter_coherence_range
        assert ccmin <= clutter.coherence <= ccmax, (
            f"Clutter coherence {clutter.coherence} not in [{ccmin}, {ccmax}]"
        )

    def test_inject_adds_metadata(self) -> None:
        """Injected patterns have noise_type metadata."""
        store = MemoryStore(dim=1024, k=50)
        tid = store.store("The quick brown fox")

        config = NoiseConfig.from_preset(NoiseLevel.MEDIUM, seed=42)
        result = inject_noise(store, [tid], config)

        # Check near-miss metadata
        nm_id = result.near_misses[tid][0]
        nm = store.patterns[nm_id]
        assert nm.metadata.get("noise_type") == "near_miss"
        assert nm.metadata.get("target") == tid

        # Check clutter metadata
        clutter_id = result.clutter_ids[0]
        clutter = store.patterns[clutter_id]
        assert clutter.metadata.get("noise_type") == "clutter"

    def test_multiple_targets(self) -> None:
        """Multiple targets each get their own near-misses."""
        store = MemoryStore(dim=1024, k=50)
        tid1 = store.store("The quick brown fox")
        tid2 = store.store("A lazy dog sleeps")
        tid3 = store.store("Python programming language")

        config = NoiseConfig.from_preset(NoiseLevel.MEDIUM, seed=42)
        result = inject_noise(store, [tid1, tid2, tid3], config)

        # Each target should have its own near-misses
        assert tid1 in result.near_misses
        assert tid2 in result.near_misses
        assert tid3 in result.near_misses

        # With ratio=1.0, each target should have 1 near-miss
        assert len(result.near_misses[tid1]) == 1
        assert len(result.near_misses[tid2]) == 1
        assert len(result.near_misses[tid3]) == 1

        # Clutter count = 3 targets * 3.0 ratio = 9
        assert len(result.clutter_ids) == 9

    def test_inject_with_none_level(self) -> None:
        """NONE noise level produces no noise patterns."""
        store = MemoryStore(dim=1024, k=50)
        tid = store.store("The quick brown fox")

        config = NoiseConfig.from_preset(NoiseLevel.NONE, seed=42)
        result = inject_noise(store, [tid], config)

        assert len(result.near_misses.get(tid, [])) == 0
        assert len(result.clutter_ids) == 0
        assert len(store.patterns) == 1  # Only target


class TestNoisyMemoryFixture:
    """Tests for noisy_memory pytest fixture."""

    def test_fixture_creates_store_with_noise(self, noisy_memory) -> None:
        """Basic fixture usage creates store with targets and noise."""
        store, target_ids, result = noisy_memory(
            ["The quick brown fox"],
            level=NoiseLevel.MEDIUM,
            seed=42,
        )

        assert isinstance(store, MemoryStore)
        assert len(target_ids) == 1
        # Use duck typing instead of isinstance to avoid module import identity issues
        assert hasattr(result, "near_misses")
        assert hasattr(result, "clutter_ids")
        assert hasattr(result, "config")
        assert len(store.patterns) > 1  # Target + noise

    def test_fixture_respects_noise_level(self, noisy_memory) -> None:
        """Different noise levels produce different noise amounts."""
        # LOW level
        store_low, _, result_low = noisy_memory(
            ["test"], level=NoiseLevel.LOW, seed=42
        )
        noise_low = len(result_low.clutter_ids) + sum(
            len(v) for v in result_low.near_misses.values()
        )

        # HIGH level
        store_high, _, result_high = noisy_memory(
            ["test"], level=NoiseLevel.HIGH, seed=42
        )
        noise_high = len(result_high.clutter_ids) + sum(
            len(v) for v in result_high.near_misses.values()
        )

        # HIGH should have more noise than LOW
        assert noise_high > noise_low, (
            f"HIGH ({noise_high}) should have more noise than LOW ({noise_low})"
        )

    def test_fixture_returns_target_ids(self, noisy_memory) -> None:
        """Returned target_ids match stored targets."""
        target_texts = ["quick brown fox", "lazy dog", "python code"]
        store, target_ids, result = noisy_memory(
            target_texts,
            level=NoiseLevel.LOW,
            seed=42,
        )

        assert len(target_ids) == 3

        # Each target ID should exist in store
        for tid in target_ids:
            pattern = store.get(tid)
            assert pattern is not None
            # Target patterns should NOT have noise_type metadata
            assert pattern.metadata.get("noise_type") is None

    def test_fixture_custom_config(self, noisy_memory) -> None:
        """Fixture accepts custom NoiseConfig."""
        custom_config = NoiseConfig(
            near_miss_ratio=5.0,
            clutter_ratio=10.0,
            seed=123,
        )
        store, target_ids, result = noisy_memory(
            ["test"],
            config=custom_config,
        )

        # With 5.0 near-miss ratio, should get 5 near-misses
        assert len(result.near_misses[target_ids[0]]) == 5
        # With 10.0 clutter ratio, should get 10 clutter
        assert len(result.clutter_ids) == 10

    def test_fixture_custom_dimensions(self, noisy_memory) -> None:
        """Fixture accepts custom dim and k."""
        store, target_ids, result = noisy_memory(
            ["test"],
            level=NoiseLevel.LOW,
            seed=42,
            dim=512,
            k=25,
        )

        # Check store dimensions
        assert store.dim == 512
        assert store.k == 25

        # Check pattern dimensions
        pattern = store.get(target_ids[0])
        assert pattern is not None
        assert pattern.dim == 512
        assert len(pattern.bits) == 25


@pytest.mark.parametrize("level", list(NoiseLevel))
class TestNoiseLevelPresets:
    """Parameterized tests across all noise levels."""

    def test_preset_reproducibility(self, level: NoiseLevel) -> None:
        """Each preset produces consistent results with same seed."""
        config1 = NoiseConfig.from_preset(level, seed=42)
        config2 = NoiseConfig.from_preset(level, seed=42)

        assert config1.near_miss_ratio == config2.near_miss_ratio
        assert config1.clutter_ratio == config2.clutter_ratio
        assert config1.near_miss_overlap == config2.near_miss_overlap

    def test_preset_creates_valid_config(self, level: NoiseLevel) -> None:
        """Each preset creates a valid NoiseConfig."""
        config = NoiseConfig.from_preset(level, seed=42)

        assert config.near_miss_ratio >= 0
        assert config.clutter_ratio >= 0
        assert 0 < config.near_miss_overlap < 0.5
        assert config.seed == 42

    def test_preset_injection_works(self, level: NoiseLevel) -> None:
        """Each preset can be used with inject_noise."""
        store = MemoryStore(dim=1024, k=50)
        tid = store.store("test pattern")

        config = NoiseConfig.from_preset(level, seed=42)
        result = inject_noise(store, [tid], config)

        # Should complete without error
        assert isinstance(result, NoiseResult)
        # Near-misses dict should exist for target
        assert tid in result.near_misses
