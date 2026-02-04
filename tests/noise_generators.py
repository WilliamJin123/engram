# tests/noise_generators.py
"""Noise generation functions for stress-testing retrieval.

This module provides generators for two types of noise:
1. Near-misses: Patterns that share bits with targets but have different phases
2. Clutter: Random patterns matching sparsity distribution with varying relatedness

All functions use torch.Generator for reproducible seeded random operations.

Usage:
    import torch
    from agentic.evolving_pattern import EvolvingPattern
    from tests.noise_generators import generate_near_misses, generate_clutter_batch
    from tests.conftest import NoiseConfig, NoiseLevel

    target = EvolvingPattern.from_text("test", dim=1024, k=50)
    config = NoiseConfig.from_preset(NoiseLevel.MEDIUM, seed=42)
    gen = torch.Generator().manual_seed(config.seed)

    near_misses = generate_near_misses(target, config, gen)
    clutter = generate_clutter_batch(1024, 50, 10, config, gen)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from agentic.evolving_pattern import EvolvingPattern

if TYPE_CHECKING:
    from tests.conftest import NoiseConfig


def create_near_miss(
    target: EvolvingPattern,
    config: "NoiseConfig",
    generator: torch.Generator,
    index: int,
) -> EvolvingPattern:
    """Create a near-miss pattern for a target.

    Near-misses share bit overlap with the target but have:
    - Different phase relationships (won't constructively interfere)
    - Aged coherence (simulates realistic memory)
    - Different text/metadata (semantically unrelated)

    Args:
        target: The target pattern to create a near-miss for
        config: NoiseConfig with overlap and phase noise parameters
        generator: Seeded torch.Generator for reproducibility
        index: Index for naming (e.g., near_miss_0, near_miss_1)

    Returns:
        EvolvingPattern that is a near-miss of the target
    """
    k = len(target.bits)
    num_shared = int(config.near_miss_overlap * k)

    # Select random subset of target bits to share
    target_bits_list = list(target.bits)
    perm = torch.randperm(k, generator=generator)
    shared_bits = {target_bits_list[i] for i in perm[:num_shared].tolist()}

    # Generate new bits from non-target positions
    available = [i for i in range(target.dim) if i not in target.bits]
    num_new = k - num_shared
    new_perm = torch.randperm(len(available), generator=generator)
    new_bits = {available[i] for i in new_perm[:num_new].tolist()}

    all_bits = shared_bits | new_bits

    # Create phases with perturbation
    phases: dict[int, float] = {}
    for bit in all_bits:
        if bit in target.phases:
            # Shared bit: perturb target's phase by Gaussian noise
            base_phase = target.phases[bit]
            noise = (
                torch.randn(1, generator=generator).item()
                * config.near_miss_phase_noise
            )
            phases[bit] = (base_phase + noise) % (2 * math.pi)
        else:
            # New bit: random phase
            phases[bit] = torch.rand(1, generator=generator).item() * 2 * math.pi

    # Aged coherence (not fresh 1.0)
    cmin, cmax = config.coherence_range
    coherence = cmin + torch.rand(1, generator=generator).item() * (cmax - cmin)

    # Random last_access_tick to simulate aging
    last_access_tick = int(torch.randint(1, 51, (1,), generator=generator).item())

    # Truncate target text for name
    target_text_short = target.text[:30] if target.text else "unknown"

    return EvolvingPattern(
        dim=target.dim,
        bits=all_bits,
        original_bits=frozenset(all_bits),
        acquired_bits=set(),
        phases=phases,
        text=f"near_miss_{index}_of_{target_text_short}",
        coherence=coherence,
        last_access_tick=last_access_tick,
        connection_count=0,
        last_modified_tick=0,
        access_count_since_modification=int(
            torch.randint(0, 5, (1,), generator=generator).item()
        ),
    )


def generate_near_misses(
    target: EvolvingPattern,
    config: "NoiseConfig",
    generator: torch.Generator,
) -> list[EvolvingPattern]:
    """Generate near-miss patterns for a target.

    The number of near-misses is determined by near_miss_ratio:
    - Integer part: guaranteed count
    - Fractional part: probability of one additional near-miss

    Example: ratio=1.5 produces 1 near-miss always, plus 50% chance of second.

    Args:
        target: The target pattern to create near-misses for
        config: NoiseConfig with ratio and other parameters
        generator: Seeded torch.Generator for reproducibility

    Returns:
        List of near-miss EvolvingPatterns
    """
    if config.near_miss_ratio <= 0:
        return []

    # Calculate count: integer part + probabilistic fractional part
    base_count = int(config.near_miss_ratio)
    fractional = config.near_miss_ratio - base_count

    # Add one more with probability = fractional part
    if fractional > 0 and torch.rand(1, generator=generator).item() < fractional:
        base_count += 1

    # Generate near-misses
    return [
        create_near_miss(target, config, generator, i) for i in range(base_count)
    ]


def create_clutter(
    dim: int,
    k: int,
    config: "NoiseConfig",
    generator: torch.Generator,
    target_bits: set[int] | None,
    index: int,
) -> EvolvingPattern:
    """Create a single clutter pattern with varying relatedness.

    Per CONTEXT.md: Clutter isn't purely unrelated - some patterns will be
    closer to targets than others, creating realistic interference.

    Relatedness distribution:
    - 50% distant (0.1 overlap with targets)
    - 30% mid-range (0.2 overlap)
    - 20% closer (0.3 overlap)

    Args:
        dim: Pattern dimensionality
        k: Sparsity (active bits per pattern)
        config: NoiseConfig with coherence range
        generator: Seeded torch.Generator
        target_bits: Combined bits from all targets (for relatedness control)
        index: Index for naming

    Returns:
        EvolvingPattern clutter pattern with exactly k bits
    """
    # Determine relatedness level
    r = torch.rand(1, generator=generator).item()
    if r < 0.5:
        # 50% distant
        overlap_frac = 0.1
    elif r < 0.8:
        # 30% mid-range
        overlap_frac = 0.2
    else:
        # 20% closer
        overlap_frac = 0.3

    # Generate bits with controlled overlap to targets
    if target_bits:
        num_overlap = int(overlap_frac * k)
        target_list = list(target_bits)

        if len(target_list) >= num_overlap:
            perm = torch.randperm(len(target_list), generator=generator)
            overlapping = {target_list[i] for i in perm[:num_overlap].tolist()}
        else:
            overlapping = set(target_list)
            num_overlap = len(overlapping)
    else:
        overlapping: set[int] = set()
        num_overlap = 0

    # Generate remaining bits from non-target positions
    if target_bits:
        available = [i for i in range(dim) if i not in target_bits]
    else:
        available = list(range(dim))

    needed = k - len(overlapping)
    new_perm = torch.randperm(len(available), generator=generator)
    new_bits = {available[i] for i in new_perm[:needed].tolist()}

    all_bits = overlapping | new_bits

    # Random phases
    phases = {
        bit: torch.rand(1, generator=generator).item() * 2 * math.pi
        for bit in all_bits
    }

    # Broader coherence range for clutter
    cmin, cmax = config.clutter_coherence_range
    coherence = cmin + torch.rand(1, generator=generator).item() * (cmax - cmin)

    # Clutter can be older than near-misses
    last_access_tick = int(torch.randint(1, 101, (1,), generator=generator).item())

    return EvolvingPattern(
        dim=dim,
        bits=all_bits,
        original_bits=frozenset(all_bits),
        acquired_bits=set(),
        phases=phases,
        text=f"clutter_{index}",
        coherence=coherence,
        last_access_tick=last_access_tick,
        connection_count=0,
        last_modified_tick=0,
        access_count_since_modification=int(
            torch.randint(0, 10, (1,), generator=generator).item()
        ),
    )


def generate_clutter_batch(
    dim: int,
    k: int,
    count: int,
    config: "NoiseConfig",
    generator: torch.Generator,
    target_bits: set[int] | frozenset[int] | None = None,
) -> list[EvolvingPattern]:
    """Generate multiple clutter patterns.

    Args:
        dim: Pattern dimensionality
        k: Sparsity (active bits per pattern)
        count: Number of clutter patterns to generate
        config: NoiseConfig with coherence range
        generator: Seeded torch.Generator
        target_bits: Optional combined bits from all targets for relatedness control

    Returns:
        List of clutter EvolvingPatterns
    """
    if count <= 0:
        return []

    # Convert frozenset to set if needed
    bits_set = set(target_bits) if target_bits else None

    return [
        create_clutter(dim, k, config, generator, bits_set, i) for i in range(count)
    ]
