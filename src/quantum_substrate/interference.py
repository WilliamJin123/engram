"""Interference-based retrieval for quantum-inspired patterns.

This module provides retrieval mechanisms that leverage phase information
in complex sparse patterns, demonstrating how quantum-inspired interference
can improve similarity detection over classical approaches.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from quantum_substrate.patterns import ComplexSparsePattern


def jaccard_retrieval(
    query: ComplexSparsePattern,
    patterns: Sequence[ComplexSparsePattern],
) -> list[tuple[int, float]]:
    """Retrieve patterns by Jaccard similarity (ignores phase).

    Classical set-based retrieval that only considers which dimensions
    are active, ignoring magnitude and phase information.

    Jaccard(A, B) = |A intersection B| / |A union B|

    Args:
        query: Query pattern.
        patterns: Patterns to search.

    Returns:
        List of (index, similarity) tuples, sorted by similarity descending.
    """
    query_set = set(query.indices)
    results = []

    for i, pattern in enumerate(patterns):
        pattern_set = set(pattern.indices)
        intersection = len(query_set & pattern_set)
        union = len(query_set | pattern_set)

        if union > 0:
            similarity = intersection / union
        else:
            similarity = 0.0

        results.append((i, similarity))

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def interference_retrieval(
    query: ComplexSparsePattern,
    patterns: Sequence[ComplexSparsePattern],
) -> list[tuple[int, float]]:
    """Retrieve patterns using phase-aware interference.

    At overlapping indices, complex amplitudes interfere:
    - Same phase (constructive): amplitudes add, increasing similarity
    - Opposite phase (destructive): amplitudes cancel, decreasing similarity

    Similarity = sum of |query + pattern| at overlapping indices, normalized
    by max possible (if all phases aligned perfectly).

    Args:
        query: Query pattern.
        patterns: Patterns to search.

    Returns:
        List of (index, similarity) tuples, sorted by similarity descending.
    """
    # Build query lookup: index -> (magnitude, phase)
    query_lookup = {
        idx: (mag, phase)
        for idx, mag, phase in zip(query.indices, query.magnitudes, query.phases)
    }

    results = []

    for i, pattern in enumerate(patterns):
        # Find overlapping indices
        pattern_lookup = {
            idx: (mag, phase)
            for idx, mag, phase in zip(pattern.indices, pattern.magnitudes, pattern.phases)
        }

        overlap_indices = set(query_lookup.keys()) & set(pattern_lookup.keys())

        if not overlap_indices:
            results.append((i, 0.0))
            continue

        # Compute interference at each overlapping index
        interference_sum = 0.0
        max_possible = 0.0

        for idx in overlap_indices:
            q_mag, q_phase = query_lookup[idx]
            p_mag, p_phase = pattern_lookup[idx]

            # Complex addition: |a*e^(i*theta1) + b*e^(i*theta2)|
            # = sqrt(a^2 + b^2 + 2*a*b*cos(theta2 - theta1))
            phase_diff = p_phase - q_phase
            combined_mag = math.sqrt(
                q_mag**2 + p_mag**2 + 2 * q_mag * p_mag * math.cos(phase_diff)
            )
            interference_sum += combined_mag

            # Max possible is when phases perfectly align (cos=1)
            max_possible += q_mag + p_mag

        # Normalize to [0, 1]
        if max_possible > 0:
            similarity = interference_sum / max_possible
        else:
            similarity = 0.0

        results.append((i, similarity))

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def create_related_pattern(
    source: ComplexSparsePattern,
    overlap_frac: float = 0.5,
    phase_noise: float = 0.0,
    generator: torch.Generator | None = None,
) -> ComplexSparsePattern:
    """Create a pattern related to source with controlled overlap and phase similarity.

    Args:
        source: Source pattern to create relation from.
        overlap_frac: Fraction of source indices to share (0 to 1).
        phase_noise: Standard deviation of phase perturbation in radians.
            0.0 = identical phases (constructive interference)
            pi = uniformly random perturbation (mixed interference)
        generator: Optional RNG for reproducibility.

    Returns:
        New pattern with specified overlap and phase relationship to source.
    """
    k = source.k
    dim = source.dim

    # Determine how many indices to share
    num_shared = max(1, int(overlap_frac * k))
    num_new = k - num_shared

    # Select which source indices to keep
    source_indices = list(source.indices)
    if generator is not None:
        perm = torch.randperm(len(source_indices), generator=generator)
    else:
        perm = torch.randperm(len(source_indices))
    shared_positions = perm[:num_shared].sort().values.tolist()

    shared_indices = [source_indices[p] for p in shared_positions]
    shared_magnitudes = [source.magnitudes[p] for p in shared_positions]
    shared_phases = [source.phases[p] for p in shared_positions]

    # Add phase noise to shared indices
    if phase_noise > 0:
        if generator is not None:
            noise = torch.randn(num_shared, generator=generator) * phase_noise
        else:
            noise = torch.randn(num_shared) * phase_noise
        shared_phases = [
            (p + n.item()) % (2 * math.pi)
            for p, n in zip(shared_phases, noise)
        ]

    # Generate new indices (not overlapping with source)
    source_set = set(source.indices)
    available = [i for i in range(dim) if i not in source_set]

    if num_new > 0 and available:
        if generator is not None:
            perm = torch.randperm(len(available), generator=generator)
        else:
            perm = torch.randperm(len(available))
        new_indices = [available[perm[j].item()] for j in range(min(num_new, len(available)))]

        # Random magnitudes and phases for new indices
        new_magnitudes = [1.0] * len(new_indices)
        if generator is not None:
            new_phases = (torch.rand(len(new_indices), generator=generator) * 2 * math.pi).tolist()
        else:
            new_phases = (torch.rand(len(new_indices)) * 2 * math.pi).tolist()
    else:
        new_indices = []
        new_magnitudes = []
        new_phases = []

    # Combine and sort
    all_indices = shared_indices + new_indices
    all_magnitudes = shared_magnitudes + new_magnitudes
    all_phases = shared_phases + new_phases

    return ComplexSparsePattern(
        dim=dim,
        indices=tuple(all_indices),
        magnitudes=tuple(all_magnitudes),
        phases=tuple(all_phases),
    )
