"""Holographic Reduced Representation (HRR) binding operations.

HRR uses circular convolution to bind patterns and correlation to unbind.
This enables role-filler binding: we can store WHO did WHAT and query by role.

Reference: Plate, T. (1995). Holographic Reduced Representations.
"""

import torch


def bind_hrr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bind two patterns via circular convolution.

    Binding creates a new pattern that represents the association of a and b.
    The result is the same dimensionality as the inputs.

    Implementation: convolution in time domain = multiplication in frequency domain.

    Args:
        a: First pattern (complex tensor).
        b: Second pattern (complex tensor).

    Returns:
        Bound pattern (complex tensor, same shape).
    """
    # FFT-based circular convolution: IFFT(FFT(a) * FFT(b))
    return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b))


def unbind_hrr(bound: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Unbind pattern b from a bound pattern.

    Given bound = bind(a, b), unbind(bound, b) approximately recovers a.

    Implementation: correlation = convolution with conjugate.

    Args:
        bound: A bound pattern.
        b: The pattern to unbind (e.g., a role).

    Returns:
        Recovered pattern (approximately the other component).
    """
    # Correlation: IFFT(FFT(bound) * conj(FFT(b)))
    return torch.fft.ifft(torch.fft.fft(bound) * torch.conj(torch.fft.fft(b)))


def similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute normalized similarity between complex patterns.

    Uses cosine similarity on the real parts after normalization.

    Args:
        a: First pattern.
        b: Second pattern.

    Returns:
        Similarity in range [-1, 1], where 1 = identical.
    """
    # Normalize
    a_norm = a / (a.abs().mean() + 1e-8)
    b_norm = b / (b.abs().mean() + 1e-8)

    # Cosine similarity on real parts
    a_real = a_norm.real
    b_real = b_norm.real

    dot = (a_real * b_real).sum()
    norm_a = (a_real * a_real).sum().sqrt()
    norm_b = (b_real * b_real).sum().sqrt()

    return (dot / (norm_a * norm_b + 1e-8)).item()
