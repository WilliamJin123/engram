# src/agentic/text_encoder.py
"""Text-to-pattern encoding for agentic memory.

Converts text into sparse patterns with phase information.
This addresses the fundamental question: "How does text get converted to patterns?"

The encoding is:
1. Deterministic - same text always produces same pattern
2. Lexically aware - shared words create shared bits
3. Order-preserving - word position affects phase

No semantic understanding - the LLM provides semantics through reranking.
The substrate provides structure through interference and binding.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EncodingConfig:
    """Configuration for text encoding."""

    use_ngrams: bool = False
    ngram_sizes: list[int] = field(default_factory=lambda: [2, 3])
    lowercase: bool = True
    remove_punctuation: bool = False


@dataclass
class EncodedPattern:
    """A sparse pattern encoded from text.

    Attributes:
        dim: Total dimensionality of the pattern space.
        bits: Active bit indices.
        phases: Phase (0 to 2*pi) at each active bit.
        text: Original source text.
        metadata: Arbitrary metadata.
    """

    dim: int
    bits: set[int]
    phases: dict[int, float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def jaccard_similarity(self, other: "EncodedPattern") -> float:
        """Compute Jaccard similarity with another pattern."""
        intersection = len(self.bits & other.bits)
        union = len(self.bits | other.bits)
        return intersection / union if union > 0 else 0.0


class TextEncoder:
    """Encodes text to sparse patterns with phase information.

    Uses deterministic hashing to convert text tokens to bit positions.
    Word order is encoded in phase values.

    Attributes:
        dim: Dimensionality of pattern space.
        k: Number of active bits (sparsity).
        config: Encoding configuration.
    """

    def __init__(
        self,
        dim: int = 1024,
        k: int = 50,
        config: EncodingConfig | None = None,
    ):
        self.dim = dim
        self.k = k
        self.config = config or EncodingConfig()

    def encode(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> EncodedPattern:
        """Encode text to a sparse pattern.

        Args:
            text: Text to encode.
            metadata: Optional metadata to attach.

        Returns:
            EncodedPattern with k active bits.
        """
        tokens = self._tokenize(text)

        if not tokens:
            return EncodedPattern(
                dim=self.dim,
                bits=set(),
                phases={},
                text=text,
                metadata=metadata or {},
            )

        bit_phases: dict[int, list[float]] = {}

        for position, token in enumerate(tokens):
            token_bits = self._token_to_bits(token)
            token_phase = self._position_to_phase(position, len(tokens))

            for bit in token_bits:
                if bit not in bit_phases:
                    bit_phases[bit] = []
                bit_phases[bit].append(token_phase)

        if self.config.use_ngrams:
            ngrams = self._extract_ngrams(text)
            for ngram in ngrams:
                ngram_bits = self._token_to_bits(ngram)
                for bit in ngram_bits:
                    if bit not in bit_phases:
                        bit_phases[bit] = []
                    bit_phases[bit].append(0.0)

        sorted_bits = sorted(
            bit_phases.keys(),
            key=lambda b: (-len(bit_phases[b]), b)
        )[:self.k]

        bits = set(sorted_bits)
        phases = {}
        for bit in bits:
            phases[bit] = self._circular_mean(bit_phases[bit])

        # Pad with additional bits if we don't have enough
        if len(bits) < self.k:
            bits, phases = self._pad_to_k(text, bits, phases)

        return EncodedPattern(
            dim=self.dim,
            bits=bits,
            phases=phases,
            text=text,
            metadata=metadata or {},
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)

        return text.split()

    def _extract_ngrams(self, text: str) -> list[str]:
        """Extract character n-grams from text."""
        if self.config.lowercase:
            text = text.lower()

        ngrams = []
        for n in self.config.ngram_sizes:
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])

        return ngrams

    def _token_to_bits(self, token: str) -> set[int]:
        """Convert a token to a set of bit indices."""
        base_hash = hashlib.sha256(token.encode()).hexdigest()
        bits_per_token = max(1, self.k // 10)

        bits = set()
        for i in range(bits_per_token):
            h = hashlib.sha256(f"{base_hash}:{i}".encode()).hexdigest()
            bit_idx = int(h[:8], 16) % self.dim
            bits.add(bit_idx)

        return bits

    def _position_to_phase(self, position: int, total: int) -> float:
        """Convert position to phase value."""
        if total <= 1:
            return 0.0
        return (2 * math.pi * position) / total

    def _circular_mean(self, phases: list[float]) -> float:
        """Compute circular mean of phases."""
        if not phases:
            return 0.0

        sin_sum = sum(math.sin(p) for p in phases)
        cos_sum = sum(math.cos(p) for p in phases)

        mean_phase = math.atan2(sin_sum, cos_sum)

        if mean_phase < 0:
            mean_phase += 2 * math.pi

        return mean_phase

    def _pad_to_k(
        self,
        text: str,
        bits: set[int],
        phases: dict[int, float],
    ) -> tuple[set[int], dict[int, float]]:
        """Pad bits to exactly k using text-derived hash bits.

        When there aren't enough unique bits from tokens, we generate
        additional deterministic bits from a hash based on sorted tokens.
        This ensures texts with the same words (in any order) get the same
        padding bits, preserving the property that word order only affects phase.
        """
        # Use sorted tokens for padding hash to ensure same words = same bits
        tokens = self._tokenize(text)
        sorted_tokens = " ".join(sorted(set(tokens)))
        text_hash = hashlib.sha256(sorted_tokens.encode()).hexdigest()
        pad_idx = 0

        while len(bits) < self.k:
            h = hashlib.sha256(f"{text_hash}:pad:{pad_idx}".encode()).hexdigest()
            bit_idx = int(h[:8], 16) % self.dim

            if bit_idx not in bits:
                bits.add(bit_idx)
                # Use a phase derived from the hash for padding bits
                phase_hash = int(h[8:16], 16)
                phases[bit_idx] = (phase_hash % 1000) / 1000 * 2 * math.pi

            pad_idx += 1

        return bits, phases
