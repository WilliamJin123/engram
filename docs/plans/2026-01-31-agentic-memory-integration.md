# Agentic Memory with LLM Integration - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address the three remaining uncertainties from Wave 2 testing:
1. **LLM integration** - How does text get converted to patterns?
2. **Real-world encoding** - Will LLM outputs produce useful patterns?
3. **Very long sequences** - Phase encoding at 1000+ items?

**Secondary goal:** Build and validate coactivation-based associative learning where semantic similarity emerges from LLM retrieval patterns over time.

**Architecture:** Text is encoded via deterministic hashing to sparse patterns. When LLM reranking groups patterns together, they undergo coactivation (share bits). Over simulated interactions, we measure whether retrieval precision improves and semantic clusters emerge - without any pre-baked semantic structure.

**Tech Stack:** PyTorch, pytest, dataclasses

**LLM Integration:** See `docs/KEYCYCLE.md` for information on how to integrate LLMs into this project. Do NOT use the Anthropic API directly.

---

## Project Structure

```
src/
├── quantum_substrate/          # Core substrate (math/physics layer) - EXISTING
│   ├── __init__.py
│   ├── patterns.py
│   ├── binding.py
│   └── interference.py
│
└── agentic/                    # Agentic memory layer - NEW
    ├── __init__.py
    ├── text_encoder.py         # Text → sparse patterns
    ├── evolving_pattern.py     # Patterns that learn over time
    ├── memory_store.py         # Storage & retrieval
    ├── coactivation.py         # Learning rule
    ├── llm_interface.py        # LLM reranking interface
    ├── agent_memory.py         # High-level API
    └── evaluation.py           # Metrics & evaluation

tests/
├── quantum_substrate/          # Existing tests (unchanged)
│   └── ...
│
└── agentic/                    # New tests
    ├── __init__.py
    ├── test_text_encoder.py
    ├── test_evolving_pattern.py
    ├── test_realistic_encoding.py
    ├── test_long_sequences.py
    ├── test_memory_store.py
    ├── test_coactivation.py
    ├── test_llm_interface.py
    ├── test_agent_memory.py
    └── test_evaluation.py
```

---

## Phase 1: Text-to-Pattern Encoding (Addresses Uncertainty #1)

### Task 1.1: Create TextEncoder class with deterministic hashing

**Rationale:** Directly addresses uncertainty #1 - "How does text get converted to patterns?"

**Files:**
- Create: `src/agentic/text_encoder.py`
- Test: `tests/agentic/test_text_encoder.py`

**Step 1: Write the failing test**

```python
# tests/agentic/test_text_encoder.py
"""Tests for text-to-pattern encoding."""

import pytest
import math
from agentic.text_encoder import TextEncoder, EncodingConfig


class TestTextEncoderDeterminism:
    """Test that encoding is deterministic and reproducible."""

    def test_same_text_same_pattern(self):
        """Identical text produces identical patterns."""
        encoder = TextEncoder(dim=1024, k=50)

        p1 = encoder.encode("hello world")
        p2 = encoder.encode("hello world")

        assert p1.bits == p2.bits
        assert p1.phases == p2.phases

    def test_different_text_different_pattern(self):
        """Different text produces different patterns."""
        encoder = TextEncoder(dim=1024, k=50)

        p1 = encoder.encode("hello world")
        p2 = encoder.encode("goodbye world")

        overlap = len(p1.bits & p2.bits)
        assert overlap < len(p1.bits)  # Not identical
        assert overlap > 0  # Some overlap from shared word

    def test_sparsity_maintained(self):
        """Output pattern has exactly k active bits."""
        encoder = TextEncoder(dim=1024, k=50)

        p = encoder.encode("test sentence with multiple words")

        assert len(p.bits) == 50


class TestTextEncoderLexicalOverlap:
    """Test that lexical similarity produces pattern overlap."""

    def test_shared_words_increase_overlap(self):
        """More shared words = higher Jaccard similarity."""
        encoder = TextEncoder(dim=1024, k=100)

        p1 = encoder.encode("the quick brown fox")
        p2 = encoder.encode("the quick brown dog")  # 3 shared words
        p3 = encoder.encode("a slow red cat")       # 0 shared words

        overlap_p1_p2 = len(p1.bits & p2.bits) / len(p1.bits | p2.bits)
        overlap_p1_p3 = len(p1.bits & p3.bits) / len(p1.bits | p3.bits)

        assert overlap_p1_p2 > overlap_p1_p3

    def test_word_order_affects_phase(self):
        """Same words in different order have different phases."""
        encoder = TextEncoder(dim=1024, k=50)

        p1 = encoder.encode("dog chased cat")
        p2 = encoder.encode("cat chased dog")

        # Same bits (same words)
        assert p1.bits == p2.bits

        # Different phases (different order)
        shared_bits = list(p1.bits & p2.bits)
        phase_diffs = [abs(p1.phases[b] - p2.phases[b]) for b in shared_bits[:10]]
        assert any(d > 0.1 for d in phase_diffs), "Phases should differ for different word order"


class TestTextEncoderPhaseEncoding:
    """Test positional phase encoding for word order."""

    def test_phases_in_valid_range(self):
        """All phases are between 0 and 2*pi."""
        encoder = TextEncoder(dim=1024, k=50)

        p = encoder.encode("some test text")

        for bit, phase in p.phases.items():
            assert 0 <= phase < 2 * math.pi

    def test_sequential_words_have_sequential_phases(self):
        """Words earlier in text have earlier phases."""
        encoder = TextEncoder(dim=1024, k=50)

        p_first = encoder.encode("first")
        p_second = encoder.encode("second")
        p_both = encoder.encode("first second")

        # Phases for "first" bits should be earlier than "second" bits


class TestEncodingConfigOptions:
    """Test configurable encoding options."""

    def test_custom_dimension(self):
        """Can create encoder with custom dimension."""
        encoder = TextEncoder(dim=2048, k=100)
        p = encoder.encode("test")

        assert all(b < 2048 for b in p.bits)

    def test_custom_sparsity(self):
        """Can create encoder with custom sparsity."""
        encoder = TextEncoder(dim=1024, k=25)
        p = encoder.encode("test")

        assert len(p.bits) == 25

    def test_ngram_encoding(self):
        """Can encode with n-grams for sub-word features."""
        config = EncodingConfig(use_ngrams=True, ngram_sizes=[2, 3])
        encoder = TextEncoder(dim=1024, k=50, config=config)

        p1 = encoder.encode("testing")
        p2 = encoder.encode("tester")  # Shares "test" prefix

        overlap = len(p1.bits & p2.bits) / len(p1.bits | p2.bits)

        assert overlap > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agentic/test_text_encoder.py -v`
Expected: FAIL with "No module named 'agentic'"

**Step 3: Create directory structure**

```bash
mkdir -p src/agentic tests/agentic
```

Create `src/agentic/__init__.py`:
```python
"""Agentic memory layer built on the quantum substrate."""
```

Create `tests/agentic/__init__.py`:
```python
"""Tests for the agentic memory layer."""
```

**Step 4: Write minimal implementation**

```python
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
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/agentic/test_text_encoder.py -v`
Expected: PASS (10 tests)

**Step 6: Commit**

```bash
git add src/agentic/ tests/agentic/
git commit -m "feat(agentic): add TextEncoder for text-to-pattern conversion"
```

---

### Task 1.2: Create EvolvingPattern dataclass with lineage tracking

**Files:**
- Create: `src/agentic/evolving_pattern.py`
- Test: `tests/agentic/test_evolving_pattern.py`

**Step 1: Write the failing test**

```python
# tests/agentic/test_evolving_pattern.py
"""Tests for EvolvingPattern with lineage tracking."""

import pytest
from agentic.evolving_pattern import EvolvingPattern
from agentic.text_encoder import TextEncoder


class TestEvolvingPatternCreation:
    """Test basic pattern creation and properties."""

    def test_create_from_encoded_pattern(self):
        """EvolvingPattern can wrap an EncodedPattern."""
        encoder = TextEncoder(dim=1024, k=50)
        encoded = encoder.encode("hello world")

        pattern = EvolvingPattern.from_encoded(encoded)

        assert pattern.dim == 1024
        assert len(pattern.bits) == 50
        assert pattern.original_bits == frozenset(pattern.bits)
        assert len(pattern.acquired_bits) == 0
        assert pattern.text == "hello world"

    def test_create_from_text(self):
        """Pattern can be created directly from text."""
        pattern = EvolvingPattern.from_text("hello world", dim=1024, k=50)

        assert pattern.dim == 1024
        assert len(pattern.bits) == 50
        assert pattern.original_bits == frozenset(pattern.bits)
        assert len(pattern.acquired_bits) == 0
        assert pattern.text == "hello world"

    def test_deterministic_encoding(self):
        """Same text produces same pattern."""
        p1 = EvolvingPattern.from_text("hello world", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("hello world", dim=1024, k=50)

        assert p1.bits == p2.bits
        assert p1.original_bits == p2.original_bits

    def test_phases_preserved(self):
        """Each bit has an associated phase."""
        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)

        assert len(pattern.phases) == len(pattern.bits)
        for bit in pattern.bits:
            assert bit in pattern.phases


class TestEvolvingPatternDenseConversion:
    """Test conversion to dense tensors for HRR operations."""

    def test_to_dense_uses_original_bits(self):
        """to_dense() uses only original bits for binding integrity."""
        import torch

        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)

        pattern.bits.add(999)
        pattern.acquired_bits.add(999)
        pattern.phases[999] = 0.0

        dense = pattern.to_dense()

        assert dense[999].abs().item() < 1e-6

        for bit in pattern.original_bits:
            assert dense[bit].abs().item() > 0.5

    def test_to_dense_evolved_uses_all_bits(self):
        """to_dense_evolved() uses all bits for retrieval."""
        import torch

        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)

        pattern.bits.add(999)
        pattern.acquired_bits.add(999)
        pattern.phases[999] = 0.0

        dense = pattern.to_dense_evolved()

        assert dense[999].abs().item() > 0.5


class TestEvolvingPatternMetrics:
    """Test metrics for tracking pattern evolution."""

    def test_obesity_metric(self):
        """Obesity is ratio of current to original bits."""
        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)

        assert pattern.obesity == 1.0

        for i in range(10):
            pattern.bits.add(900 + i)
            pattern.acquired_bits.add(900 + i)

        assert pattern.obesity == 60 / 50  # 1.2

    def test_acquired_ratio_metric(self):
        """Acquired ratio is fraction of bits that are acquired."""
        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)

        assert pattern.acquired_ratio == 0.0

        for i in range(10):
            pattern.bits.add(900 + i)
            pattern.acquired_bits.add(900 + i)

        assert pattern.acquired_ratio == 10 / 60
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agentic/test_evolving_pattern.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/agentic/evolving_pattern.py
"""Evolving pattern with coactivation learning support.

Patterns track their original bits separately from acquired bits,
enabling coactivation-based learning while preserving binding integrity.

Key insight: For HRR binding operations, we use ORIGINAL bits only.
For retrieval, we use ALL bits (original + acquired).
This prevents transitive pollution while enabling learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.text_encoder import EncodedPattern


@dataclass
class EvolvingPattern:
    """A sparse pattern that can evolve through coactivation learning.

    Attributes:
        dim: Total dimensionality of the pattern space.
        bits: Current active bit indices (original + acquired).
        original_bits: Frozen bits from initial encoding (used for binding).
        acquired_bits: Bits gained through coactivation.
        phases: Phase (0 to 2pi) at each active bit.
        text: Source text that was encoded.
        metadata: Arbitrary metadata (speaker, time, topic, etc).
        acquisition_count: Number of coactivation events.
    """

    dim: int
    bits: set[int]
    original_bits: frozenset[int]
    acquired_bits: set[int]
    phases: dict[int, float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    acquisition_count: int = 0

    @classmethod
    def from_encoded(cls, encoded: "EncodedPattern") -> EvolvingPattern:
        """Create from an EncodedPattern."""
        return cls(
            dim=encoded.dim,
            bits=set(encoded.bits),
            original_bits=frozenset(encoded.bits),
            acquired_bits=set(),
            phases=dict(encoded.phases),
            text=encoded.text,
            metadata=dict(encoded.metadata),
            acquisition_count=0,
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        dim: int = 1024,
        k: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> EvolvingPattern:
        """Create a pattern directly from text."""
        from agentic.text_encoder import TextEncoder

        encoder = TextEncoder(dim=dim, k=k)
        encoded = encoder.encode(text, metadata=metadata)
        return cls.from_encoded(encoded)

    def to_dense(self) -> "torch.Tensor":
        """Convert to dense complex tensor for HRR operations.

        Uses ORIGINAL bits only to preserve binding integrity.
        """
        import torch

        dense = torch.zeros(self.dim, dtype=torch.complex64)
        for bit in self.original_bits:
            phase = self.phases.get(bit, 0.0)
            dense[bit] = torch.exp(torch.tensor(1j * phase))
        return dense

    def to_dense_evolved(self) -> "torch.Tensor":
        """Convert to dense complex tensor using ALL bits.

        Includes both original and acquired bits.
        Use this for retrieval (similarity search).
        """
        import torch

        dense = torch.zeros(self.dim, dtype=torch.complex64)
        for bit in self.bits:
            phase = self.phases.get(bit, 0.0)
            dense[bit] = torch.exp(torch.tensor(1j * phase))
        return dense

    @property
    def obesity(self) -> float:
        """Ratio of current bits to original bits."""
        if not self.original_bits:
            return 0.0
        return len(self.bits) / len(self.original_bits)

    @property
    def acquired_ratio(self) -> float:
        """Fraction of current bits that were acquired."""
        if not self.bits:
            return 0.0
        return len(self.acquired_bits) / len(self.bits)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/agentic/test_evolving_pattern.py -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add src/agentic/evolving_pattern.py tests/agentic/test_evolving_pattern.py
git commit -m "feat(agentic): add EvolvingPattern with lineage tracking"
```

---

## Phase 2: Real-World Encoding Validation (Addresses Uncertainty #2)

### Task 2.1: Create realistic encoding benchmark tests

**Rationale:** Directly addresses uncertainty #2 - "Will LLM outputs produce useful patterns?"

**Files:**
- Create: `tests/agentic/test_realistic_encoding.py`

**Step 1: Write the test**

```python
# tests/agentic/test_realistic_encoding.py
"""Tests for realistic text encoding scenarios.

Addresses uncertainty #2: "Will LLM outputs produce useful patterns?"
"""

import pytest
from agentic.text_encoder import TextEncoder
from agentic.evolving_pattern import EvolvingPattern


ANIMAL_CONVERSATIONS = [
    "Alice said dogs are very loyal pets that love their owners",
    "Bob mentioned that wolves hunt in packs at night",
    "Carol noted cats are independent and self-sufficient",
    "Alice talked about how her dog always greets her at the door",
    "Bob observed that many dog breeds were domesticated from wolves",
]

TECH_CONVERSATIONS = [
    "Alice explained that machine learning requires lots of data",
    "Bob said neural networks can recognize images now",
    "Carol mentioned that GPT models can write convincing text",
    "Alice noted that training AI is computationally expensive",
    "Bob talked about how transformers changed NLP completely",
]

FOOD_CONVERSATIONS = [
    "Alice said Italian pizza is her favorite food",
    "Bob mentioned that sushi requires fresh fish",
    "Carol talked about baking sourdough bread at home",
    "Alice noted that cooking pasta needs salted water",
    "Bob said Thai food is often quite spicy",
]


class TestSemanticGrouping:
    """Test that semantically related texts have pattern overlap."""

    def test_within_topic_similarity_higher_than_cross_topic(self):
        """Texts about same topic have higher overlap than different topics."""
        encoder = TextEncoder(dim=1024, k=100)

        animal_patterns = [encoder.encode(t) for t in ANIMAL_CONVERSATIONS]
        tech_patterns = [encoder.encode(t) for t in TECH_CONVERSATIONS]
        food_patterns = [encoder.encode(t) for t in FOOD_CONVERSATIONS]

        def avg_similarity(patterns):
            sims = []
            for i, p1 in enumerate(patterns):
                for p2 in patterns[i+1:]:
                    sims.append(p1.jaccard_similarity(p2))
            return sum(sims) / len(sims) if sims else 0

        within_animal = avg_similarity(animal_patterns)
        within_tech = avg_similarity(tech_patterns)
        within_food = avg_similarity(food_patterns)

        cross_sims = []
        for ap in animal_patterns:
            for tp in tech_patterns:
                cross_sims.append(ap.jaccard_similarity(tp))
        cross_animal_tech = sum(cross_sims) / len(cross_sims)

        avg_within = (within_animal + within_tech + within_food) / 3

        print(f"Within-topic avg: {avg_within:.4f}")
        print(f"Cross-topic (animal-tech): {cross_animal_tech:.4f}")

        assert all(len(p.bits) == 100 for p in animal_patterns)


class TestSpeakerPatterns:
    """Test that speaker-tagged text preserves speaker information."""

    def test_alice_patterns_have_common_bits(self):
        """All Alice's utterances share some bits from her name."""
        encoder = TextEncoder(dim=1024, k=100)

        alice_texts = [t for t in ANIMAL_CONVERSATIONS + TECH_CONVERSATIONS + FOOD_CONVERSATIONS
                       if "Alice" in t]

        alice_patterns = [encoder.encode(t) for t in alice_texts]

        if alice_patterns:
            common_bits = alice_patterns[0].bits.copy()
            for p in alice_patterns[1:]:
                common_bits &= p.bits

            assert len(common_bits) > 0, "Alice patterns should share some bits"


class TestLongFormContent:
    """Test encoding of longer, more realistic content."""

    def test_paragraph_encoding(self):
        """Can encode paragraph-length content."""
        encoder = TextEncoder(dim=1024, k=100)

        paragraph = """
        The quick brown fox jumps over the lazy dog. This sentence contains
        every letter of the alphabet and is commonly used for typography testing.
        It's been a standard test since at least the late 19th century when it
        was used to test typewriters and teletype machines.
        """

        pattern = encoder.encode(paragraph)

        assert len(pattern.bits) == 100
        assert pattern.text == paragraph

    def test_multi_sentence_preserves_order(self):
        """Multi-sentence text preserves sentence order in phase."""
        encoder = TextEncoder(dim=1024, k=100)

        forward = "First sentence. Second sentence. Third sentence."
        backward = "Third sentence. Second sentence. First sentence."

        p_forward = encoder.encode(forward)
        p_backward = encoder.encode(backward)

        assert p_forward.bits == p_backward.bits or len(p_forward.bits & p_backward.bits) > 50


class TestEdgeCases:
    """Test edge cases in realistic encoding."""

    def test_empty_string(self):
        """Empty string produces empty pattern."""
        encoder = TextEncoder(dim=1024, k=50)
        p = encoder.encode("")
        assert len(p.bits) == 0

    def test_single_word(self):
        """Single word produces valid pattern."""
        encoder = TextEncoder(dim=1024, k=50)
        p = encoder.encode("hello")
        assert len(p.bits) > 0
        assert len(p.bits) <= 50

    def test_repeated_word(self):
        """Repeated word doesn't break encoding."""
        encoder = TextEncoder(dim=1024, k=50)
        p = encoder.encode("hello hello hello hello hello")
        assert len(p.bits) == 50

    def test_special_characters(self):
        """Special characters don't break encoding."""
        encoder = TextEncoder(dim=1024, k=50)
        p = encoder.encode("Hello! How are you? I'm fine, thanks.")
        assert len(p.bits) == 50

    def test_unicode(self):
        """Unicode text encodes correctly."""
        encoder = TextEncoder(dim=1024, k=50)
        p = encoder.encode("Hello world")
        assert len(p.bits) == 50
```

**Step 2: Run test**

Run: `pytest tests/agentic/test_realistic_encoding.py -v`

**Step 3: Commit**

```bash
git add tests/agentic/test_realistic_encoding.py
git commit -m "test(agentic): add realistic encoding benchmarks"
```

---

## Phase 3: Very Long Sequence Testing (Addresses Uncertainty #3)

### Task 3.1: Create long sequence phase encoding tests

**Rationale:** Directly addresses uncertainty #3 - "Phase encoding at 1000+ items?"

**Files:**
- Create: `tests/agentic/test_long_sequences.py`

**Step 1: Write the test**

```python
# tests/agentic/test_long_sequences.py
"""Tests for very long sequence phase encoding.

Addresses uncertainty #3: "Phase encoding at 1000+ items?"
"""

import pytest
import math
import torch
from agentic.evolving_pattern import EvolvingPattern


def create_sequence_patterns(n: int, dim: int = 1024, k: int = 50) -> list[EvolvingPattern]:
    """Create n patterns with sequential phases."""
    patterns = []
    for i in range(n):
        p = EvolvingPattern.from_text(f"item_{i}", dim=dim, k=k)

        seq_phase = (2 * math.pi * i) / n
        for bit in p.bits:
            p.phases[bit] = seq_phase

        patterns.append(p)

    return patterns


def measure_sequence_accuracy(patterns: list[EvolvingPattern]) -> float:
    """Measure how accurately we can recover sequence order from phase."""
    n = len(patterns)
    correct = 0

    for i in range(n - 1):
        current_phase = list(patterns[i].phases.values())[0]

        expected_delta = (2 * math.pi) / n
        expected_next = (current_phase + expected_delta) % (2 * math.pi)

        min_diff = float('inf')
        closest_idx = -1

        for j in range(n):
            if j == i:
                continue
            other_phase = list(patterns[j].phases.values())[0]

            diff = abs(expected_next - other_phase)
            diff = min(diff, 2 * math.pi - diff)

            if diff < min_diff:
                min_diff = diff
                closest_idx = j

        if closest_idx == i + 1:
            correct += 1

    return correct / (n - 1) if n > 1 else 1.0


class TestSequenceLengthLimits:
    """Test phase encoding at various sequence lengths."""

    @pytest.mark.parametrize("length", [10, 50, 100, 200, 500])
    def test_medium_sequences(self, length):
        """Test medium-length sequences."""
        patterns = create_sequence_patterns(length)
        accuracy = measure_sequence_accuracy(patterns)

        assert accuracy >= 0.95, f"Accuracy {accuracy:.2%} too low at length {length}"

    @pytest.mark.parametrize("length", [1000, 2000, 5000])
    def test_long_sequences(self, length):
        """Test long sequences (the uncertainty region)."""
        patterns = create_sequence_patterns(length)
        accuracy = measure_sequence_accuracy(patterns)

        print(f"Sequence length {length}: accuracy = {accuracy:.4f}")

    @pytest.mark.parametrize("length", [10000, 20000])
    def test_very_long_sequences(self, length):
        """Test very long sequences (expected to show degradation)."""
        patterns = create_sequence_patterns(length)
        accuracy = measure_sequence_accuracy(patterns)

        print(f"Sequence length {length}: accuracy = {accuracy:.4f}")


class TestPhaseResolutionTheory:
    """Theoretical analysis of phase resolution limits."""

    def test_phase_resolution_calculation(self):
        """Calculate theoretical phase resolution at various lengths."""
        for n in [100, 1000, 10000, 100000]:
            resolution_rad = 2 * math.pi / n
            resolution_deg = math.degrees(resolution_rad)

            print(f"n={n:6d}: resolution = {resolution_rad:.6f} rad = {resolution_deg:.4f} degrees")

    def test_noise_threshold_estimate(self):
        """Estimate noise threshold that would cause errors."""
        noise_rad = 0.01
        threshold_n = int(2 * math.pi / (2 * noise_rad))

        print(f"With noise = {noise_rad} rad, expect errors above n = {threshold_n}")


class TestPracticalSequenceScenarios:
    """Test realistic sequence scenarios."""

    def test_conversation_history(self):
        """Simulate encoding a long conversation history."""
        n_turns = 200

        patterns = []
        for i in range(n_turns):
            speaker = ["Alice", "Bob", "Carol"][i % 3]
            text = f"{speaker} said turn number {i} content here"

            p = EvolvingPattern.from_text(text, dim=1024, k=50)

            seq_phase = (2 * math.pi * i) / n_turns
            for bit in p.bits:
                original_phase = p.phases[bit]
                p.phases[bit] = (original_phase + seq_phase) / 2

            patterns.append(p)

        def avg_phase(p):
            return sum(p.phases.values()) / len(p.phases) if p.phases else 0

        phase_order = sorted(range(n_turns), key=lambda i: avg_phase(patterns[i]))

        correct_positions = sum(1 for i, j in enumerate(phase_order) if abs(i - j) <= 5)
        position_accuracy = correct_positions / n_turns

        print(f"Conversation ordering accuracy (within 5): {position_accuracy:.2%}")
```

**Step 2: Run test**

Run: `pytest tests/agentic/test_long_sequences.py -v`

**Step 3: Commit**

```bash
git add tests/agentic/test_long_sequences.py
git commit -m "test(agentic): add long sequence phase encoding tests"
```

---

## Phase 4: Memory Store with LLM-Agnostic Retrieval

### Task 4.1: Create MemoryStore class

**Files:**
- Create: `src/agentic/memory_store.py`
- Test: `tests/agentic/test_memory_store.py`

**Step 1: Write the failing test**

```python
# tests/agentic/test_memory_store.py
"""Tests for MemoryStore with pattern-based retrieval."""

import pytest
from agentic.memory_store import MemoryStore
from agentic.evolving_pattern import EvolvingPattern


class TestMemoryStoreBasics:
    """Test basic store and retrieve operations."""

    def test_store_and_retrieve_by_id(self):
        """Can store pattern and retrieve by ID."""
        store = MemoryStore(dim=1024, k=50)

        pattern_id = store.store("Hello world", metadata={"speaker": "alice"})

        retrieved = store.get(pattern_id)
        assert retrieved is not None
        assert retrieved.text == "Hello world"
        assert retrieved.metadata["speaker"] == "alice"

    def test_retrieve_by_pattern_similarity(self):
        """Can retrieve similar patterns."""
        store = MemoryStore(dim=1024, k=50)

        store.store("dogs are great pets")
        store.store("cats are independent")
        store.store("the weather is nice today")

        results = store.retrieve("dogs are great pets", top_k=3)

        assert len(results) == 3
        assert results[0].text == "dogs are great pets"
        assert results[0].score == 1.0

    def test_jaccard_retrieval(self):
        """Jaccard similarity finds overlapping patterns."""
        store = MemoryStore(dim=1024, k=50)

        store.store("alpha beta gamma")
        store.store("alpha beta delta")
        store.store("epsilon zeta eta")

        results = store.retrieve("alpha beta gamma", top_k=3, method="jaccard")

        texts = [r.text for r in results]
        assert "alpha beta gamma" in texts[:2]

    def test_interference_retrieval(self):
        """Interference retrieval uses phase information."""
        store = MemoryStore(dim=1024, k=50)

        store.store("meeting at noon")
        store.store("lunch at noon")
        store.store("random other thing")

        results = store.retrieve("noon appointment", top_k=3, method="interference")

        assert len(results) == 3


class TestMemoryStoreMetrics:
    """Test metric tracking."""

    def test_tracks_pattern_count(self):
        """Store tracks number of patterns."""
        store = MemoryStore(dim=1024, k=50)

        assert store.pattern_count == 0
        store.store("one")
        assert store.pattern_count == 1
        store.store("two")
        assert store.pattern_count == 2

    def test_get_all_patterns(self):
        """Can retrieve all patterns for analysis."""
        store = MemoryStore(dim=1024, k=50)

        store.store("one")
        store.store("two")
        store.store("three")

        all_patterns = store.get_all_patterns()
        assert len(all_patterns) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agentic/test_memory_store.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/agentic/memory_store.py
"""Memory store with pattern-based retrieval.

LLM integration is handled externally - this module is LLM-agnostic.
See docs/KEYCYCLE.md for LLM integration details.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

import torch

from agentic.evolving_pattern import EvolvingPattern


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    pattern_id: str
    pattern: EvolvingPattern
    score: float

    @property
    def text(self) -> str:
        return self.pattern.text

    @property
    def metadata(self) -> dict[str, Any]:
        return self.pattern.metadata


class MemoryStore:
    """Store and retrieve patterns with similarity search.

    This is the substrate layer - it handles pattern storage and retrieval.
    LLM reranking should be done externally (see docs/KEYCYCLE.md).
    """

    def __init__(self, dim: int = 1024, k: int = 50):
        self.dim = dim
        self.k = k
        self.patterns: dict[str, EvolvingPattern] = {}

    def store(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        pattern_id: str | None = None,
    ) -> str:
        """Store text as a pattern."""
        if pattern_id is None:
            pattern_id = str(uuid.uuid4())

        pattern = EvolvingPattern.from_text(
            text=text,
            dim=self.dim,
            k=self.k,
            metadata=metadata or {},
        )
        pattern.metadata["id"] = pattern_id

        self.patterns[pattern_id] = pattern
        return pattern_id

    def get(self, pattern_id: str) -> EvolvingPattern | None:
        """Get pattern by ID."""
        return self.patterns.get(pattern_id)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        method: Literal["jaccard", "interference"] = "jaccard",
        use_evolved: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve patterns similar to query."""
        query_pattern = EvolvingPattern.from_text(query, dim=self.dim, k=self.k)

        if method == "jaccard":
            return self._retrieve_jaccard(query_pattern, top_k, use_evolved)
        else:
            return self._retrieve_interference(query_pattern, top_k, use_evolved)

    def _retrieve_jaccard(
        self,
        query: EvolvingPattern,
        top_k: int,
        use_evolved: bool,
    ) -> list[RetrievalResult]:
        """Retrieve using Jaccard similarity on bit overlap."""
        query_bits = query.bits if use_evolved else set(query.original_bits)

        results = []
        for pattern_id, pattern in self.patterns.items():
            pattern_bits = pattern.bits if use_evolved else set(pattern.original_bits)

            intersection = len(query_bits & pattern_bits)
            union = len(query_bits | pattern_bits)
            score = intersection / union if union > 0 else 0.0

            results.append(RetrievalResult(
                pattern_id=pattern_id,
                pattern=pattern,
                score=score,
            ))

        results.sort(key=lambda r: -r.score)
        return results[:top_k]

    def _retrieve_interference(
        self,
        query: EvolvingPattern,
        top_k: int,
        use_evolved: bool,
    ) -> list[RetrievalResult]:
        """Retrieve using phase-aware interference."""
        if use_evolved:
            query_dense = query.to_dense_evolved()
        else:
            query_dense = query.to_dense()

        results = []
        for pattern_id, pattern in self.patterns.items():
            if use_evolved:
                pattern_dense = pattern.to_dense_evolved()
            else:
                pattern_dense = pattern.to_dense()

            combined = query_dense + pattern_dense

            query_active = query_dense.abs() > 1e-6
            pattern_active = pattern_dense.abs() > 1e-6
            overlap = query_active & pattern_active

            if overlap.sum() == 0:
                score = 0.0
            else:
                interference = combined[overlap].abs().sum().item()
                max_possible = (query_dense[overlap].abs() + pattern_dense[overlap].abs()).sum().item()
                score = interference / max_possible if max_possible > 0 else 0.0

            results.append(RetrievalResult(
                pattern_id=pattern_id,
                pattern=pattern,
                score=score,
            ))

        results.sort(key=lambda r: -r.score)
        return results[:top_k]

    @property
    def pattern_count(self) -> int:
        """Number of stored patterns."""
        return len(self.patterns)

    def get_all_patterns(self) -> list[EvolvingPattern]:
        """Get all patterns for analysis."""
        return list(self.patterns.values())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/agentic/test_memory_store.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agentic/memory_store.py tests/agentic/test_memory_store.py
git commit -m "feat(agentic): add MemoryStore with Jaccard and interference retrieval"
```

---

## Phase 5: Coactivation Learning Rule

### Task 5.1: Implement coactivation rule

**Files:**
- Create: `src/agentic/coactivation.py`
- Test: `tests/agentic/test_coactivation.py`

**Step 1: Write the failing test**

```python
# tests/agentic/test_coactivation.py
"""Tests for coactivation-based associative learning."""

import pytest
from agentic.evolving_pattern import EvolvingPattern
from agentic.coactivation import coactivate, CoactivationConfig


class TestCoactivationBasics:
    """Test basic coactivation mechanics."""

    def test_coactivation_shares_bits(self):
        """Co-activated patterns should share some bits."""
        p1 = EvolvingPattern.from_text("dogs are animals", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("wolves are animals", dim=1024, k=50)

        initial_overlap = len(p1.bits & p2.bits)

        for _ in range(10):
            coactivate([p1, p2], strength=0.05)

        final_overlap = len(p1.bits & p2.bits)

        assert final_overlap > initial_overlap, "Coactivation should increase overlap"
        assert p1.acquisition_count > 0
        assert p2.acquisition_count > 0

    def test_original_bits_unchanged(self):
        """Original bits should never change."""
        p1 = EvolvingPattern.from_text("hello", dim=1024, k=50)
        original = frozenset(p1.original_bits)

        p2 = EvolvingPattern.from_text("world", dim=1024, k=50)

        for _ in range(20):
            coactivate([p1, p2], strength=0.1)

        assert p1.original_bits == original, "Original bits must not change"

    def test_only_original_bits_transferred(self):
        """Only original bits should be transferred, not acquired bits."""
        p1 = EvolvingPattern.from_text("alpha", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("beta", dim=1024, k=50)
        p3 = EvolvingPattern.from_text("gamma", dim=1024, k=50)

        for _ in range(5):
            coactivate([p1, p2], strength=0.1)

        p1_acquired_from_p2 = p1.acquired_bits & p2.original_bits
        assert len(p1_acquired_from_p2) > 0, "p1 should have acquired bits from p2"

        p3_bits_before = set(p3.bits)
        for _ in range(5):
            coactivate([p1, p3], strength=0.1)

        p3_new_bits = p3.bits - p3_bits_before

        from_p1_original = p3_new_bits & p1.original_bits
        from_p1_acquired = p3_new_bits & p1.acquired_bits

        assert len(from_p1_original) >= len(from_p1_acquired), \
            "Transferred bits should come from original, not acquired"

    def test_max_bits_respected(self):
        """Patterns should not exceed max bit budget."""
        config = CoactivationConfig(max_bits=60, strength=0.2)

        p1 = EvolvingPattern.from_text("one", dim=1024, k=50)
        p2 = EvolvingPattern.from_text("two", dim=1024, k=50)

        for _ in range(100):
            coactivate([p1, p2], config=config)

        assert len(p1.bits) <= 60, f"Pattern exceeded max_bits: {len(p1.bits)}"
        assert len(p2.bits) <= 60, f"Pattern exceeded max_bits: {len(p2.bits)}"

    def test_obesity_decay(self):
        """Large patterns should learn slower."""
        config = CoactivationConfig(obesity_decay=True, max_bits=100)

        p_large = EvolvingPattern.from_text("large pattern", dim=1024, k=50)
        p_small = EvolvingPattern.from_text("small pattern", dim=1024, k=50)
        p_donor = EvolvingPattern.from_text("donor", dim=1024, k=50)

        for i in range(30):
            p_large.bits.add(800 + i)
            p_large.acquired_bits.add(800 + i)
            p_large.phases[800 + i] = 0.0

        large_before = len(p_large.bits)
        small_before = len(p_small.bits)

        for _ in range(10):
            coactivate([p_large, p_donor], config=config)
            coactivate([p_small, p_donor], config=config)

        large_gained = len(p_large.bits) - large_before
        small_gained = len(p_small.bits) - small_before

        assert small_gained >= large_gained, \
            f"Small pattern should gain more: small={small_gained}, large={large_gained}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agentic/test_coactivation.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/agentic/coactivation.py
"""Coactivation-based associative learning for evolving patterns.

When patterns are retrieved together (by LLM judgment or substrate retrieval),
they undergo coactivation which causes them to share bits over time.

This creates emergent semantic similarity without pre-baked structure:
- Patterns frequently retrieved together become more similar
- The similarity is in the substrate, not in LLM embeddings
- Retrieval improves over time as patterns cluster

Key design: Only ORIGINAL bits are transferred, not acquired bits.
This prevents transitive pollution (A learns from B, C learns from A's
borrowed bits, leading to everything becoming similar).

LLM integration: The LLM decides WHICH patterns to coactivate (via reranking).
This module provides the learning rule. See docs/KEYCYCLE.md for LLM details.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from agentic.evolving_pattern import EvolvingPattern


@dataclass
class CoactivationConfig:
    """Configuration for coactivation learning."""

    strength: float = 0.01  # Base probability of bit transfer
    max_bits: int = 200  # Never exceed this many bits
    obesity_decay: bool = True  # Large patterns learn slower
    bidirectional: bool = True  # Transfer in both directions


def coactivate(
    patterns: Sequence[EvolvingPattern],
    strength: float | None = None,
    config: CoactivationConfig | None = None,
) -> None:
    """Apply coactivation to a group of patterns.

    Patterns that are retrieved together share bits.
    Only ORIGINAL bits are transferred to prevent transitive pollution.

    Args:
        patterns: Patterns to coactivate (typically top-k from retrieval).
        strength: Learning rate (overrides config if provided).
        config: Full configuration (uses defaults if not provided).
    """
    if config is None:
        config = CoactivationConfig()
    if strength is not None:
        config = CoactivationConfig(
            strength=strength,
            max_bits=config.max_bits,
            obesity_decay=config.obesity_decay,
            bidirectional=config.bidirectional,
        )

    for i, p1 in enumerate(patterns):
        for p2 in patterns[i + 1:]:
            _transfer_bits(p1, p2, config)
            if config.bidirectional:
                _transfer_bits(p2, p1, config)


def _transfer_bits(
    source: EvolvingPattern,
    target: EvolvingPattern,
    config: CoactivationConfig,
) -> None:
    """Transfer bits from source to target.

    Only transfers ORIGINAL bits from source (not acquired bits).
    Respects max_bits budget and obesity decay.
    """
    transferable = source.original_bits - target.bits

    if not transferable:
        return

    effective_strength = config.strength
    if config.obesity_decay:
        obesity = target.obesity
        effective_strength = config.strength / max(1.0, obesity)

    n_transfer = max(1, int(len(transferable) * effective_strength))

    available_slots = config.max_bits - len(target.bits)
    if available_slots <= 0:
        return
    n_transfer = min(n_transfer, available_slots)

    bits_to_add = set(random.sample(list(transferable), min(n_transfer, len(transferable))))

    for bit in bits_to_add:
        target.bits.add(bit)
        target.acquired_bits.add(bit)
        if bit in source.phases:
            target.phases[bit] = source.phases[bit]

    target.acquisition_count += 1
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/agentic/test_coactivation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agentic/coactivation.py tests/agentic/test_coactivation.py
git commit -m "feat(agentic): add coactivation-based associative learning"
```

---

## Phase 6: LLM Reranking Interface (Mockable)

### Task 6.1: Create LLM-agnostic reranking interface

**Rationale:** Provides hooks for LLM integration without hardcoding API calls. Real LLM integration should follow docs/KEYCYCLE.md.

**Files:**
- Create: `src/agentic/llm_interface.py`
- Test: `tests/agentic/test_llm_interface.py`

**Step 1: Write the failing test**

```python
# tests/agentic/test_llm_interface.py
"""Tests for LLM reranking interface.

The interface is LLM-agnostic. Real LLM integration should use
the approach described in docs/KEYCYCLE.md.
"""

import pytest
from agentic.llm_interface import BaseLLMReranker, MockLLMReranker
from agentic.memory_store import MemoryStore, RetrievalResult
from agentic.evolving_pattern import EvolvingPattern


class TestMockLLMReranker:
    """Test mock LLM for offline testing."""

    def test_mock_reranker_returns_same_order_for_exact_match(self):
        """Mock reranker keeps exact matches at top."""
        reranker = MockLLMReranker()
        store = MemoryStore(dim=1024, k=50)

        store.store("dogs are great pets")
        store.store("cats are independent")

        candidates = store.retrieve("dogs are great pets", top_k=2)
        reranked = reranker.rerank("dogs are great pets", candidates)

        assert reranked[0].text == "dogs are great pets"

    def test_mock_reranker_uses_keyword_overlap(self):
        """Mock reranker uses simple keyword matching."""
        reranker = MockLLMReranker()
        store = MemoryStore(dim=1024, k=50)

        store.store("dogs bark loudly")
        store.store("cats meow quietly")
        store.store("birds sing songs")

        candidates = store.retrieve("dogs", top_k=3, method="jaccard")
        reranked = reranker.rerank("tell me about dogs", candidates)

        assert "dogs" in reranked[0].text


class TestLLMRerankerInterface:
    """Test the LLM reranker interface structure."""

    def test_reranker_accepts_candidates_and_query(self):
        """Reranker interface takes query and candidates."""
        reranker = MockLLMReranker()

        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)
        candidates = [
            RetrievalResult(pattern_id="1", pattern=pattern, score=0.5),
            RetrievalResult(pattern_id="2", pattern=pattern, score=0.3),
        ]

        result = reranker.rerank("query", candidates)

        assert isinstance(result, list)
        assert all(isinstance(r, RetrievalResult) for r in result)

    def test_custom_reranker_can_be_implemented(self):
        """Custom rerankers can implement the interface."""

        class ReverseReranker(BaseLLMReranker):
            """A reranker that reverses the order (for testing)."""

            def rerank(self, query, candidates, top_k=None):
                result = list(reversed(candidates))
                if top_k:
                    result = result[:top_k]
                return result

        reranker = ReverseReranker()

        pattern = EvolvingPattern.from_text("test", dim=1024, k=50)
        candidates = [
            RetrievalResult(pattern_id="1", pattern=pattern, score=0.9),
            RetrievalResult(pattern_id="2", pattern=pattern, score=0.5),
            RetrievalResult(pattern_id="3", pattern=pattern, score=0.1),
        ]

        result = reranker.rerank("query", candidates)

        assert result[0].pattern_id == "3"
        assert result[-1].pattern_id == "1"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agentic/test_llm_interface.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/agentic/llm_interface.py
"""LLM interface for semantic reranking.

This module provides an abstract interface for LLM reranking.
The actual LLM integration should follow docs/KEYCYCLE.md.

Includes a MockLLMReranker for testing without API calls.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.memory_store import RetrievalResult


class BaseLLMReranker(ABC):
    """Abstract base for LLM rerankers.

    Implement this interface to integrate with your LLM of choice.
    See docs/KEYCYCLE.md for integration details.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list["RetrievalResult"],
        top_k: int | None = None,
    ) -> list["RetrievalResult"]:
        """Rerank candidates by semantic relevance to query.

        Args:
            query: The user's query.
            candidates: Candidate results from pattern retrieval.
            top_k: Limit results (None = return all).

        Returns:
            Candidates reordered by semantic relevance.
        """
        pass


class MockLLMReranker(BaseLLMReranker):
    """Mock LLM reranker using keyword overlap.

    For testing without API calls. Simulates semantic judgment
    using simple word overlap between query and candidates.

    This is NOT a real LLM - it's a placeholder for testing.
    For real LLM integration, see docs/KEYCYCLE.md.
    """

    def rerank(
        self,
        query: str,
        candidates: list["RetrievalResult"],
        top_k: int | None = None,
    ) -> list["RetrievalResult"]:
        """Rerank using keyword overlap as proxy for semantics."""
        query_words = set(self._tokenize(query.lower()))

        scored = []
        for candidate in candidates:
            text_words = set(self._tokenize(candidate.text.lower()))

            overlap = len(query_words & text_words)
            if query.lower() in candidate.text.lower():
                overlap += 10

            scored.append((candidate, overlap))

        scored.sort(key=lambda x: -x[1])

        result = [c for c, _ in scored]
        if top_k is not None:
            result = result[:top_k]
        return result

    def _tokenize(self, text: str) -> list[str]:
        """Simple word tokenization."""
        return re.findall(r'\b\w+\b', text)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/agentic/test_llm_interface.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agentic/llm_interface.py tests/agentic/test_llm_interface.py
git commit -m "feat(agentic): add LLM-agnostic reranking interface"
```

---

## Phase 7: Agent Memory Integration

### Task 7.1: Create AgentMemory class integrating all components

**Files:**
- Create: `src/agentic/agent_memory.py`
- Test: `tests/agentic/test_agent_memory.py`

(Implementation follows the same pattern - integrates MemoryStore, coactivation, and LLM interface)

---

## Phase 8: Evaluation Framework

### Task 8.1: Create evaluation metrics and test harness

**Files:**
- Create: `src/agentic/evaluation.py`
- Test: `tests/agentic/test_evaluation.py`

(Implementation follows the same pattern)

---

## Success Criteria Checklist

After completing all phases, verify:

| Metric | Target | How to Check |
|--------|--------|--------------|
| All tests pass | 100% | `pytest tests/agentic/ -v` |
| Text encoding works | Lexical overlap creates pattern overlap | Check test_text_encoder.py |
| Long sequences tested | Find degradation threshold | Check test_long_sequences.py |
| Realistic encoding validated | Pass realistic encoding tests | Check test_realistic_encoding.py |
| Semantic separation increases | > 0 at end | Check simulation output |
| Pattern obesity bounded | < 2.0x | Check simulation output |
| Binding integrity maintained | 100% | Existing binding tests still pass |
| LLM integration is modular | Uses docs/KEYCYCLE.md pattern | Code review |

---

## File Summary

```
src/agentic/
├── __init__.py
├── text_encoder.py        # Task 1.1 (Uncertainty #1)
├── evolving_pattern.py    # Task 1.2
├── memory_store.py        # Task 4.1
├── coactivation.py        # Task 5.1
├── llm_interface.py       # Task 6.1
├── agent_memory.py        # Task 7.1
└── evaluation.py          # Task 8.1

tests/agentic/
├── __init__.py
├── test_text_encoder.py         # Task 1.1 (Uncertainty #1)
├── test_evolving_pattern.py     # Task 1.2
├── test_realistic_encoding.py   # Task 2.1 (Uncertainty #2)
├── test_long_sequences.py       # Task 3.1 (Uncertainty #3)
├── test_memory_store.py         # Task 4.1
├── test_coactivation.py         # Task 5.1
├── test_llm_interface.py        # Task 6.1
├── test_agent_memory.py         # Task 7.1
└── test_evaluation.py           # Task 8.1
```

---

## Summary

This plan implements:

1. **Text encoding** (Phase 1) - Converts text to sparse patterns with deterministic hashing
2. **Realistic validation** (Phase 2) - Tests encoding on real-world conversational data
3. **Long sequence testing** (Phase 3) - Finds phase encoding limits at 1000+ items
4. **Memory store** (Phase 4) - Pattern storage with Jaccard and interference retrieval
5. **Coactivation learning** (Phase 5) - Patterns retrieved together become similar over time
6. **LLM interface** (Phase 6) - Mockable interface for semantic reranking
7. **Agent memory** (Phase 7) - High-level API integrating all components
8. **Evaluation** (Phase 8) - Metrics for measuring emergent semantic clustering

**Key design decisions:**
- **Coactivation, not Hebbian** - More accurate terminology for the mechanism
- **Separate `agentic/` package** - Clean separation from core `quantum_substrate/`
- **Only original bits transfer** - Prevents transitive pollution
- **LLM-agnostic interface** - Real LLM integration follows docs/KEYCYCLE.md

NOTE: Summarize the test results and findings for what works and what doesn't, and append it to TEST_SUMMARY.md
