# Coding Conventions

**Analysis Date:** 2026-01-31

## Naming Patterns

**Files:**
- Modules use lowercase with underscores: `agent_memory.py`, `text_encoder.py`, `evolving_pattern.py`
- Test files follow pattern `test_<module>.py`: `test_agent_memory.py`, `test_coactivation.py`

**Classes:**
- PascalCase: `AgentMemory`, `MemoryStore`, `EvolvingPattern`, `BaseLLMReranker`, `TextEncoder`
- Dataclasses follow same convention: `AgentMemoryConfig`, `CoactivationConfig`, `RetrievalResult`
- Abstract base classes prefixed with Base: `BaseLLMReranker`

**Functions:**
- snake_case: `coactivate()`, `semantic_separation()`, `pattern_obesity_stats()`, `bind_hrr()`, `unbind_hrr()`
- Private functions prefixed with underscore: `_tokenize()`, `_retrieve_jaccard()`, `_transfer_bits()`
- Properties use `@property` decorator: `memory_count`, `average_obesity`, `text`, `metadata`

**Variables:**
- snake_case for local variables and parameters: `top_k`, `use_reranking`, `pattern_id`, `memory_id`
- Constants in UPPER_SNAKE_CASE: `SEED = 42` (in conftest.py), `ANIMAL_TEXTS`, `TECH_TEXTS`
- Internal state prefixed with underscore: `self._store`, `self._patterns`

**Types:**
- Type hints used throughout: `str`, `int`, `float`, `list[RetrievalResult]`, `dict[str, Any]`
- Union types use `|` syntax: `dict[str, Any] | None`, `float | None`, `str | None`
- TYPE_CHECKING blocks for circular imports: `from __future__ import annotations` and `if TYPE_CHECKING: from module import Class`

## Code Style

**Formatting:**
- Follows PEP 8 style guidelines
- No explicit linter/formatter config found (no .eslintrc, .prettierrc, or black config)
- Line length appears to follow standard ~88 character convention
- 4-space indentation (standard Python)

**Linting:**
- No linting tool configuration detected
- No type checking tool (mypy) configuration found
- Code assumes runtime type hints via `from __future__ import annotations`

## Import Organization

**Order:**
1. Future imports: `from __future__ import annotations`
2. Standard library: `import sys`, `import uuid`, `import hashlib`, `import math`, `import re`, `import random`
3. Third-party: `import torch`, `import pytest`
4. Local imports: `from agentic.memory_store import MemoryStore`, `from quantum_substrate.binding import bind_hrr`

**Path Aliases:**
- No path aliases configured in pyproject.toml or code
- Imports use absolute paths from src root: `from agentic.memory_store import MemoryStore`
- Tests import from src via pythonpath configuration in pyproject.toml: `pythonpath = ["src"]`

**Example from `src/agentic/agent_memory.py`:**
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from agentic.memory_store import MemoryStore, RetrievalResult
from agentic.coactivation import coactivate, CoactivationConfig
from agentic.evolving_pattern import EvolvingPattern

if TYPE_CHECKING:
    from agentic.llm_interface import BaseLLMReranker
```

## Error Handling

**Patterns:**
- Explicit None returns for missing data: `return self.patterns.get(pattern_id)` returns `None` if not found
- Simple assertions for validation: `assert len(self.indices) == len(self.magnitudes)`
- No exception raising observed in core logic (substrate/agentic layer)
- Graceful empty list returns: `return []` when no matches found in retrieval
- Early returns for empty states: `if not patterns: return` in learning functions

**Example from `src/agentic/agent_memory.py`:**
```python
def get(self, memory_id: str) -> EvolvingPattern | None:
    """Get a memory by ID."""
    return self._store.get(memory_id)  # None if not found

def learn(self, results: list[RetrievalResult], strength: float | None = None) -> None:
    patterns = [r.pattern for r in results]

    if not patterns:  # Early return on empty
        return
```

## Logging

**Framework:** None detected

**Patterns:**
- No logging framework (print, logging module) used in source code
- Docstrings serve as inline documentation
- Module-level docstrings explain purpose and usage patterns
- No debug output or logging statements in core logic

## Comments

**When to Comment:**
- Comments used sparingly; code is self-documenting
- Block comments explain design decisions: "Only ORIGINAL bits are transferred, not acquired bits"
- Comments clarify non-obvious algorithms: FFT-based circular convolution explanation in `bind_hrr()`

**JSDoc/TSDoc:**
- Uses Google-style docstrings for all public functions, classes, and methods
- Includes Args, Returns, and sometimes Attributes sections
- Example from `src/agentic/agent_memory.py`:

```python
def recall(
    self,
    query: str,
    top_k: int = 10,
    use_reranking: bool = False,
    retrieval_top_k: int | None = None,
) -> list[RetrievalResult]:
    """Recall memories relevant to query.

    First retrieves candidates using pattern similarity,
    then optionally reranks using LLM for semantic relevance.

    Args:
        query: Query text.
        top_k: Number of results to return.
        use_reranking: Whether to use LLM reranking.
        retrieval_top_k: Number of candidates for reranking
            (defaults to 2 * top_k if reranking, else top_k).

    Returns:
        List of RetrievalResult sorted by relevance.
    """
```

## Function Design

**Size:** Functions typically 5-30 lines (focused, single responsibility)

**Parameters:**
- Positional-only for required parameters
- Keyword arguments with defaults for optional: `top_k: int = 10`, `use_reranking: bool = False`
- Config objects used for complex parameter sets: `config: AgentMemoryConfig | None = None`
- Example: `store(self, text: str, metadata: dict[str, Any] | None = None, memory_id: str | None = None) -> str`

**Return Values:**
- Single return type per function
- `None` for void functions with side effects: `def learn(self, ...) -> None`
- Union types when multiple: `EvolvingPattern | None`, `list[RetrievalResult]`
- Dataclass return objects for multiple related values: `SemanticSeparationResult`, `ObesityStats`

## Module Design

**Exports:**
- Each module focused on single responsibility
- `src/agentic/agent_memory.py` - high-level API for store/recall/learn
- `src/agentic/memory_store.py` - pattern storage and retrieval
- `src/agentic/coactivation.py` - learning rule
- `src/agentic/text_encoder.py` - text to pattern conversion
- `src/quantum_substrate/binding.py` - HRR operations
- `src/quantum_substrate/__init__.py` exports main public functions

**Barrel Files:**
- Used in `src/quantum_substrate/__init__.py` to expose public API:
  - `from quantum_substrate.patterns import ComplexSparsePattern`
  - `from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity`
- Agentic layer (`src/agentic/__init__.py`) minimal - only docstring

**Example barrel from `src/quantum_substrate/__init__.py`:**
```python
"""Quantum-inspired substrate for memory systems."""

from quantum_substrate.patterns import ComplexSparsePattern
from quantum_substrate.binding import bind_hrr, unbind_hrr, similarity
from quantum_substrate.interference import (
    jaccard_retrieval,
    interference_retrieval,
    create_related_pattern,
)

__version__ = "0.1.0"
__all__ = [
    "ComplexSparsePattern",
    "bind_hrr",
    "unbind_hrr",
    "similarity",
    "jaccard_retrieval",
    "interference_retrieval",
    "create_related_pattern",
]
```

---

*Convention analysis: 2026-01-31*
