# Testing Patterns

**Analysis Date:** 2026-01-31

## Test Framework

**Runner:**
- pytest >= 7.0.0
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`
- Test discovery: `testpaths = ["tests"]`

**Assertion Library:**
- pytest built-in assertions (no pytest-extra or custom assertion library)
- Simple assert statements: `assert memory_id is not None`, `assert isinstance(result, RetrievalResult)`

**Run Commands:**
```bash
pytest                          # Run all tests
pytest tests/                   # Run specific directory
pytest -v                       # Verbose output (enabled by default via addopts)
pytest tests/agentic/           # Run module-specific tests
pytest tests/agentic/test_agent_memory.py::TestAgentMemoryStore::test_store_text_returns_id  # Single test
```

## Test File Organization

**Location:**
- Mirrored structure to src: `tests/` mirrors `src/` package layout
- `tests/agentic/test_agent_memory.py` tests `src/agentic/agent_memory.py`
- `tests/quantum_substrate/binding/test_hrr_basics.py` tests `src/quantum_substrate/binding.py`

**Naming:**
- Test files: `test_<module>.py` (not `<module>_test.py`)
- Test classes: `Test<Feature>` e.g., `TestAgentMemoryStore`, `TestCoactivationBasics`, `TestSemanticSeparation`
- Test methods: `test_<behavior_or_scenario>` e.g., `test_store_text_returns_id`, `test_recall_finds_exact_match`

**Structure:**
```
tests/
├── agentic/
│   ├── conftest.py
│   ├── test_agent_memory.py
│   ├── test_coactivation.py
│   ├── test_evaluation.py
│   ├── test_evolving_pattern.py
│   ├── test_llm_interface.py
│   ├── test_memory_store.py
│   ├── test_text_encoder.py
│   └── test_long_sequences.py
├── quantum_substrate/
│   ├── conftest.py
│   ├── binding/
│   │   ├── conftest.py
│   │   ├── test_hrr_basics.py
│   │   ├── test_sparse_binding.py
│   │   └── test_superposition_limits.py
│   ├── coherence/
│   │   ├── __init__.py
│   │   └── test_decay.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_agentic_memory.py
│   └── interference/
│       └── test_basics.py
```

## Test Structure

**Suite Organization:**
```python
# From tests/agentic/test_agent_memory.py
class TestAgentMemoryStore:
    """Test storing memories."""

    def test_store_text_returns_id(self):
        """Storing text returns a memory ID."""
        memory = AgentMemory(dim=1024, k=50)

        memory_id = memory.store("Hello world")

        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0
```

**Patterns:**
- One class per major feature/function: `TestAgentMemoryStore`, `TestAgentMemoryRecall`, `TestAgentMemoryLearn`
- Each test is independent and self-contained
- Setup uses constructor calls in test body (no setUp method)
- No tearDown methods needed - tests don't require cleanup

**Arrange-Act-Assert pattern:**
```python
def test_learn_applies_coactivation(self):
    """Learn applies coactivation to retrieved patterns."""
    # ARRANGE
    memory = AgentMemory(dim=1024, k=50)
    id1 = memory.store("dogs are animals")
    id2 = memory.store("wolves are animals")
    p1_before = set(memory.get(id1).bits)
    p2_before = set(memory.get(id2).bits)
    initial_overlap = len(p1_before & p2_before)

    # ACT
    for _ in range(10):
        results = memory.recall("animals", top_k=2)
        memory.learn(results, strength=0.05)

    # ASSERT
    p1_after = memory.get(id1).bits
    p2_after = memory.get(id2).bits
    final_overlap = len(p1_after & p2_after)
    assert final_overlap > initial_overlap
```

## Mocking

**Framework:** No external mocking framework

**Patterns:**
- Concrete mock implementations inherit from abstract base: `MockLLMReranker(BaseLLMReranker)`
- Example from `src/agentic/llm_interface.py`:

```python
class MockLLMReranker(BaseLLMReranker):
    """Mock LLM reranker using keyword overlap.

    For testing without API calls. Simulates semantic judgment
    using simple word overlap between query and candidates.
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
```

- Inline custom implementations for test-specific behavior:

```python
def test_custom_reranker(self):
    """Can use custom reranker implementation."""
    class TopicReranker(BaseLLMReranker):
        """Reranker that boosts specific topic."""
        def __init__(self, boost_topic: str):
            self.boost_topic = boost_topic

        def rerank(self, query, candidates, top_k=None):
            scored = []
            for c in candidates:
                score = 10 if self.boost_topic in c.text.lower() else 0
                scored.append((c, score))
            scored.sort(key=lambda x: -x[1])
            result = [c for c, _ in scored]
            return result[:top_k] if top_k else result

    memory = AgentMemory(dim=1024, k=50, reranker=TopicReranker("dogs"))
    # Test follows...
```

**What to Mock:**
- External service interfaces: `BaseLLMReranker` - use `MockLLMReranker` for testing
- Nothing else is mocked - prefer real implementations where possible

**What NOT to Mock:**
- Core data structures: always use real `AgentMemory`, `MemoryStore`, `EvolvingPattern`
- Pattern operations: always use real encoding and retrieval
- Learning algorithms: always use real coactivation - testing its behavior is the point

## Fixtures and Factories

**Test Data:**
```python
# From tests/agentic/test_evaluation.py
ANIMAL_TEXTS = [
    "dogs are loyal pets that love their owners",
    "wolves hunt in packs at night",
    "cats are independent and self-sufficient",
    "horses run fast across open fields",
]

TECH_TEXTS = [
    "machine learning requires lots of training data",
    "neural networks can recognize images accurately",
    "databases store information efficiently",
    "algorithms process data in sorted order",
]

FOOD_TEXTS = [
    "pizza is delicious with extra cheese",
    "sushi requires fresh fish and rice",
    "bread needs time to rise properly",
    "soup is best served hot in winter",
]
```

**Fixtures Location:**
- Shared fixtures in `tests/conftest.py` (root)
- Module-specific fixtures in `tests/<module>/conftest.py`
- Example from `tests/quantum_substrate/conftest.py`:

```python
"""Shared fixtures for quantum substrate tests."""

import pytest
import torch

SEED = 42

@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return torch.Generator().manual_seed(SEED)

@pytest.fixture
def dim():
    """Default pattern dimensionality."""
    return 1024

@pytest.fixture
def sparsity():
    """Default sparsity (number of active dimensions)."""
    return 50  # k=50 active of dim=1024
```

- Specialized fixtures in `tests/quantum_substrate/binding/conftest.py`:

```python
"""Binding-specific fixtures."""

import torch
import pytest

@pytest.fixture
def role_set(rng, dim):
    """Standard set of role vectors for testing."""
    return {
        "agent": torch.randn(dim, dtype=torch.complex64, generator=rng),
        "patient": torch.randn(dim, dtype=torch.complex64, generator=rng),
        "instrument": torch.randn(dim, dtype=torch.complex64, generator=rng),
        "location": torch.randn(dim, dtype=torch.complex64, generator=rng),
        "time": torch.randn(dim, dtype=torch.complex64, generator=rng),
    }
```

## Coverage

**Requirements:** No coverage requirement enforced

**View Coverage:** Not configured

## Test Types

**Unit Tests:**
- Dominant test type in codebase
- Test individual functions and classes in isolation
- Example: `test_store_text_returns_id()` tests only `AgentMemory.store()` behavior
- Located in `tests/agentic/test_*.py` and `tests/quantum_substrate/*/test_*.py`

**Integration Tests:**
- Full workflow tests combining multiple components
- Example from `tests/agentic/test_agent_memory.py`:

```python
class TestAgentMemoryWorkflow:
    """Test full workflow: store -> recall -> learn."""

    def test_full_workflow(self):
        """Full workflow improves retrieval over time."""
        memory = AgentMemory(dim=1024, k=50, reranker=MockLLMReranker())

        # Store memories about dogs and cats
        memory.store("dogs are loyal pets")
        memory.store("dogs love to play fetch")
        memory.store("cats are independent")
        memory.store("cats like to sleep")

        # Simulate multiple interactions about dogs
        for _ in range(5):
            results = memory.recall("tell me about dogs", top_k=2, use_reranking=True)
            memory.learn(results, strength=0.1)

        # Verify behavior
        all_patterns = memory.get_all_patterns()
        assert len(all_patterns) == 4
```

**E2E Tests:**
- Not present - no external services to test end-to-end against

## Common Patterns

**Async Testing:**
- Not applicable - no async code in codebase

**Error Testing:**
- Tests for None returns rather than exceptions:

```python
def test_get_missing_pattern(self):
    """Get returns None for missing pattern."""
    store = MemoryStore(dim=1024, k=50)
    result = store.get("nonexistent-id")
    assert result is None
```

- Tests for empty returns on invalid input:

```python
def test_recall_empty_store(self):
    """Recall on empty store returns empty list."""
    memory = AgentMemory(dim=1024, k=50)
    results = memory.recall("anything", top_k=5)
    assert len(results) == 0
```

**Determinism Testing:**
- Tests verify reproducible behavior:

```python
# From tests/agentic/test_text_encoder.py
def test_same_text_same_pattern(self):
    """Identical text produces identical patterns."""
    encoder = TextEncoder(dim=1024, k=50)

    p1 = encoder.encode("hello world")
    p2 = encoder.encode("hello world")

    assert p1.bits == p2.bits
    assert p1.phases == p2.phases
```

**Configuration Testing:**
- Tests verify config options work correctly:

```python
def test_custom_config(self):
    """Can use custom config."""
    config = AgentMemoryConfig(
        retrieval_method="interference",
        use_evolved_bits=True,
        coactivation=CoactivationConfig(max_bits=100),
    )
    memory = AgentMemory(dim=2048, k=100, config=config)

    assert memory.dim == 2048
    assert memory.k == 100
```

---

*Testing analysis: 2026-01-31*
