# Codebase Structure

**Analysis Date:** 2026-01-31

## Directory Layout

```
engram/
├── src/                        # Source code (Python packages)
│   ├── agentic/                # High-level agentic memory layer
│   │   ├── __init__.py
│   │   ├── agent_memory.py      # Main API: AgentMemory class
│   │   ├── memory_store.py      # Pattern storage and retrieval
│   │   ├── evolving_pattern.py  # Pattern that learns via coactivation
│   │   ├── coactivation.py      # Associative learning rule
│   │   ├── text_encoder.py      # Text-to-pattern encoding
│   │   ├── llm_interface.py     # Abstract LLM reranker interface
│   │   └── evaluation.py        # Evaluation metrics (clustering, silhouette)
│   │
│   └── quantum_substrate/       # Low-level quantum-inspired primitives
│       ├── __init__.py
│       ├── patterns.py          # ComplexSparsePattern representation
│       ├── binding.py           # HRR binding/unbinding (circular convolution)
│       └── interference.py      # Phase-aware retrieval methods
│
├── tests/                       # Test suite mirroring src/ structure
│   ├── agentic/                 # Tests for agentic layer
│   │   ├── conftest.py
│   │   ├── test_agent_memory.py
│   │   ├── test_memory_store.py
│   │   ├── test_evolving_pattern.py
│   │   ├── test_coactivation.py
│   │   ├── test_text_encoder.py
│   │   ├── test_llm_interface.py
│   │   ├── test_evaluation.py
│   │   ├── test_realistic_encoding.py
│   │   └── test_long_sequences.py
│   │
│   └── quantum_substrate/       # Tests for quantum substrate
│       ├── binding/             # HRR binding tests
│       ├── coherence/           # Phase coherence tests
│       ├── integration/         # Multi-layer integration tests
│       ├── interference/        # Interference mechanism tests
│       ├── patterns/            # Sparse pattern tests
│       ├── scale/               # Scalability tests
│       └── sequence/            # Sequential pattern tests
│
├── docs/                        # Documentation
│   ├── plans/                   # Planning documents
│   └── reviews/                 # Review and analysis documents
│
├── .planning/                   # GSD planning artifacts
│   └── codebase/                # Codebase analysis (this location)
│
├── pyproject.toml               # Python package config
└── README.md                    # Project overview
```

## Directory Purposes

**src/agentic/:**
- Purpose: User-facing API for persistent, learnable memory systems with semantic retrieval
- Contains: Agent memory orchestrator, pattern storage, coactivation learning, text encoding, LLM interface
- Key files: `agent_memory.py` (start here), `memory_store.py`, `coactivation.py`

**src/quantum_substrate/:**
- Purpose: Low-level pattern primitives and operations inspired by quantum mechanics
- Contains: Sparse complex-valued patterns, circular convolution binding, phase-aware interference retrieval
- Key files: `patterns.py` (pattern representation), `binding.py` (HRR operations), `interference.py` (retrieval)

**tests/agentic/:**
- Purpose: Unit and integration tests for memory layer components
- Contains: Test classes for each agentic module, fixtures, configuration
- Key files: `test_agent_memory.py` (main API tests), `conftest.py` (fixtures)

**tests/quantum_substrate/:**
- Purpose: Comprehensive tests for quantum primitives and their interactions
- Contains: Tests organized by concern (binding, coherence, interference, patterns)
- Key files: Subdirectories by feature area

**docs/:**
- Purpose: Design decisions, planning, and analysis
- Contains: Architecture notes, phase cycle documentation, review documents

**.planning/codebase/:**
- Purpose: GSD codebase analysis artifacts
- Contains: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, etc.

## Key File Locations

**Entry Points:**
- `src/agentic/agent_memory.py`: Main AgentMemory class for application use
- `src/quantum_substrate/__init__.py`: Exports public API (ComplexSparsePattern, bind_hrr, etc.)

**Configuration:**
- `pyproject.toml`: Project metadata, dependencies (torch), test paths, pytest options

**Core Logic:**
- `src/agentic/agent_memory.py`: Orchestration of store, recall, learn
- `src/agentic/memory_store.py`: Pattern storage dict and retrieval logic (Jaccard vs. interference)
- `src/agentic/coactivation.py`: Bit transfer learning rule and obesity decay
- `src/agentic/text_encoder.py`: Token-to-bit hashing and phase encoding
- `src/quantum_substrate/patterns.py`: ComplexSparsePattern dataclass and conversions
- `src/quantum_substrate/binding.py`: HRR circular convolution and correlation
- `src/quantum_substrate/interference.py`: Phase-aware similarity metrics

**Testing:**
- `tests/agentic/test_agent_memory.py`: High-level API tests
- `tests/agentic/conftest.py`: Pytest fixtures (fixture_memory, fixture_patterns)
- `tests/quantum_substrate/binding/test_hrr_basics.py`: HRR binding verification

## Naming Conventions

**Files:**
- Python modules: lowercase with underscores (agent_memory.py, evolving_pattern.py)
- Test files: test_{module_name}.py matching source structure
- Config files: pyproject.toml, .gitignore, etc.

**Directories:**
- Package names: lowercase, descriptive (agentic, quantum_substrate)
- Test subdirectories: mirror src/ structure (tests/agentic/, tests/quantum_substrate/)
- Organizational folders: concern-based (binding, coherence, patterns)

**Python Identifiers:**
- Classes: PascalCase (AgentMemory, EvolvingPattern, ComplexSparsePattern)
- Functions: snake_case (coactivate, jaccard_retrieval, bind_hrr)
- Module-level constants: UPPERCASE (rarely used)
- Private functions: _leading_underscore (_transfer_bits, _tokenize)
- Type hints: Full type annotations with modern Python typing (Union → | syntax)

## Where to Add New Code

**New Feature:**
- Primary code: Add to appropriate module in `src/agentic/` or `src/quantum_substrate/`
- Tests: Create corresponding test file in `tests/agentic/` or `tests/quantum_substrate/`
- Example: Adding a new retrieval method → `src/quantum_substrate/interference.py` + `tests/quantum_substrate/interference/test_new_method.py`

**New Component/Module:**
- Implementation: Create new .py file in `src/agentic/` or `src/quantum_substrate/`
- Update __init__.py: Export public API from package __init__.py
- Tests: Create test_{module_name}.py in corresponding tests/ directory
- Example: New LLM backend → `src/agentic/my_llm.py` implementing BaseLLMReranker + `tests/agentic/test_my_llm.py`

**Utilities:**
- Shared helpers: Define in the module where they're primarily used; use _leading_underscore for internal helpers
- Cross-module utils: Create shared utility module (e.g., src/agentic/utils.py) if needed
- Note: Currently minimal shared utilities; favor module-internal functions

**New Layer/Integration:**
- Location: Consider new subdirectory in src/ if it represents a distinct architectural layer
- Example: Adding schema/database layer → `src/persistence/` with store implementations

## Special Directories

**src/agentic/__pycache__/:**
- Purpose: Python bytecode cache (auto-generated)
- Generated: Yes
- Committed: No (gitignored)

**src/engram.egg-info/:**
- Purpose: Package installation metadata (created by pip install -e .)
- Generated: Yes
- Committed: No (gitignored)

**tests/quantum_substrate/{binding,coherence,interference,patterns,scale,sequence}/:**
- Purpose: Test subdirectories organizing tests by concern/feature area
- Generated: No
- Committed: Yes

**.planning/codebase/:**
- Purpose: GSD codebase analysis documents
- Generated: No (manually created by GSD mappers)
- Committed: Yes (to preserve analysis across runs)

## Package Import Structure

**From agentic:**
```python
from agentic import AgentMemory, AgentMemoryConfig  # Top-level exports
from agentic.memory_store import MemoryStore, RetrievalResult
from agentic.evolving_pattern import EvolvingPattern
from agentic.coactivation import coactivate, CoactivationConfig
from agentic.text_encoder import TextEncoder, EncodedPattern
from agentic.llm_interface import BaseLLMReranker, MockLLMReranker
```

**From quantum_substrate:**
```python
from quantum_substrate import ComplexSparsePattern, bind_hrr, unbind_hrr, similarity
from quantum_substrate.interference import jaccard_retrieval, interference_retrieval, create_related_pattern
```

## Build and Distribution

**Source discovery:** `pyproject.toml` with `[tool.setuptools.packages.find]` set to `where = ["src"]`

**Install:** `pip install -e .` (editable/development mode) or `pip install .` (normal install)

**Dependencies:** Only torch (>=2.0.0); no other runtime dependencies

**Dev dependencies:** pytest (>=7.0.0) optional, installed via `pip install -e ".[dev]"`

---

*Structure analysis: 2026-01-31*
